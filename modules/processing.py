from PIL import Image, ImageFilter
import logging
import torch
import math
from nodes import common_ksampler, VAEEncode, VAEDecode, VAEDecodeTiled
from comfy_extras.nodes_custom_sampler import SamplerCustom
from usdu_utils import pil_to_tensor, tensor_to_pil, get_crop_region, expand_crop, crop_cond
from modules import shared
from tqdm import tqdm
import comfy.utils as comfy_utils
import comfy.sample
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import json
import os

logger = logging.getLogger(__name__)

# Compatibility for older Pillow versions
try:
    Image.Resampling  # type: ignore
except Exception:
    Image.Resampling = Image  # type: ignore

# Taken from the USDU script
class USDUMode(Enum):
    LINEAR = 0
    CHESS = 1
    NONE = 2

class USDUSFMode(Enum):
    NONE = 0
    BAND_PASS = 1
    HALF_TILE = 2
    HALF_TILE_PLUS_INTERSECTIONS = 3


class TileOverlapMode(Enum):
    """Modes for handling tile overlap during upscaling."""
    IGNORE = 0       # Current uniform_tile_mode=False behavior - minimal tile sizes
    REPROCESS = 1    # Current uniform_tile_mode=True behavior - uniform tiles, overlap may be regenerated
    CONTEXT_ONLY = 2 # New mode - overlap regions are context only, not re-denoised


@dataclass
class ProcessedRegion:
    """Rectangular region that has been denoised."""
    x1: int
    y1: int
    x2: int  # exclusive
    y2: int  # exclusive


class ProcessedRegionTracker:
    """Tracks which regions have been denoised to avoid reprocessing in CONTEXT_ONLY mode."""

    def __init__(self):
        self.regions: List[ProcessedRegion] = []

    def add_region(self, x1: int, y1: int, x2: int, y2: int):
        """Record a region as processed."""
        self.regions.append(ProcessedRegion(x1, y1, x2, y2))

    def get_exclusion_mask(self, tile_x1: int, tile_y1: int,
                           tile_x2: int, tile_y2: int,
                           full_width: int, full_height: int) -> Image.Image:
        """
        Create a full-size mask where:
        - 255 (white) = needs denoising
        - 0 (black) = already processed, context only

        Returns mask covering just the tile region, positioned for the full image.
        """
        width = tile_x2 - tile_x1
        height = tile_y2 - tile_y1

        # Start with all white (everything needs denoising)
        tile_mask = np.ones((height, width), dtype=np.uint8) * 255

        for region in self.regions:
            # Calculate intersection in tile-local coordinates
            ix1 = max(region.x1 - tile_x1, 0)
            iy1 = max(region.y1 - tile_y1, 0)
            ix2 = min(region.x2 - tile_x1, width)
            iy2 = min(region.y2 - tile_y1, height)

            if ix1 < ix2 and iy1 < iy2:
                tile_mask[iy1:iy2, ix1:ix2] = 0  # Mark as already processed

        # Create full-size mask
        full_mask = Image.new("L", (full_width, full_height), 0)
        tile_mask_pil = Image.fromarray(tile_mask, mode='L')
        full_mask.paste(tile_mask_pil, (tile_x1, tile_y1))

        return full_mask


def get_region_mask_for_canvas(p, canvas_size):
    """Return p.region_mask resized to canvas_size ('L'), cached on p by size.

    Returns None when no region mask is set (including non-guider processing).
    canvas_size must be image_mask.size -- p.width/p.height hold the PER-TILE
    processing size inside process_images, NOT the canvas size.
    """
    region_mask = getattr(p, 'region_mask', None)
    if region_mask is None:
        return None
    cache = getattr(p, '_region_mask_cache', None)
    if cache is None:
        cache = p._region_mask_cache = {}
    if canvas_size not in cache:
        if region_mask.size == canvas_size:
            cache[canvas_size] = region_mask
        else:
            # BILINEAR deliberately: monotone, no LANCZOS ringing on hard
            # edges; mask_blur feathers afterwards.
            cache[canvas_size] = region_mask.resize(canvas_size, Image.Resampling.BILINEAR)
    return cache[canvas_size]


class StableDiffusionProcessing:

    def __init__(
        self,
        init_img,
        model,
        positive,
        negative,
        vae,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        upscale_by,
        tile_overlap_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        custom_sampler=None,
        custom_sigmas=None,
        batch_size=1,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = Image.new('L', init_img.size, 0)  # Placeholder mask
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # ComfyUI Sampler inputs
        self.model = model
        self.positive = positive
        self.negative = negative
        self.vae = vae
        self.seed = seed
        self.steps = steps
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

        # Optional custom sampler and sigmas
        self.custom_sampler = custom_sampler
        self.custom_sigmas = custom_sigmas

        if (custom_sampler is not None) ^ (custom_sigmas is not None):
            logger.warning("Both custom sampler and custom sigmas must be provided, defaulting to widget sampler and sigmas")

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        # Handle tile_overlap_mode - default to REPROCESS for backward compatibility
        if tile_overlap_mode is None:
            self.tile_overlap_mode = TileOverlapMode.REPROCESS
        else:
            self.tile_overlap_mode = tile_overlap_mode
        # Tracker for CONTEXT_ONLY mode (initialized by tile loop)
        self.processed_tracker: Optional[ProcessedRegionTracker] = None
        self.tiled_decode = tiled_decode
        self.batch_size = batch_size
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            logger.info("Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.pbar: Optional[tqdm] = None
        self.tiles = 0
        self.progress_bar_enabled = False
        if comfy_utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy_utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)

    @property
    def uniform_tile_mode(self):
        """Derived property for backward compatibility with _prepare_tile_for_batch()."""
        return self.tile_overlap_mode != TileOverlapMode.IGNORE

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy_utils.PROGRESS_BAR_ENABLED = True


class StableDiffusionProcessingGuider:
    """
    Processing class for guider-based sampling.
    Similar to StableDiffusionProcessing but uses a GUIDER instead of
    separate model, positive, negative, and cfg inputs.
    """

    def __init__(
        self,
        init_img,
        guider,
        sampler,
        sigmas,
        vae,
        seed,
        upscale_by,
        tile_overlap_mode,
        tiled_decode,
        tile_width,
        tile_height,
        redraw_mode,
        seam_fix_mode,
        batch_size=1,
        region_mask=None,
        anchor_context=False,
    ):
        # Variables used by the USDU script
        self.init_images = [init_img]
        self.image_mask = Image.new('L', init_img.size, 0)  # Placeholder mask
        self.mask_blur = 0
        self.inpaint_full_res_padding = 0
        self.width = init_img.width * upscale_by
        self.height = init_img.height * upscale_by
        self.rows = round(self.height / tile_height)
        self.cols = round(self.width / tile_width)

        # Guider-based sampling inputs
        self.guider = guider
        self.sampler = sampler
        self.sigmas = sigmas
        self.vae = vae
        self.seed = seed

        # Mark this as guider-based processing
        self.use_guider = True

        # Not used in guider path but kept for compatibility
        self.model = None
        self.positive = None
        self.negative = None
        self.cfg = None
        self.steps = None
        self.sampler_name = None
        self.scheduler = None
        self.denoise = None
        self.custom_sampler = None
        self.custom_sigmas = None

        # Variables used only by this script
        self.init_size = init_img.width, init_img.height
        self.upscale_by = upscale_by
        # Handle tile_overlap_mode - default to REPROCESS for backward compatibility
        if tile_overlap_mode is None:
            self.tile_overlap_mode = TileOverlapMode.REPROCESS
        else:
            self.tile_overlap_mode = tile_overlap_mode
        # Tracker for CONTEXT_ONLY mode (initialized by tile loop)
        self.processed_tracker: Optional[ProcessedRegionTracker] = None
        self.tiled_decode = tiled_decode
        self.batch_size = batch_size
        # Optional region mask (PIL 'L' image) limiting what is composited back
        self.region_mask = region_mask
        self._region_mask_cache = {}
        # Optional inpaint-style anchoring of non-composited tile areas
        self.anchor_context = anchor_context
        self.vae_decoder = VAEDecode()
        self.vae_encoder = VAEEncode()
        self.vae_decoder_tiled = VAEDecodeTiled()

        if self.tiled_decode:
            logger.info("Using tiled decode")

        # Other required A1111 variables for the USDU script that is currently unused in this script
        self.extra_generation_params = {}

        # Load config file for USDU
        config_path = os.path.join(os.path.dirname(__file__), os.pardir, 'config.json')
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Progress bar for the entire process instead of per tile
        self.pbar: Optional[tqdm] = None
        self.tiles = 0
        self.progress_bar_enabled = False
        if comfy_utils.PROGRESS_BAR_ENABLED:
            self.progress_bar_enabled = True
            comfy_utils.PROGRESS_BAR_ENABLED = config.get('per_tile_progress', True)
            if redraw_mode.value != USDUMode.NONE.value:
                self.tiles += self.rows * self.cols
            if seam_fix_mode.value == USDUSFMode.BAND_PASS.value:
                self.tiles += (self.rows - 1) + (self.cols - 1)
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows
            elif seam_fix_mode.value == USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS.value:
                self.tiles += (self.rows - 1) * self.cols + (self.cols - 1) * self.rows + (self.rows - 1) * (self.cols - 1)

    @property
    def uniform_tile_mode(self):
        """Derived property for backward compatibility with _prepare_tile_for_batch()."""
        return self.tile_overlap_mode != TileOverlapMode.IGNORE

    def __del__(self):
        # Undo changes to progress bar flag when node is done or cancelled
        if self.progress_bar_enabled:
            comfy_utils.PROGRESS_BAR_ENABLED = True


class Processed:

    def __init__(self, p: StableDiffusionProcessing, images: list, seed: int, info: str):
        self.images = images
        self.seed = seed
        self.info = info

    def infotext(self, p: StableDiffusionProcessing, index):
        return None


def fix_seed(p: StableDiffusionProcessing):
    pass


def sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise, custom_sampler, custom_sigmas):
    """Choose the way to sample based on given inputs."""

    # Custom sampler and sigmas
    if custom_sampler is not None and custom_sigmas is not None:
        kwargs = dict(
            model=model,
            add_noise=True,
            noise_seed=seed,
            cfg=cfg,
            positive=positive,
            negative=negative,
            sampler=custom_sampler,
            sigmas=custom_sigmas,
            latent_image=latent
        )
        if hasattr(SamplerCustom, "execute"):
            (samples, _) = SamplerCustom.execute(**kwargs)
        else:
            custom_sample = SamplerCustom()
            (samples, _) = getattr(custom_sample, custom_sample.FUNCTION)(**kwargs)
        return samples

    # Default
    (samples,) = common_ksampler(model, seed, steps, cfg, sampler_name,
                                 scheduler, positive, negative, latent, denoise=denoise)
    return samples


def sample_with_guider(guider, seed, sampler, sigmas, latent):
    """
    Sample using a guider instead of separate model/conditioning/cfg.

    Args:
        guider: A GUIDER object that encapsulates model, conditioning, and CFG
        seed: Random seed for noise generation
        sampler: A SAMPLER object (from KSamplerSelect node)
        sigmas: A SIGMAS tensor (noise schedule from BasicScheduler, etc.)
        latent: Latent image dict with 'samples' key

    Returns:
        Latent samples dict
    """
    # Generate noise from seed
    latent_image = latent["samples"]
    batch_inds = latent.get("batch_index", None)

    # Generate noise using ComfyUI's prepare_noise
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    # Get noise mask if present
    noise_mask = latent.get("noise_mask", None)

    # Call guider.sample() - the guider handles the denoising loop
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask=noise_mask,
        callback=None,
        disable_pbar=not comfy_utils.PROGRESS_BAR_ENABLED,
        seed=seed
    )

    # Return in latent dict format
    out = latent.copy()
    out["samples"] = samples
    return out


def process_images(p: StableDiffusionProcessing) -> Processed:
    # Where the main image generation happens in A1111

    # Show the progress bar
    if p.progress_bar_enabled and p.pbar is None:
        p.pbar = tqdm(total=p.tiles, desc='USDU', unit='tile')

    # Setup
    image_mask = p.image_mask.convert('L')
    init_image = p.init_images[0]

    # Locate the white region of the mask outlining the tile and add padding
    crop_region = get_crop_region(image_mask, p.inpaint_full_res_padding)

    # Get the mask bounding box (the white rectangle drawn by tile loop)
    mask_bbox = image_mask.getbbox()
    if mask_bbox is None:
        # No white pixels, nothing to process
        return Processed(p, [], p.seed, "")

    # Optional region mask: clip the tile mask to the user region (composite-only).
    # crop_region was already computed from the full tile rect above, so the
    # sampled tile context is unchanged; the region only limits what is
    # composited back (and lets non-intersecting tiles be skipped entirely).
    region_mask = get_region_mask_for_canvas(p, image_mask.size)
    if region_mask is not None:
        combined = np.minimum(np.array(image_mask), np.array(region_mask))
        if not combined.any():
            # Tile does not touch the region: skip VAE encode + sampling,
            # but still advance the progress bar so it completes.
            if p.progress_bar_enabled and p.pbar is not None:
                p.pbar.update(1)
            return Processed(p, [], p.seed, "")
        image_mask = Image.fromarray(combined, mode='L')

    if p.tile_overlap_mode == TileOverlapMode.IGNORE:
        # Current uniform_tile_mode=False behavior
        # Uses the minimal size that can fit the mask, minimizes tile size but may lead to image sizes that the model is not trained on
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        target_width = math.ceil(crop_width / 8) * 8
        target_height = math.ceil(crop_height / 8) * 8
        crop_region, tile_size = expand_crop(crop_region, image_mask.width,
                                             image_mask.height, target_width, target_height)

    elif p.tile_overlap_mode == TileOverlapMode.REPROCESS:
        # Current uniform_tile_mode=True behavior
        # Expand the crop region to match the processing size ratio and then resize it to the processing size
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height

    elif p.tile_overlap_mode == TileOverlapMode.CONTEXT_ONLY:
        # NEW: Context-only overlap mode
        # Use uniform tile size (like REPROCESS)
        x1, y1, x2, y2 = crop_region
        crop_width = x2 - x1
        crop_height = y2 - y1
        crop_ratio = crop_width / crop_height
        p_ratio = p.width / p.height
        if crop_ratio > p_ratio:
            target_width = crop_width
            target_height = round(crop_width / p_ratio)
        else:
            target_width = round(crop_height * p_ratio)
            target_height = crop_height
        crop_region, _ = expand_crop(crop_region, image_mask.width, image_mask.height, target_width, target_height)
        tile_size = p.width, p.height

        # Apply exclusion mask if tracker exists
        if p.processed_tracker is not None:
            # Get the extended mask region (original mask bbox)
            # The mask was already drawn with overlap extension by the tile loop
            extended_x1, extended_y1, extended_x2, extended_y2 = mask_bbox

            # Create exclusion mask (black where already processed)
            exclusion_mask = p.processed_tracker.get_exclusion_mask(
                extended_x1, extended_y1, extended_x2, extended_y2,
                image_mask.width, image_mask.height
            )

            # Apply exclusion: where exclusion is black (0), set image_mask to black
            # This is done by taking the minimum of both masks
            mask_array = np.array(image_mask)
            exclusion_array = np.array(exclusion_mask)
            combined = np.minimum(mask_array, exclusion_array)
            image_mask = Image.fromarray(combined, mode='L')

            # Record what we're about to denoise (after exclusions)
            # Find the actual bbox of what remains white
            final_bbox = image_mask.getbbox()
            if final_bbox:
                p.processed_tracker.add_region(*final_bbox)
            elif region_mask is not None:
                # Region-white area lies entirely inside already-processed
                # overlap: nothing would composite; skip the sample.
                if p.progress_bar_enabled and p.pbar is not None:
                    p.pbar.update(1)
                return Processed(p, [], p.seed, "")

    # Blur the mask
    if p.mask_blur > 0:
        image_mask = image_mask.filter(ImageFilter.GaussianBlur(p.mask_blur))

    # Crop the images to get the tiles that will be used for generation
    tiles = [img.crop(crop_region) for img in shared.batch]

    # Assume the same size for all images in the batch
    initial_tile_size = tiles[0].size

    # Resize if necessary
    for i, tile in enumerate(tiles):
        if tile.size != tile_size:
            tiles[i] = tile.resize(tile_size, Image.Resampling.LANCZOS)

    # Encode the image
    batched_tiles = torch.cat([pil_to_tensor(tile) for tile in tiles], dim=0)
    (latent,) = p.vae_encoder.encode(p.vae, batched_tiles)

    # Optional inpaint-style anchoring: give the sampler a denoise mask so the
    # non-composited parts of the tile (masked-out areas, CONTEXT_ONLY
    # already-processed overlap, and the tile-padding ring) are renoised from
    # the original latent at every step instead of freely redrawn and discarded.
    if getattr(p, 'anchor_context', False) and (
            region_mask is not None
            or p.tile_overlap_mode == TileOverlapMode.CONTEXT_ONLY):
        noise_mask = image_mask.crop(crop_region)
        if noise_mask.size != tile_size:
            # BILINEAR deliberately: monotone, stays in [0, 1]; LANCZOS (used
            # for the image tile) rings past the range on hard mask edges.
            noise_mask = noise_mask.resize(tile_size, Image.Resampling.BILINEAR)
        mask_tensor = torch.from_numpy(np.array(noise_mask).astype(np.float32) / 255.0)
        # Same shape convention as ComfyUI's SetLatentNoiseMask: [B, 1, H, W]
        latent["noise_mask"] = mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1]))

    # Generate samples - use guider path or standard path
    if getattr(p, 'use_guider', False):
        # Guider path: conditioning is internal to the guider, skip crop_cond
        samples = sample_with_guider(p.guider, p.seed, p.sampler, p.sigmas, latent)
    else:
        # Standard path: crop conditioning for each tile
        positive_cropped = crop_cond(p.positive, crop_region, p.init_size, init_image.size, tile_size)
        negative_cropped = crop_cond(p.negative, crop_region, p.init_size, init_image.size, tile_size)
        samples = sample(p.model, p.seed, p.steps, p.cfg, p.sampler_name, p.scheduler, positive_cropped,
                         negative_cropped, latent, p.denoise, p.custom_sampler, p.custom_sigmas)

    # Update the progress bar
    if p.progress_bar_enabled and p.pbar is not None:
        p.pbar.update(1)

    # Decode the sample
    if not p.tiled_decode:
        (decoded,) = p.vae_decoder.decode(p.vae, samples)
    else:
        (decoded,) = p.vae_decoder_tiled.decode(p.vae, samples, 512)  # Default tile size is 512

    # Convert the sample to a PIL image
    tiles_sampled = [tensor_to_pil(decoded, i) for i in range(len(decoded))]

    for i, tile_sampled in enumerate(tiles_sampled):
        init_image = shared.batch[i]

        # Resize back to the original size
        if tile_sampled.size != initial_tile_size:
            tile_sampled = tile_sampled.resize(initial_tile_size, Image.Resampling.LANCZOS)

        # Put the tile into position
        image_tile_only = Image.new('RGBA', init_image.size)
        image_tile_only.paste(tile_sampled, crop_region[:2])

        # Add the mask as an alpha channel
        # Must make a copy due to the possibility of an edge becoming black
        temp = image_tile_only.copy()
        temp.putalpha(image_mask)
        image_tile_only.paste(temp, image_tile_only)

        # Add back the tile to the initial image according to the mask in the alpha channel
        result = init_image.convert('RGBA')
        result.alpha_composite(image_tile_only)

        # Convert back to RGB
        result = result.convert('RGB')

        shared.batch[i] = result

    processed = Processed(p, [shared.batch[0]], p.seed, "")
    return processed
