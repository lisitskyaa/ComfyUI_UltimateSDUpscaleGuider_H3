# ComfyUI Node for Ultimate SD Upscale by Coyote-A: https://github.com/Coyote-A/ultimate-upscale-for-automatic1111

import logging
from contextlib import contextmanager
import torch
import comfy
import comfy.utils as comfy_utils
from usdu_patch import usdu
from usdu_utils import tensor_to_pil, pil_to_tensor, mask_tensor_to_pil
from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingGuider, TileOverlapMode
import modules.shared as shared
from modules.upscaler import UpscalerData

logger = logging.getLogger(__name__)


@contextmanager
def suppress_logging(level=logging.CRITICAL + 1):
    """Context manager to temporarily suppress logging output."""
    root_logger = logging.getLogger()
    old_level = root_logger.getEffectiveLevel()
    root_logger.setLevel(level)
    try:
        yield
    finally:
        root_logger.setLevel(old_level)

MAX_RESOLUTION = 8192
# The modes available for Ultimate SD Upscale
MODES = {
    "Linear": usdu.USDUMode.LINEAR,
    "Chess": usdu.USDUMode.CHESS,
    "None": usdu.USDUMode.NONE,
}
# The seam fix modes
SEAM_FIX_MODES = {
    "None": usdu.USDUSFMode.NONE,
    "Band Pass": usdu.USDUSFMode.BAND_PASS,
    "Half Tile": usdu.USDUSFMode.HALF_TILE,
    "Half Tile + Intersections": usdu.USDUSFMode.HALF_TILE_PLUS_INTERSECTIONS,
}
# Tile overlap modes mapping
TILE_OVERLAP_MODES = {
    "Ignore Overlap": TileOverlapMode.IGNORE,
    "Reprocess Overlap": TileOverlapMode.REPROCESS,
    "Context Only Overlap": TileOverlapMode.CONTEXT_ONLY,
}


def USDU_base_inputs():
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Sampling Params
        ("model", ("MODEL", {"tooltip": "The model to use for image-to-image."})),
        ("positive", ("CONDITIONING", {"tooltip": "The positive conditioning for each tile."})),
        ("negative", ("CONDITIONING", {"tooltip": "The negative conditioning for each tile."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for image-to-image."})),
        ("steps", ("INT", {"default": 20, "min": 1, "max": 10000, "step": 1, "tooltip": "The number of steps to use for each tile."})),
        ("cfg", ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "The CFG scale to use for each tile."})),
        ("sampler_name", (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The sampler to use for each tile."})),
        ("scheduler", (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler to use for each tile."})),
        ("denoise", ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for each tile."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("force_uniform_tiles", ("BOOLEAN", {"default": True, "tooltip": "Force all tiles to be the same as the set tile size, even when tiles could be smaller. This can help prevent the model from working with irregular tile sizes."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."})),
    ]

    optional = []

    return required, optional


def USDU_guider_base_inputs():
    """
    Input definitions for guider-based USDU nodes.
    Uses GUIDER, SAMPLER, and SIGMAS instead of model/conditioning/cfg/sampler_name/scheduler.
    """
    required = [
        ("image", ("IMAGE", {"tooltip": "The image to upscale."})),
        # Guider-based sampling params
        ("guider", ("GUIDER", {"tooltip": "A guider that encapsulates the model, conditioning, and CFG. Use CFGGuider, PerpNegGuider, or other guider nodes."})),
        ("sampler", ("SAMPLER", {"tooltip": "The sampler to use. Use KSamplerSelect node to create this."})),
        ("sigmas", ("SIGMAS", {"tooltip": "The noise schedule. Use BasicScheduler or other scheduler nodes to create this. The denoise is configured in the scheduler."})),
        ("vae", ("VAE", {"tooltip": "The VAE model to use for tiles."})),
        ("upscale_by", ("FLOAT", {"default": 2, "min": 0.05, "max": 4, "step": 0.05, "tooltip": "The factor to upscale the image by."})),
        ("seed", ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The seed to use for noise generation."})),
        # Upscale Params
        ("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The upscaler model for upscaling the image."})),
        ("mode_type", (list(MODES.keys()), {"tooltip": "The tiling order to use for the redraw step."})),
        ("tile_width", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of each tile."})),
        ("tile_height", ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The height of each tile."})),
        ("mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the mask."})),
        ("tile_padding", ("INT", {"default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply between tiles."})),
        # Seam fix params
        ("seam_fix_mode", (list(SEAM_FIX_MODES.keys()), {"tooltip": "The seam fix mode to use."})),
        ("seam_fix_denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The denoising strength to use for the seam fix."})),
        ("seam_fix_width", ("INT", {"default": 64, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The width of the bands used for the Band Pass seam fix mode."})),
        ("seam_fix_mask_blur", ("INT", {"default": 8, "min": 0, "max": 64, "step": 1, "tooltip": "The blur radius for the seam fix mask."})),
        ("seam_fix_padding", ("INT", {"default": 16, "min": 0, "max": MAX_RESOLUTION, "step": 8, "tooltip": "The padding to apply for the seam fix tiles."})),
        # Misc
        ("tile_overlap_mode", (list(TILE_OVERLAP_MODES.keys()), {"default": "Reprocess Overlap", "tooltip": "How to handle tile overlap regions. 'Ignore Overlap' uses minimal tile sizes. 'Reprocess Overlap' uses uniform tiles with overlap regions potentially regenerated. 'Context Only Overlap' uses uniform tiles where overlap regions from previous tiles become read-only context."})),
        ("tiled_decode", ("BOOLEAN", {"default": False, "tooltip": "Whether to use tiled decoding when decoding tiles."})),
        ("batch_size", ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1, "tooltip": "The number of tiles to process in a batch. Higher values can reduce processing time but use more VRAM. Yields different results than individual tiles. Only affects the main redraw step, not the seam fix step."})),
    ]

    optional = [
        ("mask", ("MASK", {"tooltip": "Optional region mask. Only masked (white) areas are re-diffused; tiles that do not touch the mask are skipped entirely, which greatly speeds up small-region upscales. Sampling still sees the full tile for context, and blending uses the same mask_blur feathering as tile edges (the edit extends about mask_blur pixels past the mask). The mask may be any resolution and is resized to the upscaled canvas. Grayscale values give partial blending. With batched images, a single mask applies to every image, and a batch of masks maps one mask to each image. Not compatible with batch_size > 1 (tile batching)."})),
        ("anchor_context", ("BOOLEAN", {"default": False, "tooltip": "Hold the areas a tile will not composite back to the original image at every sampling step, so the model sees the true surroundings instead of a re-diffused version that can drift. Keeps detail consistent between sections and blends seams into the real image. Takes effect when a mask is connected or tile_overlap_mode is 'Context Only Overlap'; otherwise it has no effect."})),
    ]

    return required, optional


def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type in required:
            inputs["required"][name] = type
    if optional:
        inputs["optional"] = {}
        for name, type in optional:
            inputs["optional"][name] = type
    return inputs


def remove_input(inputs: list, input_name: str):
    for i, (n, _) in enumerate(inputs):
        if n == input_name:
            del inputs[i]
            break


def rename_input(inputs: list, old_name: str, new_name: str):
    for i, (n, t) in enumerate(inputs):
        if n == old_name:
            inputs[i] = (new_name, t)
            break


class UltimateSDUpscale:
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                custom_sampler=None, custom_sigmas=None):
        redraw_mode = MODES[mode_type]
        seam_fix_mode = SEAM_FIX_MODES[seam_fix_mode]

        #
        # Set up A1111 patches
        #

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        logger.debug("UltimateSDUpscale.upscale() using batch_size=%s", batch_size)
        if batch_size > 1 and not force_uniform_tiles:
            raise ValueError("batch_size > 1 requires force_uniform_tiles to be True; all tiles in the batch must be the same size.")

        # Convert boolean to TileOverlapMode enum
        tile_overlap_mode = TileOverlapMode.REPROCESS if force_uniform_tiles else TileOverlapMode.IGNORE

        # Processing
        sdprocessing = StableDiffusionProcessing(
            shared.batch[0], model, positive, negative, vae,
            seed, steps, cfg, sampler_name, scheduler, denoise, upscale_by,
            tile_overlap_mode, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode,
            custom_sampler, custom_sigmas, batch_size,
        )

        # Suppress logging to prevent duplicate tqdm progress bars
        with suppress_logging():
            try:
                script = usdu.Script()
                processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                                   mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                                   seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                                   upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                                   save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                                   seams_fix_type=seam_fix_mode, target_size_type=2,
                                   custom_width=None, custom_height=None, custom_scale=upscale_by)

                # Return the resulting images
                images = [pil_to_tensor(img) for img in shared.batch]
                tensor = torch.cat(images, dim=0)
                return (tensor,)
            finally:
                # Restore progress bar (belt-and-suspenders with __del__)
                if sdprocessing.progress_bar_enabled:
                    comfy_utils.PROGRESS_BAR_ENABLED = True

class UltimateSDUpscaleNoUpscale(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image."

    def upscale(self, upscaled_image, model, positive, negative, vae, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1):
        upscale_by = 1.0

        logger.debug("UltimateSDUpscaleNoUpscale.upscale() received batch_size=%s", batch_size)

        return super().upscale(upscaled_image, model, positive, negative, vae, upscale_by, seed,
                               steps, cfg, sampler_name, scheduler, denoise, None,
                               mode_type, tile_width, tile_height, mask_blur, tile_padding,
                               seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                               seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size)

class UltimateSDUpscaleCustomSample(UltimateSDUpscale):
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_base_inputs()
        remove_input(required, "upscale_model")
        optional.append(("upscale_model", ("UPSCALE_MODEL", {"tooltip": "The model to use for upscaling the image. If not provided, a simple Lanczos scaling will be used instead."})))
        optional.append(("custom_sampler", ("SAMPLER", {"tooltip": "A custom sampler to use instead of the built-in ComfyUI sampler specified by sampler_name. Only used if both custom_sampler and custom_sigmas are provided."})))
        optional.append(("custom_sigmas", ("SIGMAS", {"tooltip": "A custom noise schedule to use during sampling. Only used if both custom_sampler and custom_sigmas are provided."})))
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image."

    def upscale(self, image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size=1,
                upscale_model=None,
                custom_sampler=None, custom_sigmas=None):
        return super().upscale(image, model, positive, negative, vae, upscale_by, seed,
                steps, cfg, sampler_name, scheduler, denoise, upscale_model,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, force_uniform_tiles, tiled_decode, batch_size,
                custom_sampler, custom_sigmas)


class UltimateSDUpscaleGuider:
    """
    Ultimate SD Upscale node that uses a GUIDER input for sampling.
    This allows using custom guiders like PerpNegGuider during tiled upscaling.
    """
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_guider_base_inputs()
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final upscaled image.",)
    DESCRIPTION = "Upscales an image and runs image-to-image on tiles using a custom guider (e.g., PerpNegGuider, CFGGuider)."

    def upscale(self, image, guider, sampler, sigmas, vae, upscale_by, seed,
                upscale_model, mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tile_overlap_mode, tiled_decode, batch_size=1,
                mask=None, anchor_context=False):

        tile_overlap_mode_enum = TILE_OVERLAP_MODES[tile_overlap_mode]

        # Validate batch_size incompatibilities
        if batch_size > 1 and tile_overlap_mode_enum == TileOverlapMode.CONTEXT_ONLY:
            raise ValueError("batch_size > 1 is not compatible with Context Only Overlap mode. "
                             "Context Only Overlap requires sequential tile processing.")
        if batch_size > 1 and tile_overlap_mode_enum == TileOverlapMode.IGNORE:
            raise ValueError("batch_size > 1 requires uniform tile sizes. "
                             "Use 'Reprocess Overlap' or 'Context Only Overlap' mode.")
        if mask is not None and batch_size > 1:
            raise ValueError("mask is not compatible with batch_size > 1. "
                             "The batched tile path does not support region masks; use batch_size=1.")

        #
        # Set up A1111 patches
        #

        # Upscaler
        shared.sd_upscalers[0] = UpscalerData()
        shared.actual_upscaler = upscale_model

        # Set the batch of images
        shared.batch = [tensor_to_pil(image, i) for i in range(len(image))]
        shared.batch_as_tensor = image

        redraw_mode = MODES[mode_type]
        seam_fix_mode_enum = SEAM_FIX_MODES[seam_fix_mode]

        # Convert the optional region mask to PIL 'L' image(s). A single mask
        # is shared by every image in the batch; a batch of masks maps one
        # mask to each image (cyclically on mismatch, matching ComfyUI's
        # repeat_to_batch_size convention).
        region_mask = None
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            num_images = len(image)
            num_masks = mask.shape[0]
            if num_masks == 1:
                region_mask = mask_tensor_to_pil(mask, 0)
            else:
                if num_masks != num_images:
                    logger.warning("Mask batch size (%d) does not match image batch size (%d); "
                                   "mask[i %% num_masks] is used for image i.", num_masks, num_images)
                region_mask = [mask_tensor_to_pil(mask, i % num_masks) for i in range(num_images)]

        # Processing with guider
        sdprocessing = StableDiffusionProcessingGuider(
            shared.batch[0], guider, sampler, sigmas, vae,
            seed, upscale_by, tile_overlap_mode_enum, tiled_decode,
            tile_width, tile_height, redraw_mode, seam_fix_mode_enum,
            batch_size,
            region_mask=region_mask,
            anchor_context=anchor_context,
        )

        # Suppress logging to prevent duplicate tqdm progress bars
        with suppress_logging():
            try:
                script = usdu.Script()
                processed = script.run(p=sdprocessing, _=None, tile_width=tile_width, tile_height=tile_height,
                                   mask_blur=mask_blur, padding=tile_padding, seams_fix_width=seam_fix_width,
                                   seams_fix_denoise=seam_fix_denoise, seams_fix_padding=seam_fix_padding,
                                   upscaler_index=0, save_upscaled_image=False, redraw_mode=redraw_mode,
                                   save_seams_fix_image=False, seams_fix_mask_blur=seam_fix_mask_blur,
                                   seams_fix_type=seam_fix_mode_enum, target_size_type=2,
                                   custom_width=None, custom_height=None, custom_scale=upscale_by)

                # Return the resulting images
                images = [pil_to_tensor(img) for img in shared.batch]
                tensor = torch.cat(images, dim=0)
                return (tensor,)
            finally:
                # Restore progress bar (belt-and-suspenders with __del__)
                if sdprocessing.progress_bar_enabled:
                    comfy_utils.PROGRESS_BAR_ENABLED = True


class UltimateSDUpscaleNoUpscaleGuider(UltimateSDUpscaleGuider):
    """
    Ultimate SD Upscale node (no initial upscale) that uses a GUIDER input for sampling.
    """
    @classmethod
    def INPUT_TYPES(s):
        required, optional = USDU_guider_base_inputs()
        remove_input(required, "upscale_model")
        remove_input(required, "upscale_by")
        rename_input(required, "image", "upscaled_image")
        return prepare_inputs(required, optional)

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"
    OUTPUT_TOOLTIPS = ("The final refined image.",)
    DESCRIPTION = "Runs image-to-image on tiles from the input image using a custom guider (e.g., PerpNegGuider, CFGGuider)."

    def upscale(self, upscaled_image, guider, sampler, sigmas, vae, seed,
                mode_type, tile_width, tile_height, mask_blur, tile_padding,
                seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                seam_fix_width, seam_fix_padding, tile_overlap_mode, tiled_decode, batch_size=1,
                mask=None, anchor_context=False):
        upscale_by = 1.0
        return super().upscale(upscaled_image, guider, sampler, sigmas, vae, upscale_by, seed,
                               None, mode_type, tile_width, tile_height, mask_blur, tile_padding,
                               seam_fix_mode, seam_fix_denoise, seam_fix_mask_blur,
                               seam_fix_width, seam_fix_padding, tile_overlap_mode, tiled_decode, batch_size,
                               mask=mask, anchor_context=anchor_context)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
# This fork only exports Guider nodes - for non-guider nodes, use the original:
# https://github.com/ssitu/ComfyUI_UltimateSDUpscale
NODE_CLASS_MAPPINGS = {
    "UltimateSDUpscaleGuider": UltimateSDUpscaleGuider,
    "UltimateSDUpscaleNoUpscaleGuider": UltimateSDUpscaleNoUpscaleGuider,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "UltimateSDUpscaleGuider": "Ultimate SD Upscale (Guider)",
    "UltimateSDUpscaleNoUpscaleGuider": "Ultimate SD Upscale (No Upscale, Guider)",
}
