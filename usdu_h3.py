"""H3 geometry and run-local compatibility for masked tile sampling."""

from copy import copy
from functools import lru_cache
import logging
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def is_h3_guider(guider):
    model = getattr(getattr(guider, "model_patcher", None), "model", None)
    diffusion = getattr(model, "diffusion_model", None)
    return any(base.__name__ == "MiniMaxH3" for base in type(model).__mro__) or any(
        base.__name__ == "MiniMaxH3Model" for base in type(diffusion).__mro__)


def aligned_size(size):
    return tuple(((int(edge) + 31) // 32) * 32 for edge in size)


def build_noise_mask(samples, *, masks=None,
                     source_frames=None, crop_region=None, tile_size=None):
    """Map pixel edit masks to H3's temporal tokens and 32px spatial patches.

    Any editable pixel makes its whole model patch editable. Pixel compositing
    remains responsible for the exact, soft boundary. Padded time repeats the
    last frame's mask; padded spatial context is held when anchoring is on.
    """
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN
    from comfy.nested_tensor import NestedTensor

    video, audio = samples.unbind()
    video_mask = torch.ones_like(video)
    if masks is not None:
        token_rows = []
        frame_start = 0
        pixel_height, pixel_width = video.shape[-2] * 16, video.shape[-1] * 16
        for token in range(1 if len(masks) == 1 else video.shape[2]):
            count = FRAME_PER_TOKEN[token % len(FRAME_PER_TOKEN)]
            union = None
            # Pool every source frame represented by this token, not a generic
            # linear resize in time (H3's first token in each block is shorter).
            indices = (0,) if len(masks) == 1 else range(frame_start, frame_start + count)
            for index in indices:
                mask = masks[min(index, source_frames - 1)].crop(crop_region)
                if mask.size != tile_size:
                    mask = mask.resize(tile_size, Image.Resampling.BILINEAR)
                pixels = np.asarray(mask) > 0
                union = pixels.copy() if union is None else np.logical_or(union, pixels)
            row = torch.from_numpy(union.astype(np.float32))[None, None]
            row = F.pad(row, (0, pixel_width - tile_size[0], 0, pixel_height - tile_size[1]))
            row = F.max_pool2d(row, 32, 32)
            row = row.repeat_interleave(2, -2).repeat_interleave(2, -1)
            token_rows.append(row[0, 0])
            frame_start += count
        video_mask.copy_(torch.stack(token_rows, dim=0).to(video_mask)[None, None])
    audio_mask = torch.ones_like(audio)
    return NestedTensor((video_mask, audio_mask))


def mask_velocity_wrapper(executor, *args, **kwargs):
    """Apply core PR #15988's scaling before H3's audio carry conversion."""
    output = list(executor(*args, **kwargs))
    for index, name in enumerate(("denoise_mask", "audio_denoise_mask")):
        mask = kwargs.get(name)
        if mask is not None:
            output[index] = output[index] * mask
    return output


@lru_cache(maxsize=8)
def needs_mask_velocity_fix(forward):
    """Check the actual forward contract on CPU without constructing weights.

    A behavior check also accepts backported core fixes without depending on a
    version string. Unknown implementations fail before source video decoding.
    """
    video = torch.ones(1, 1, 1, 2, 2)
    audio = torch.ones(1, 1, 2, 2)
    masks = (torch.full_like(video, 0.5), torch.full_like(audio, 0.25))
    probe = SimpleNamespace(
        _forward=lambda *args, **kwargs: [video.clone(), audio.clone()],
        sigma_shift_video=12.0, sigma_shift_audio=3.0)
    try:
        with torch.inference_mode():
            output = forward(
                probe, [torch.zeros_like(video), torch.zeros_like(audio)],
                torch.tensor([500.0]), torch.empty(1, 1, 1),
                transformer_options={}, minimax_payload={"audio_scale": 1.0},
                denoise_mask=masks[0], audio_denoise_mask=masks[1])
        if len(output) == 2 and all(torch.equal(value, mask) for value, mask in zip(output, masks)):
            return False
        if len(output) == 2 and torch.equal(output[0], video) and torch.equal(output[1], audio):
            return True
    except Exception as exc:
        raise RuntimeError("Cannot verify this H3 model's mask support. Use native ComfyUI H3 "
                           "with the mask correction from PR #15988.") from exc
    raise RuntimeError("Unrecognized H3 mask conversion. Use native ComfyUI H3 "
                       "with the mask correction from PR #15988.")


def prepare_masked_guider(guider):
    """Copy sampling state and apply the correction only to this node's guider."""
    import comfy.model_patcher
    import comfy.patcher_extension as extension

    forward = guider.model_patcher.get_model_object("diffusion_model.forward")
    forward = getattr(forward, "__func__", forward)
    needs_fix = needs_mask_velocity_fix(forward)
    prepared = copy(guider)
    prepared.model_patcher = guider.model_patcher.clone()
    prepared.model_options = comfy.model_patcher.create_model_options_clone(guider.model_options)
    wrappers = prepared.model_options.get("transformer_options", {}).get("wrappers", {})
    wrappers.get(extension.WrappersMP.DIFFUSION_MODEL, {}).pop("usdu_h3_mask_velocity", None)
    if needs_fix:
        extension.add_wrapper_with_key(
            extension.WrappersMP.DIFFUSION_MODEL, "usdu_h3_mask_velocity",
            mask_velocity_wrapper, prepared.model_options, is_model_options=True)
        logger.info("USDU H3: applying run-local mask velocity correction (ComfyUI PR #15988).")
    return prepared
