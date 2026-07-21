"""
Tests for the optional region `mask` input on the Guider nodes.

All equality assertions are on in-memory tensors (test image I/O is JPEG).
Input sizes are multiples of 64 (512/1024) so the NoUpscale canvas resize is
an identity copy, which the bit-identity assertions depend on.
"""

import pathlib
import pytest
import torch
from PIL import Image

from setup_utils import execute
from io_utils import save_image
from configs import DirectoryConfig
from fixtures_images import EXT
from usdu_utils import tensor_to_pil, pil_to_tensor

# Image file names
CATEGORY = pathlib.Path("masked_upscale")

# Hard box used by most masked tests, in 1024-canvas coordinates (x1, y1, x2, y2)
BOX_1024 = (300, 300, 800, 800)


#
# # Helpers
#
def make_mask(h, w, box=None):
    """Build a [1, H, W] mask tensor: zeros, with the (x1, y1, x2, y2) box set to 1.0."""
    mask = torch.zeros(1, h, w)
    if box is not None:
        x1, y1, x2, y2 = box
        mask[:, y1:y2, x1:x2] = 1.0
    return mask


class CountingGuider:
    """Wraps a guider and counts .sample() calls; delegates everything else."""

    def __init__(self, guider):
        self._guider = guider
        self.call_count = 0
        self.denoise_mask_flags = []

    def sample(self, *args, **kwargs):
        self.call_count += 1
        self.denoise_mask_flags.append(kwargs.get("denoise_mask") is not None)
        return self._guider.sample(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._guider, name)


def assert_outside_unchanged(out, inp, box, margin=0):
    """Zero the margin-expanded box in both tensors and require exact equality."""
    x1, y1, x2, y2 = box
    h, w = inp.shape[1], inp.shape[2]
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, w)
    y2 = min(y2 + margin, h)
    out_masked = out.clone()
    inp_masked = inp.clone()
    out_masked[:, y1:y2, x1:x2, :] = 0
    inp_masked[:, y1:y2, x1:x2, :] = 0
    assert torch.equal(out_masked.cpu(), inp_masked.cpu()), \
        "Pixels outside the mask box changed"


def region_mae(a, b, box, shrink=0):
    """Mean absolute error inside the (optionally shrunk) box."""
    x1, y1, x2, y2 = box
    a_in = a[:, y1 + shrink:y2 - shrink, x1 + shrink:x2 - shrink, :]
    b_in = b[:, y1 + shrink:y2 - shrink, x1 + shrink:x2 - shrink, :]
    return (a_in.cpu() - b_in.cpu()).abs().mean().item()


def run_full(node_classes, image, guider, sampler, sigmas, vae, upscale_model, seed,
             mask=None, upscale_by=2.0, mode_type="Linear",
             tile_overlap_mode="Reprocess Overlap", mask_blur=0, tile_padding=32,
             seam_fix_mode="None", seam_fix_denoise=1.0, seam_fix_width=64,
             seam_fix_mask_blur=0, seam_fix_padding=16, batch_size=1,
             anchor_context=False):
    """Run UltimateSDUpscaleGuider with masked-test defaults and return the output tensor."""
    usdu = node_classes["UltimateSDUpscaleGuider"]
    with torch.inference_mode():
        (out,) = usdu().upscale(
            image=image,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            vae=vae,
            upscale_by=upscale_by,
            seed=seed,
            upscale_model=upscale_model,
            mode_type=mode_type,
            tile_width=512,
            tile_height=512,
            mask_blur=mask_blur,
            tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=seam_fix_denoise,
            seam_fix_width=seam_fix_width,
            seam_fix_mask_blur=seam_fix_mask_blur,
            seam_fix_padding=seam_fix_padding,
            tile_overlap_mode=tile_overlap_mode,
            tiled_decode=False,
            batch_size=batch_size,
            mask=mask,
            anchor_context=anchor_context,
        )
    # Clone to move the result out of inference mode for in-place test helpers
    return out.clone()


def run_noupscale(node_classes, image, guider, sampler, sigmas, vae, seed,
                  mask=None, mode_type="Linear", tile_overlap_mode="Reprocess Overlap",
                  mask_blur=0, tile_padding=32, seam_fix_mode="None",
                  seam_fix_denoise=1.0, seam_fix_width=64, seam_fix_mask_blur=0,
                  seam_fix_padding=16, batch_size=1, anchor_context=False):
    """Run UltimateSDUpscaleNoUpscaleGuider with masked-test defaults and return the output tensor."""
    usdu = node_classes["UltimateSDUpscaleNoUpscaleGuider"]
    with torch.inference_mode():
        (out,) = usdu().upscale(
            upscaled_image=image,
            guider=guider,
            sampler=sampler,
            sigmas=sigmas,
            vae=vae,
            seed=seed,
            mode_type=mode_type,
            tile_width=512,
            tile_height=512,
            mask_blur=mask_blur,
            tile_padding=tile_padding,
            seam_fix_mode=seam_fix_mode,
            seam_fix_denoise=seam_fix_denoise,
            seam_fix_width=seam_fix_width,
            seam_fix_mask_blur=seam_fix_mask_blur,
            seam_fix_padding=seam_fix_padding,
            tile_overlap_mode=tile_overlap_mode,
            tiled_decode=False,
            batch_size=batch_size,
            mask=mask,
            anchor_context=anchor_context,
        )
    # Clone to move the result out of inference mode for in-place test helpers
    return out.clone()


#
# # Fixtures
#
@pytest.fixture(scope="module")
def guider_setup(base_image, loaded_checkpoint, node_classes):
    """Build (guider, sampler, sigmas) mirroring the main workflow test, denoise 0.4."""
    image, positive, negative = base_image
    model, clip, vae = loaded_checkpoint

    with torch.inference_mode():
        custom_scheduler = node_classes["KarrasScheduler"]
        (sigmas,) = execute(custom_scheduler, 20, 14.614642, 0.0291675, 7.0)
        (_, sigmas) = execute(node_classes["SplitSigmasDenoise"], sigmas, 0.4)

        custom_sampler = node_classes["KSamplerSelect"]
        (sampler,) = execute(custom_sampler, "dpmpp_2m")

        CFGGuider = node_classes["CFGGuider"]
        (guider,) = execute(
            CFGGuider,
            model=model,
            positive=positive,
            negative=negative,
            cfg=8.0,
        )

    return guider, sampler, sigmas


@pytest.fixture(scope="module")
def input_512(base_image):
    """First base image as a [1, 512, 512, 3] tensor (JPEG-derived, quantization-safe)."""
    image, _, _ = base_image
    return image[0:1]


@pytest.fixture(scope="module")
def input_1024(input_512):
    """1024x1024 multiple-of-64 input built from the base image (quantization-safe)."""
    pil = tensor_to_pil(input_512)
    pil = pil.resize((1024, 1024), Image.Resampling.LANCZOS)
    return pil_to_tensor(pil)


#
# # Tests
#
class TestMaskedUpscale:
    """Integration tests for the optional region mask on the Guider nodes."""

    def test_all_white_mask_equals_no_mask(
        self, input_512, guider_setup, loaded_checkpoint, upscale_model, node_classes, seed
    ):
        """An all-white mask must match a no-mask run with the same seed.

        The mask is low-res (512) against a 1024 canvas, so this also
        exercises the resize path.
        """
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        out_no_mask = run_full(node_classes, input_512, guider, sampler, sigmas,
                               vae, upscale_model, seed)
        out_white = run_full(node_classes, input_512, guider, sampler, sigmas,
                             vae, upscale_model, seed, mask=torch.ones(1, 512, 512))

        mae = (out_no_mask - out_white).abs().mean().item()
        assert mae < 1e-4, f"All-white mask output differs from no-mask output (MAE={mae})"

    def test_all_black_mask_skips_everything(
        self, input_512, guider_setup, loaded_checkpoint, node_classes, seed
    ):
        """An all-black mask must skip all sampling and return the input bit-identical."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        counting = CountingGuider(guider)
        out = run_noupscale(node_classes, input_512, counting, sampler, sigmas,
                            vae, seed, mask=make_mask(512, 512))

        assert counting.call_count == 0, "All-black mask must not sample any tile"
        assert torch.equal(out.cpu(), input_512.cpu()), \
            "All-black mask output must be bit-identical to the input"

    def test_outside_mask_unchanged_inside_changed(
        self, input_1024, guider_setup, loaded_checkpoint, node_classes, seed,
        test_dirs: DirectoryConfig,
    ):
        """With mask_blur=0 and a hard box mask, only pixels inside the box change."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        mask = make_mask(1024, 1024, box=BOX_1024)
        out = run_noupscale(node_classes, input_1024, guider, sampler, sigmas,
                            vae, seed, mask=mask)
        save_image(out, test_dirs.sample_images / CATEGORY / ("masked_box" + EXT))

        assert_outside_unchanged(out, input_1024, BOX_1024, margin=0)
        inside_mae = region_mae(out, input_1024, BOX_1024)
        assert inside_mae > 0.005, \
            f"Pixels inside the mask box did not change (MAE={inside_mae})"

    def test_tile_skip_count_2x2(
        self, input_512, guider_setup, loaded_checkpoint, upscale_model, node_classes, seed
    ):
        """A mask confined to one tile of a 2x2 grid must sample exactly one tile."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        counting = CountingGuider(guider)
        mask = make_mask(1024, 1024, box=(100, 100, 400, 400))
        run_full(node_classes, input_512, counting, sampler, sigmas,
                 vae, upscale_model, seed, mask=mask)

        assert counting.call_count == 1, \
            f"Expected exactly 1 sampled tile, got {counting.call_count}"

    def test_low_res_mask_resizes(
        self, input_1024, guider_setup, loaded_checkpoint, node_classes, seed
    ):
        """A half-resolution mask must behave like the equivalent full-resolution mask."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        # 512 mask box (100,100,250,250) -> effective (200,200,500,500) at 1024
        mask = make_mask(512, 512, box=(100, 100, 250, 250))
        effective_box = (200, 200, 500, 500)
        out = run_noupscale(node_classes, input_1024, guider, sampler, sigmas,
                            vae, seed, mask=mask)

        # Small margin absorbs bilinear resize softness at the box edge
        assert_outside_unchanged(out, input_1024, effective_box, margin=4)
        inside_mae = region_mae(out, input_1024, effective_box, shrink=4)
        assert inside_mae > 0.005, \
            f"Pixels inside the resized mask box did not change (MAE={inside_mae})"

    def test_context_only_with_mask(
        self, input_1024, guider_setup, loaded_checkpoint, node_classes, seed,
        test_dirs: DirectoryConfig,
    ):
        """Context Only Overlap + mask completes cleanly with the same composite clipping."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        mask = make_mask(1024, 1024, box=BOX_1024)
        out = run_noupscale(node_classes, input_1024, guider, sampler, sigmas,
                            vae, seed, mask=mask,
                            tile_overlap_mode="Context Only Overlap")
        save_image(out, test_dirs.sample_images / CATEGORY / ("masked_context_only" + EXT))

        assert_outside_unchanged(out, input_1024, BOX_1024, margin=0)
        inside_mae = region_mae(out, input_1024, BOX_1024)
        assert inside_mae > 0.005, \
            f"Pixels inside the mask box did not change (MAE={inside_mae})"

    def test_mask_with_batch_size_raises(
        self, input_512, guider_setup, loaded_checkpoint, upscale_model, node_classes, seed
    ):
        """mask + batch_size > 1 must raise a clear ValueError."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        with pytest.raises(ValueError, match="mask is not compatible"):
            run_full(node_classes, input_512, guider, sampler, sigmas,
                     vae, upscale_model, seed,
                     mask=torch.ones(1, 512, 512), batch_size=2)

    @pytest.mark.parametrize(
        "seam_fix_mode", ["Band Pass", "Half Tile", "Half Tile + Intersections"]
    )
    def test_seam_fix_modes_with_mask(
        self, input_1024, guider_setup, loaded_checkpoint, node_classes, seed, seam_fix_mode
    ):
        """Each seam fix mode + mask runs and leaves outside-mask pixels unchanged."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        mask = make_mask(1024, 1024, box=BOX_1024)
        # mode_type "None" isolates the seam fix pass
        out = run_noupscale(node_classes, input_1024, guider, sampler, sigmas,
                            vae, seed, mask=mask, mode_type="None",
                            seam_fix_mode=seam_fix_mode, seam_fix_denoise=0.6,
                            seam_fix_mask_blur=0)

        assert_outside_unchanged(out, input_1024, BOX_1024, margin=0)


class TestAnchorContext:
    """Integration tests for the anchor_context toggle on the Guider nodes."""

    def test_anchor_off_mask_no_denoise_mask(
        self, input_512, guider_setup, loaded_checkpoint, node_classes, seed
    ):
        """With anchor off (default), a masked run must not pass a denoise mask."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        counting = CountingGuider(guider)
        mask = make_mask(512, 512, box=(100, 100, 400, 400))
        run_noupscale(node_classes, input_512, counting, sampler, sigmas,
                      vae, seed, mask=mask)

        assert counting.call_count == 1, \
            f"Expected exactly 1 sampled tile, got {counting.call_count}"
        assert counting.denoise_mask_flags == [False], \
            "Anchor off must keep denoise_mask=None (current behavior)"

    def test_anchor_on_mask_denoise_mask_passed(
        self, input_1024, guider_setup, loaded_checkpoint, node_classes, seed,
        test_dirs: DirectoryConfig,
    ):
        """Anchor on + hard box mask: every tile gets a denoise mask; composite semantics unchanged."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        counting = CountingGuider(guider)
        mask = make_mask(1024, 1024, box=BOX_1024)
        out = run_noupscale(node_classes, input_1024, counting, sampler, sigmas,
                            vae, seed, mask=mask, anchor_context=True)
        save_image(out, test_dirs.sample_images / CATEGORY / ("anchored_box" + EXT))

        assert counting.call_count == 4, \
            f"Expected 4 sampled tiles, got {counting.call_count}"
        assert all(counting.denoise_mask_flags), \
            "Every sampled tile must receive a denoise mask with anchor on"
        assert_outside_unchanged(out, input_1024, BOX_1024, margin=0)
        inside_mae = region_mae(out, input_1024, BOX_1024)
        assert inside_mae > 0.005, \
            f"Pixels inside the mask box did not change (MAE={inside_mae})"

    def test_anchor_on_no_mask_reprocess_is_noop(
        self, input_512, guider_setup, loaded_checkpoint, node_classes, seed
    ):
        """Anchor on without a mask in Reprocess Overlap is a documented silent no-op."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        counting = CountingGuider(guider)
        run_noupscale(node_classes, input_512, counting, sampler, sigmas,
                      vae, seed, anchor_context=True)

        assert counting.call_count >= 1, "Expected at least one sampled tile"
        assert counting.denoise_mask_flags == [False] * counting.call_count, \
            "Anchor without mask in Reprocess Overlap must not pass a denoise mask"

    def test_anchor_on_context_only_no_mask(
        self, input_1024, guider_setup, loaded_checkpoint, node_classes, seed
    ):
        """Anchor on + Context Only Overlap (no mask): every tile gets a denoise mask."""
        guider, sampler, sigmas = guider_setup
        _, _, vae = loaded_checkpoint

        counting = CountingGuider(guider)
        out = run_noupscale(node_classes, input_1024, counting, sampler, sigmas,
                            vae, seed, tile_overlap_mode="Context Only Overlap",
                            anchor_context=True)

        assert counting.call_count == 4, \
            f"Expected 4 sampled tiles, got {counting.call_count}"
        assert all(counting.denoise_mask_flags), \
            "Every sampled tile must receive a denoise mask with anchor on"
        assert out.shape == input_1024.shape, \
            f"Output shape {tuple(out.shape)} != input shape {tuple(input_1024.shape)}"


class TestMaskedSkipUnit:
    """No-GPU unit test for the skip path in process_images."""

    def test_skip_returns_before_vae_access(self):
        """A tile whose post-intersection mask is empty must skip before VAE access."""
        from types import SimpleNamespace
        import modules.processing as processing

        class ExplodingVAEEncoder:
            def encode(self, *args, **kwargs):
                raise AssertionError("VAE encode must not be reached for a skipped tile")

        class FakePbar:
            def __init__(self):
                self.count = 0

            def update(self, n):
                self.count += n

        canvas = (128, 128)
        # White tile rect on the left half
        tile_mask = Image.new('L', canvas, 0)
        tile_mask.paste(255, (0, 0, 64, 128))
        # Region only on the right half -> empty intersection
        region = Image.new('L', canvas, 0)
        region.paste(255, (96, 0, 128, 128))

        pbar = FakePbar()
        p = SimpleNamespace(
            image_mask=tile_mask,
            init_images=[Image.new('RGB', canvas)],
            inpaint_full_res_padding=0,
            region_mask=region,
            _region_mask_cache={},
            tile_overlap_mode=processing.TileOverlapMode.REPROCESS,
            progress_bar_enabled=True,
            pbar=pbar,
            seed=0,
            vae=None,
            vae_encoder=ExplodingVAEEncoder(),
        )

        result = processing.process_images(p)

        assert result.images == [], "Skipped tile must return an empty Processed"
        assert pbar.count == 1, "Skipped tile must still advance the progress bar"


# Allow running directly for debugging
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
