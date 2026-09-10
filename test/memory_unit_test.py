"""CPU-only memory regressions; run directly, without the GPU pytest fixtures.

COMFYUI_ROOT=/path/to/ComfyUI python test/memory_unit_test.py
"""
import gc
import os
from pathlib import Path
import sys
import unittest
import weakref
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get("COMFYUI_ROOT", ROOT.parent.parent))
sys.path[:0] = [str(ROOT), str(COMFY)]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import comfy.utils
import numpy as np
from PIL import Image
import torch
import usdu_utils
import usdu_nodes
from modules import shared, processing


def run_node(images, **overrides):
    options = dict(upscaled_image=images, guider=None, sampler=None,
                   sigmas=torch.tensor([1., 0.]), vae=None, seed=7,
                   mode_type="None", tile_width=32, tile_height=32,
                   mask_blur=0, tile_padding=0, seam_fix_mode="None",
                   seam_fix_denoise=0.2, seam_fix_mask_blur=0,
                   seam_fix_width=0, seam_fix_padding=0,
                   tile_overlap_mode="Reprocess Overlap", tiled_decode=False,
                   batch_size=1)
    options.update(overrides)
    return usdu_nodes.UltimateSDUpscaleNoUpscaleGuider().upscale(**options)[0]


class MemoryTests(unittest.TestCase):
    def tearDown(self):
        shared.batch = []
        shared.batch_as_tensor = None
        shared.actual_upscaler = None
        shared.sd_upscalers[0] = None

    def assert_clean(self):
        self.assertEqual(shared.batch, [])
        self.assertIsNone(shared.batch_as_tensor)
        self.assertIsNone(shared.actual_upscaler)
        self.assertIsNone(shared.sd_upscalers[0])

    def test_exact_preallocated_conversion(self):
        rng = np.random.default_rng(7)
        for channels in (1, 3, 4):
            shape = (17, 13) if channels == 1 else (17, 13, channels)
            images = [Image.fromarray(rng.integers(0, 256, shape, dtype=np.uint8))
                      for _ in range(5)]
            expected = torch.cat([usdu_utils.pil_to_tensor(img) for img in images])
            original = usdu_utils.pil_to_tensor
            live = []

            def one_frame(image):
                self.assertTrue(all(ref() is None for ref in live))
                tensor = original(image)
                live.append(weakref.ref(tensor))
                return tensor

            with patch.object(usdu_utils, "pil_to_tensor", side_effect=one_frame), \
                    patch.object(torch, "cat", side_effect=AssertionError("full batch copy")):
                result = usdu_utils.pil_batch_to_tensor(images)
            self.assertTrue(torch.equal(result, expected))
            self.assertTrue(result.is_contiguous())
        with self.assertRaises(ValueError):
            usdu_utils.pil_batch_to_tensor([])
        with self.assertRaises(ValueError):
            usdu_utils.pil_batch_to_tensor([Image.new("RGB", (3, 3)), Image.new("RGB", (4, 3))])

    def test_pixel_conversion_clamps_without_changing_valid_values(self):
        values = torch.tensor([-0.1, 0., 0.5, 1., 1.1, float('nan'), float('inf'), -float('inf')])
        image = values.view(1, 1, -1, 1).expand(-1, -1, -1, 3)
        converted = np.array(usdu_utils.tensor_to_pil(image))
        self.assertEqual(converted[0, :, 0].tolist(), [0, 0, 127, 255, 255, 0, 255, 0])

    def test_lazy_crops_preserve_pixels_and_do_not_retain_tile_clip(self):
        rng = np.random.default_rng(7)
        images = [Image.fromarray(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8))
                  for _ in range(5)]
        for region, size in (((2, 3, 24, 28), (22, 25)),
                             ((-4, -3, 44, 34), (32, 32))):
            expected = usdu_utils.pil_batch_to_tensor([
                image.crop(region).resize(size, Image.Resampling.LANCZOS)
                for image in images])
            crops = usdu_utils.CroppedImages(images, region, size)
            original = usdu_utils.pil_to_tensor
            live = []

            def one_frame(image):
                self.assertTrue(all(ref() is None for ref in live))
                live.append(weakref.ref(image))
                return original(image)

            with patch.object(usdu_utils, "pil_to_tensor", new=one_frame):
                result = usdu_utils.pil_batch_to_tensor(crops)
            self.assertTrue(torch.equal(result, expected))
            self.assertTrue(all(ref() is None for ref in live))

    def test_success_releases_globals_and_preserves_input(self):
        images = torch.linspace(0, 1, 5 * 32 * 32 * 3).reshape(5, 32, 32, 3)
        snapshot = images.clone()
        expected = torch.cat([usdu_utils.pil_to_tensor(usdu_utils.tensor_to_pil(images, i))
                              for i in range(5)])
        with patch.object(usdu_nodes.usdu.Script, "run", return_value=None):
            result = run_node(images)
        self.assert_clean()
        self.assertTrue(torch.equal(images, snapshot))
        self.assertTrue(torch.equal(result, expected))
        ref = weakref.ref(images)
        del images
        gc.collect()
        self.assertIsNone(ref())

    def test_cleanup_after_setup_sampling_and_cancellation_failures(self):
        images = torch.zeros(5, 32, 32, 3)
        for target, name in ((usdu_nodes, "tensor_to_pil"),
                             (usdu_nodes, "StableDiffusionProcessingGuider"),
                             (usdu_nodes.usdu.Script, "run")):
            for exception in (RuntimeError("test"), KeyboardInterrupt()):
                enabled = comfy.utils.PROGRESS_BAR_ENABLED
                with patch.object(target, name, side_effect=exception):
                    with self.assertRaises(type(exception)):
                        run_node(images)
                self.assert_clean()
                self.assertEqual(comfy.utils.PROGRESS_BAR_ENABLED, enabled)

    def test_tile_rgb_released_before_sampling(self):
        shared.batch = [Image.new("RGB", (32, 32), (i * 30, 0, 0)) for i in range(5)]
        p = processing.StableDiffusionProcessingGuider(
            shared.batch[0], None, None, torch.tensor([1., 0.]), None,
            7, 1, processing.TileOverlapMode.REPROCESS, False, 32, 32,
            usdu_nodes.MODES["Linear"], usdu_nodes.SEAM_FIX_MODES["None"])
        p.image_mask = Image.new("L", (32, 32), 255)
        p.progress_bar_enabled = False
        encoded_inputs = []

        def encode(vae, images):
            encoded_inputs.append(weakref.ref(images))
            return ({"samples": torch.zeros(1, 4, 4, 4)},)

        def sample(*args):
            self.assertTrue(all(ref() is None for ref in encoded_inputs))
            return {"samples": torch.zeros(1, 4, 4, 4)}

        # A Mock records its call arguments and would itself retain the RGB
        # tensor; install a plain function for this lifetime assertion.
        with patch.object(p.vae_encoder, "encode", new=encode), \
                patch.object(processing, "sample_with_guider", side_effect=sample), \
                patch.object(p.vae_decoder, "decode", return_value=(torch.zeros(5, 32, 32, 3),)), \
                patch.object(processing, "_usdu_h3_startlatent_is_h3_guider", return_value=False):
            processing.process_images(p)
        self.assertEqual(len(shared.batch), 5)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
