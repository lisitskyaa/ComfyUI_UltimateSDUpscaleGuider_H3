"""CPU regressions for H3 contracts; no model weights or GPU required.

COMFYUI_ROOT=/path/to/ComfyUI python -B test/h3_unit_test.py
"""
import contextlib
import io as stdio
import unittest
from unittest.mock import patch

import video_unit_test as env
from PIL import Image
import numpy as np
import torch
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, MiniMaxH3Model, time_shift_sigma
from comfy.model_patcher import ModelPatcher
from comfy.samplers import CFGGuider
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

h3 = env.processing.usdu_h3


class MiniMaxH3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = MiniMaxH3Model.__new__(MiniMaxH3Model)
        torch.nn.Module.__init__(self.diffusion_model)
        self.diffusion_model.sigma_shift_video = 12.0
        self.diffusion_model.sigma_shift_audio = 3.0


def make_guider():
    return CFGGuider(ModelPatcher(MiniMaxH3(), torch.device('cpu'), torch.device('cpu')))


class VAE:
    def __init__(self):
        self.encoded = []

    def encode(self, pixels):
        self.encoded.append(tuple(pixels.shape))
        latent, _ = _empty_av_latent(pixels.shape[2], pixels.shape[1], pixels.shape[0])
        return torch.full_like(latent['samples'].unbind()[0], 0.5)

    def decode(self, latent):
        # Color-code the padded edge so accidental resizing instead of cropping
        # is observable in the output. Only used by the geometry regression.
        height, width = latent.shape[-2] * 16, latent.shape[-1] * 16
        frames = sum(FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(latent.shape[2]))
        pixels = torch.full((frames, height, width, 3), 0.5)
        if width == 96:
            pixels[:, :, 80:] = 1.0
        return pixels


class H3Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = env.VideoTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.guider = make_guider()
        self.vae = VAE()

    def run_node(self, *, sample_override=None, **overrides):
        calls = []

        def sample(guider, seed, sampler, sigmas, latent):
            calls.append((guider, sigmas.clone(), latent.get('noise_mask')))
            if sample_override is not None:
                return sample_override(guider, seed, sampler, sigmas, latent)
            return latent

        options = env.settings(mode_type='Linear', guider=self.guider, vae=self.vae)
        options.update(overrides)
        with contextlib.redirect_stdout(stdio.StringIO()), contextlib.redirect_stderr(stdio.StringIO()), \
                patch.object(env.processing, 'sample_with_guider', new=sample):
            output = env.video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(
                self.fixture.input, **options)
        self.fixture.assert_clean()
        return calls, output

    def test_rejects_h3_tile_batching_before_source_decode(self):
        # Test both native and subclass detection using the actual ModelPatcher.
        class Derived(MiniMaxH3):
            pass
        for model in (MiniMaxH3(), Derived()):
            guider = CFGGuider(ModelPatcher(model, torch.device('cpu'), torch.device('cpu')))
            with patch.object(env.video.video_io, 'frames', side_effect=AssertionError('decoded source')):
                with self.assertRaisesRegex(ValueError, 'batch_size=1'):
                    self.run_node(guider=guider, batch_size=2)
            self.fixture.assert_clean()

    def test_legacy_defaults_leave_sampling_unmasked(self):
        calls, _ = self.run_node()
        self.assertTrue(all(mask is None for _, _, mask in calls))
        self.assertEqual(self.guider.model_options, {'transformer_options': {}})
        self.assertEqual(self.guider.model_patcher.wrappers, {})

    def test_anchor_context_and_exact_unedited_pixels(self):
        region = torch.zeros(5, 64, 128)
        region[:, 16:48, 32:64] = 1
        calls, output = self.run_node(mask=region, anchor_context=True)
        self.assertTrue(calls)
        video_masks = [mask.unbind()[0] for _, _, mask in calls]
        self.assertTrue(any(torch.any(mask == 0) and torch.any(mask == 1) for mask in video_masks))
        self.assertTrue(all(set(torch.unique(mask).tolist()) <= {0., 1.} for mask in video_masks))
        self.assertTrue(all(torch.all(mask.unbind()[1] == 1) for _, _, mask in calls))
        original = torch.cat([frame for frame, _ in env.io.frames(self.fixture.source)])
        expected = env.usdu.UltimateSDUpscaleGuider._prepare_images(None, original)[0]
        result = torch.cat([frame for frame, _ in env.io.frames(output[1])])
        # Compare in uint8, the canvas's explicit precision, outside the edit.
        for index in range(5):
            original_pixels = torch.from_numpy(np.array(expected[index])).float() / 255
            self.assertTrue(torch.allclose(result[index][region[index] == 0],
                                           original_pixels[region[index] == 0], atol=1/65535, rtol=0))

    def test_context_overlap_also_passes_h3_anchor_mask(self):
        calls, _ = self.run_node(anchor_context=True, tile_padding=32,
                                 tile_overlap_mode='Context Only Overlap')
        self.assertTrue(all(mask is not None for _, _, mask in calls))
        self.assertTrue(any(torch.any(mask.unbind()[0] == 0) for _, _, mask in calls))

    def test_storage_preserves_h3_shapes_and_masked_source(self):
        original = torch.cat([frame for frame, _ in env.io.frames(self.fixture.source)])
        region = torch.zeros(5, 64, 128)
        region[:, 16:48, 32:64] = 1
        expected = (original * 255).to(torch.uint8).float() / 255
        shapes, outputs = [], []
        for storage in ('ram', 'disk'):
            self.vae.encoded.clear()
            calls, output = self.run_node(mask=region, anchor_context=True, tile_padding=16,
                                          canvas_storage=storage)
            result = torch.cat([frame for frame, _ in env.io.frames(output[1])])
            self.assertTrue(torch.equal(result[region == 0], expected[region == 0]))
            outputs.append(result)
            shapes.append((list(self.vae.encoded), [tuple(mask.unbind()[0].shape) for _, _, mask in calls]))
        self.assertTrue(torch.equal(*outputs))
        self.assertEqual(*shapes)

    def test_padding_and_all_seam_modes_preserve_output_geometry(self):
        for overlap in ('Ignore Overlap', 'Reprocess Overlap', 'Context Only Overlap'):
            for seams in env.usdu.SEAM_FIX_MODES:
                with self.subTest(overlap=overlap, seams=seams):
                    calls, output = self.run_node(tile_padding=16, seam_fix_padding=16,
                                                  seam_fix_width=16, seam_fix_mode=seams,
                                                  tile_overlap_mode=overlap)
                    self.assertTrue(calls)
                    self.assertEqual(output[2], 5)
                    frames = list(env.io.frames(output[1]))
                    self.assertEqual(tuple(frames[0][0].shape), (1, 64, 128, 3))
                    self.assertTrue(all(shape[1] % 32 == 0 and shape[2] % 32 == 0
                                        for shape in self.vae.encoded))
                    if overlap == 'Reprocess Overlap' and seams == 'None':
                        self.assertTrue(torch.allclose(frames[0][0], torch.full_like(frames[0][0], 127/255), atol=1/65535, rtol=0))

    def test_odd_canvas_and_frame_grid_keep_pixels_shape_and_clock(self):
        for count in (5, 21, 22, 23, 39, 175):
            with self.subTest(frames=count):
                source = self.fixture.root / f'odd_{count}.mkv'
                writer = env.io.RGBWriter(source, 24)
                for index in range(count):
                    writer.write(torch.full((65, 71, 3), .5))
                writer.close()
                self.fixture.input = env.io.from_path(source)
                _, output = self.run_node(tile_padding=16)
                self.assertEqual(output[2], count)
                decoded = list(env.io.frames(output[1]))
                self.assertEqual(len(decoded), count)
                self.assertEqual(tuple(decoded[0][0].shape), (1, 65, 71, 3))
                self.assertEqual(list(env.io.timestamps(source)), list(env.io.timestamps(output[1])))

    def test_masked_sampling_failures_do_not_change_input_guider(self):
        for exception in (RuntimeError('sampling failed'), KeyboardInterrupt()):
            def fail(*args):
                self.assertIsNotNone(args[-1].get('noise_mask'))
                raise exception
            with self.assertRaises(type(exception)):
                self.run_node(anchor_context=True, tile_overlap_mode='Context Only Overlap',
                              sample_override=fail)
            self.fixture.assert_clean()
            self.assertEqual(self.guider.model_options, {'transformer_options': {}})
            self.assertEqual(self.guider.model_patcher.wrappers, {})

    def test_h3_temporal_mask_mapping_and_padding(self):
        latent, _ = _empty_av_latent(96, 64, 22)
        frames = [Image.new('L', (80, 64)) for _ in range(21)]
        # Token 1 covers frames 1..4; token 5 covers frame 17 alone.
        frames[4].putpixel((0, 0), 255)
        frames[17].putpixel((32, 0), 255)
        frames[20].putpixel((64, 0), 255)
        video, audio = h3.build_noise_mask(latent['samples'],
            masks=frames, source_frames=21, crop_region=(0, 0, 80, 64), tile_size=(80, 64)).unbind()
        self.assertEqual(tuple(video.shape), (1, 24, 7, 4, 6))
        self.assertEqual(torch.nonzero(video[0, 0].flatten(1).any(1)).flatten().tolist(), [1, 5, 6])
        self.assertTrue(torch.all(video[0, :, 1, :2, :2] == 1))
        self.assertTrue(torch.all(video[0, :, 5, :2, 2:4] == 1))
        self.assertTrue(torch.all(video[0, :, 6, :2, 4:6] == 1))
        self.assertTrue(torch.all(audio == 1))

    def test_core_mask_correction_and_audio_carry(self):
        prepared = h3.prepare_masked_guider(self.guider)
        model = prepared.model_patcher.model.diffusion_model
        video = torch.ones(1, 1, 1, 2, 2) * 2
        audio = torch.ones(1, 1, 2, 2) * 3
        model._forward = lambda *args, **kwargs: [video.clone(), audio.clone()]
        video_mask = torch.tensor([[[[[0., .25], [.5, 1.]]]]])
        audio_mask = torch.full_like(audio, 0.25)
        sigma = torch.tensor(0.5)
        for scale in (1.0, 2.0):
            output = model.forward([torch.zeros_like(video), torch.ones_like(audio)], sigma[None] * 1000,
                torch.empty(1, 1, 1), transformer_options=prepared.model_options['transformer_options'],
                minimax_payload={'audio_scale': scale}, denoise_mask=video_mask, audio_denoise_mask=audio_mask)
            torch.testing.assert_close(output[0], video * video_mask)
            if scale == 1:
                expected_audio = audio * audio_mask
            else:
                audio_sigma = time_shift_sigma(sigma, 12.0, 3.0)
                expected_audio = (1-scale) * (audio_sigma/sigma) + (1+(scale-1)*audio_sigma) * audio * audio_mask
            torch.testing.assert_close(output[1], expected_audio)
        self.assertEqual(self.guider.model_options, {'transformer_options': {}})

    def test_probe_handles_fixed_legacy_and_unknown_contracts(self):
        def legacy(model, *args, **kwargs):
            return model._forward(*args, **kwargs)
        def corrected(model, *args, **kwargs):
            return h3.mask_velocity_wrapper(model._forward, *args, **kwargs)
        def unknown(model, *args, **kwargs):
            return [torch.zeros(1), torch.zeros(1)]
        self.assertTrue(h3.needs_mask_velocity_fix(legacy))
        self.assertFalse(h3.needs_mask_velocity_fix(corrected))
        with self.assertRaisesRegex(RuntimeError, 'Unrecognized'):
            h3.needs_mask_velocity_fix(unknown)

    def test_compatibility_wrapper_is_never_stacked(self):
        import comfy.patcher_extension as extension
        for needs_fix in (True, False):
            with patch.object(h3, 'needs_mask_velocity_fix', return_value=needs_fix):
                prepared = h3.prepare_masked_guider(h3.prepare_masked_guider(self.guider))
            wrappers = extension.get_wrappers_with_key(extension.WrappersMP.DIFFUSION_MODEL,
                'usdu_h3_mask_velocity', prepared.model_options, is_model_options=True)
            self.assertEqual(len(wrappers), int(needs_fix))


if __name__ == '__main__':
    unittest.main(argv=[__file__])
