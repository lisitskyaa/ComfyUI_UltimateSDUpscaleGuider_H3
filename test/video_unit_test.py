"""CPU tests for native VIDEO transport; no CAT pack or model weights required.

COMFYUI_ROOT=/path/to/ComfyUI python -B test/video_unit_test.py
"""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ['COMFYUI_ROOT'])
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import numpy as np
import av
import folder_paths
import nodes  # Load core before USDU isolates its A1111 imports.

spec = importlib.util.spec_from_file_location('usdu_test_pack', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
usdu = pack.usdu_nodes
video = pack.usdu_video
io = video.video_io
processing = usdu.usdu.processing


def settings(**kwargs):
    result = dict(guider=None, sampler=None, sigmas=torch.tensor([1., 0.]),
                  vae=None, seed=7, mode_type='None', tile_width=64,
                  tile_height=64, mask_blur=0, tile_padding=0,
                  seam_fix_mode='None', seam_fix_denoise=0.2,
                  seam_fix_mask_blur=0, seam_fix_width=0, seam_fix_padding=0,
                  tile_overlap_mode='Reprocess Overlap', tiled_decode=False,
                  batch_size=1)
    result.update(kwargs)
    return result


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_temp = folder_paths.get_temp_directory()
        folder_paths.set_temp_directory(str(self.root))
        self.source = self.root / 'source.mkv'
        rng = torch.Generator().manual_seed(17)
        writer = io.RGBWriter(self.source, 24)
        for i in range(5):
            writer.write(torch.rand((64, 128, 3), generator=rng), timestamp=i / 24)
        writer.close()
        self.input = io.from_path(self.source)

    def tearDown(self):
        folder_paths.set_temp_directory(self.old_temp)
        self.temp.cleanup()

    def assert_clean(self):
        self.assertEqual(usdu.shared.batch, [])
        self.assertIsNone(usdu.shared.batch_as_tensor)
        self.assertIsNone(usdu.shared.actual_upscaler)
        self.assertFalse(list(self.root.glob('usdu_canvas_*')))

    def test_registered_schema_inherits_original(self):
        cls = pack.NODE_CLASS_MAPPINGS['UltimateSDUpscaleNoUpscaleGuiderVideo']
        self.assertIs(cls, video.UltimateSDUpscaleNoUpscaleGuiderVideo)
        native = cls.INPUT_TYPES()
        original = usdu.UltimateSDUpscaleNoUpscaleGuider.INPUT_TYPES()
        self.assertEqual({k: v for k, v in native['optional'].items()
                          if k not in {'canvas_storage', 'canvas_directory'}}, original['optional'])
        self.assertEqual(native['optional']['canvas_storage'][1]['default'], 'ram')
        self.assertEqual(native['optional']['canvas_directory'][1]['default'], '')
        self.assertEqual(list(native['optional']),
                         ['mask', 'anchor_context', 'canvas_storage', 'canvas_directory'])
        for name, value in original['required'].items():
            self.assertEqual(native['required']['video' if name == 'upscaled_image' else name],
                             ('VIDEO', *value[1:]) if name == 'upscaled_image' else value)

    def test_no_refine_exact_pixels_clock_and_no_full_float_batch(self):
        inputs = torch.cat([frame for frame, _ in io.frames(self.source)])
        expected = usdu.UltimateSDUpscaleNoUpscaleGuider().upscale(inputs, **settings())[0]
        del inputs
        # A full final float clip would go through this hook in the old adapter.
        with patch.dict(usdu.UltimateSDUpscaleGuider._finish_images.__globals__,
                        {'pil_batch_to_tensor': lambda *_: self.fail('Full float output clip')}):
            output, path, count = video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(self.input, canvas_storage="disk", **settings())
        self.assertEqual(count, 5)
        actual = torch.cat([frame for frame, _ in io.frames(Path(path))])
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(list(io.timestamps(self.source)), list(io.timestamps(path)))
        with av.open(path) as c:
            self.assertEqual(c.streams.video[0].codec_context.name, 'ffv1')
            self.assertEqual(c.streams.video[0].codec_context.pix_fmt, 'gbrp16le')
        self.assertFalse(list(self.root.glob('*.f32')))
        self.assert_clean()

    def test_real_tile_loop_matches_image_path(self):
        inputs = torch.cat([frame for frame, _ in io.frames(self.source)])
        configurations = [
            {},
            {"tile_padding": 16},
            {"mode_type": "Chess", "tile_overlap_mode": "Context Only Overlap"},
            {"tile_overlap_mode": "Ignore Overlap"},
            {"batch_size": 2},
            {"batch_size": 2, "mode_type": "Chess"},
            {"mask": torch.linspace(0, 1, 128).expand(1, 64, 128)},
            {"seam_fix_mode": "Half Tile"},
            {"seam_fix_mode": "Band Pass", "seam_fix_width": 16},
            {"seam_fix_mode": "Half Tile + Intersections"},
        ]
        def encode(vae, images):
            return ({'samples': images.clone()},)
        def sample(guider, seed, sampler, sigmas, latent):
            return {'samples': 1 - latent['samples']}
        def decode(vae, samples):
            return (samples['samples'],)
        original_run = usdu.usdu.Script.run

        def region_only_run(*args, **kwargs):
            with patch.object(video.DiskFrameCanvas, '__getitem__', side_effect=AssertionError('Full canvas read in tile loop')), \
                    patch.object(video.DiskFrameCanvas, '_write', side_effect=AssertionError('Full canvas write in tile loop')):
                return original_run(*args, **kwargs)
        # Real tile order, crop, mask, resize and composition, deterministic VAE/sampler stand-ins.
        with patch.object(processing.VAEEncode, 'encode', side_effect=encode), \
             patch.object(processing, 'sample_with_guider', new=sample), \
             patch.dict(usdu.usdu.Script.run.__globals__, {'sample_with_guider': sample}), \
             patch.object(usdu.usdu.Script, 'run', new=region_only_run), \
             patch.object(processing.VAEDecode, 'decode', side_effect=decode):
            for overrides in configurations:
                options = settings(mode_type='Linear', tile_width=64, tile_height=64,
                                   tile_padding=0, mask_blur=3)
                options.update(overrides)
                expected = usdu.UltimateSDUpscaleNoUpscaleGuider().upscale(inputs, **options)[0]
                for storage in ('disk', 'ram'):
                    with self.subTest(storage=storage, settings=list(overrides)):
                        output, path, count = video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(
                            self.input, canvas_storage=storage, **options)
                        actual = torch.cat([frame for frame, _ in io.frames(path)])
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertFalse(torch.equal(actual, inputs))
                        self.assertEqual(count, 5)
                        self.assert_clean()

    def test_h3_native_temporal_latent_path(self):
        from comfy_extras.nodes_minimax_h3 import _empty_av_latent
        calls = []

        class VAE:
            def encode(self, images):
                self_shape = tuple(images.shape)
                calls.append(self_shape)
                latent, _ = _empty_av_latent(images.shape[2], images.shape[1], images.shape[0])
                return torch.full_like(latent["samples"].unbind()[0], 0.5)

        def sample(guider, seed, sampler, sigmas, latent):
            image, audio = latent["samples"].unbind()
            self.assertEqual(tuple(image.shape), (1, 24, 2, 4, 4))
            self.assertTrue(torch.equal(audio, torch.zeros_like(audio)))
            return latent

        def decode(vae, samples):
            self.assertEqual(samples["samples"].unbind()[0].shape[2], 2)
            return (torch.full((5, 64, 64, 3), 0.5),)

        with patch.object(processing, '_usdu_h3_startlatent_is_h3_guider', return_value=True), \
             patch.object(processing, 'sample_with_guider', new=sample), \
             patch.object(processing.VAEDecode, 'decode', side_effect=decode):
            _, path, count = video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(
                self.input, canvas_storage='disk', **settings(mode_type='Linear', vae=VAE()))
        self.assertEqual(calls, [(5, 64, 64, 3), (5, 64, 64, 3)])
        self.assertEqual(count, 5)
        self.assertEqual(len(list(io.timestamps(path))), 5)
        self.assert_clean()

    def test_failures_release_images_and_partial_outputs(self):
        for target, name in ((usdu.usdu.Script, 'run'), (io.RGBWriter, 'write')):
            for error in (RuntimeError('test failure'), KeyboardInterrupt()):
                before = set(self.root.iterdir())
                with patch.object(target, name, side_effect=error):
                    with self.assertRaises(type(error)):
                        video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(self.input, canvas_storage="disk", **settings())
                self.assertEqual(set(self.root.iterdir()), before)
                self.assert_clean()

    def test_scratch_directory_only_used_in_disk_mode(self):
        scratch = self.root / 'scratch' / 'canvas'
        cls = video.UltimateSDUpscaleNoUpscaleGuiderVideo
        cls().refine(self.input, canvas_directory=str(scratch), **settings())
        self.assertFalse(scratch.exists())
        original = cls._prepare_images

        def prepare(instance, source):
            self.assertEqual(source.canvas.directory.parent, scratch)
            return original(instance, source)

        with patch.object(cls, '_prepare_images', new=prepare):
            cls().refine(self.input, canvas_storage='disk', canvas_directory=str(scratch), **settings())
        self.assertEqual(list(scratch.iterdir()), [])
        self.assert_clean()

        with patch.object(video.DiskFrameCanvas, '_write', side_effect=OSError('Disk full')):
            with self.assertRaisesRegex(OSError, 'Disk full'):
                cls().refine(self.input, canvas_storage='disk', canvas_directory=str(scratch), **settings())
        self.assertEqual(list(scratch.iterdir()), [])
        self.assert_clean()

    def test_audio_copy(self):
        import subprocess
        source = self.root / 'audio.mkv'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(self.source), '-f', 'lavfi', '-i',
                        'sine=frequency=440:sample_rate=8000:duration=0.208333333',
                        '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'pcm_s16le', str(source)], check=True)
        output, path, _ = video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(io.from_path(source), canvas_storage="disk", **settings())
        def audio(path):
            with av.open(str(path)) as c:
                return np.concatenate([f.to_ndarray() for f in c.decode(audio=0)], axis=-1)
        np.testing.assert_array_equal(audio(source), audio(path))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
