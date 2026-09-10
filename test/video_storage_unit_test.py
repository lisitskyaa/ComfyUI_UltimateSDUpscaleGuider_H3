"""Raw disk canvas tests. Run directly: python -B test/video_storage_unit_test.py."""
import gc
import importlib.util
from pathlib import Path
import tempfile
import sys
import unittest
import weakref

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location('usdu_video_storage_test', ROOT / 'usdu_video_storage.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DiskFrameCanvas = module.DiskFrameCanvas


class CanvasTests(unittest.TestCase):
    def test_exact_pixels_indexing_and_replacement(self):
        rng = np.random.default_rng(17)
        originals = [Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)) for _ in range(5)]
        with tempfile.TemporaryDirectory() as root:
            with DiskFrameCanvas(root) as canvas:
                for image in originals:
                    canvas.append(image)
                self.assertEqual(len(canvas), 5)
                for index, image in enumerate(canvas):
                    self.assertEqual(image.tobytes(), originals[index].tobytes())
                self.assertEqual(canvas[-1].tobytes(), originals[-1].tobytes())
                self.assertEqual(len(canvas[1:4]), 3)
                for index in (5, -6):
                    with self.assertRaises(IndexError):
                        canvas[index]
                changed = Image.new('RGB', (19, 27), (7, 32, 255))
                canvas[2] = changed
                self.assertEqual(canvas.frame_size(2), (19, 27))
                self.assertEqual(canvas[2].tobytes(), changed.tobytes())
                retained = canvas[2]
            self.assertFalse(list(Path(root).iterdir()))
            self.assertEqual(retained.tobytes(), changed.tobytes())

    def test_canvas_does_not_retain_loaded_or_written_images(self):
        with tempfile.TemporaryDirectory() as root, DiskFrameCanvas(root) as canvas:
            image = Image.new('RGB', (32, 32))
            ref = weakref.ref(image)
            canvas.append(image)
            del image
            gc.collect()
            self.assertIsNone(ref())
            image = canvas[0]
            ref = weakref.ref(image)
            del image
            gc.collect()
            self.assertIsNone(ref())

    def test_cleanup_after_failure_or_cancellation(self):
        for error in (RuntimeError('test'), KeyboardInterrupt()):
            with tempfile.TemporaryDirectory() as root:
                with self.assertRaises(type(error)):
                    with DiskFrameCanvas(root) as canvas:
                        canvas.append(Image.new('RGB', (32, 32)))
                        raise error
                self.assertFalse(list(Path(root).iterdir()))


if __name__ == '__main__':
    unittest.main()
