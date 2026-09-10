"""Execution-scoped RGB canvas with bounded tile-region disk I/O."""

import operator
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from usdu_canvas import CanvasFrameReference, clipped_region


class DiskFrameCanvas:
    """Mutable frame sequence backed by raw 8-bit RGB files.

    Returned images own their pixels and remain valid after the canvas closes.
    No tensor/mmap views or image cache survive an indexed read.
    """

    def __init__(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(prefix="usdu_canvas_", dir=directory)
        self.directory = Path(self._temporary.name)
        self.sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._temporary.cleanup()

    def __len__(self):
        return len(self.sizes)

    def _index(self, index):
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("Canvas frame index out of range")
        return index

    def _path(self, index):
        return self.directory / f"{index:08d}.rgb"

    def frame_size(self, index):
        return self.sizes[self._index(index)]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = self._index(index)
        width, height = self.sizes[index]
        return self.read_region(index, (0, 0, width, height))

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def _write(self, index, image):
        self._validate_frame(image)
        self._path(index).write_bytes(image.tobytes())

    def _validate_frame(self, image):
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("USDU's VIDEO canvas requires 8-bit RGB frames.")

    def __setitem__(self, index, image):
        index = self._index(index)
        if isinstance(image, CanvasFrameReference) and image.canvas is self and image.index == index:
            return
        self._write(index, image)
        self.sizes[index] = image.size

    def append(self, image):
        self._write(len(self), image)
        self.sizes.append(image.size)

    def read_region(self, index, region):
        """Read just the requested row segments, black-padding outside the frame."""
        index = self._index(index)
        x1, y1, x2, y2 = region
        if x2 < x1 or y2 < y1:
            raise ValueError("Invalid crop region.")
        pixels = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
        left, top, right, bottom = clipped_region(region, self.sizes[index])
        if right > left and bottom > top:
            width = self.sizes[index][0]
            with self._path(index).open("rb", buffering=0) as stream:
                for row in range(top, bottom):
                    stream.seek(((row * width) + left) * 3)
                    target = memoryview(pixels[row - y1, left - x1:right - x1]).cast("B")
                    if stream.readinto(target) != len(target):
                        raise OSError("Incomplete canvas region read.")
        return Image.fromarray(pixels)

    def write_region(self, index, region, image):
        """Overwrite just a tile rectangle; full-frame replacement stays explicit."""
        index = self._index(index)
        self._validate_frame(image)
        x1, y1, x2, y2 = region
        if (x2 <= x1 or y2 <= y1 or clipped_region(region, self.sizes[index]) != tuple(region)
                or image.size != (x2 - x1, y2 - y1)):
            raise ValueError("Canvas write region must fit the frame and match the image size.")
        pixels = np.ascontiguousarray(image, dtype=np.uint8)
        width = self.sizes[index][0]
        with self._path(index).open("r+b", buffering=0) as stream:
            for row in range(y1, y2):
                stream.seek(((row * width) + x1) * 3)
                source = memoryview(pixels[row - y1]).cast("B")
                if stream.write(source) != len(source):
                    raise OSError("Incomplete canvas region write.")
