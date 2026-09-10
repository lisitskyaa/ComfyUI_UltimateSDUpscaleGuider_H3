"""Region compositing and lazy references for the standard RGB canvas."""

from PIL import Image


def clipped_region(region, size):
    x1, y1, x2, y2 = region
    return max(0, x1), max(0, y1), min(size[0], x2), min(size[1], y2)


class CanvasFrameReference:
    """Planner handle: dimensions cost no disk read; crop reads only its region."""

    mode = "RGB"

    def __init__(self, canvas, index):
        self.canvas = canvas
        self.index = index

    @property
    def size(self):
        return self.canvas.frame_size(self.index)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    def crop(self, region):
        return self.canvas.read_region(self.index, region)

    def resize(self, size, resample=Image.Resampling.LANCZOS):
        if tuple(size) == self.size:
            return self
        return self.canvas[self.index].resize(size, resample=resample)


def frame_reference(frames, index):
    if hasattr(frames, "read_region"):
        return CanvasFrameReference(frames, index)
    return frames[index]


def composite_tile(frames, index, tile, region, mask):
    """Read/blend/write only the tile rectangle. Preserve legacy uint8 rounding."""
    frame = frame_reference(frames, index)
    left, top, right, bottom = clipped_region(region, frame.size)
    if right <= left or bottom <= top:
        return
    box = (left, top, right, bottom)
    alpha = mask.crop(box)
    if alpha.getbbox() is None:
        return
    tile = tile.crop((left - region[0], top - region[1],
                      right - region[0], bottom - region[1]))
    base = frame.crop(box)
    layer = tile.convert("RGBA")
    layer.putalpha(alpha)
    result = base.convert("RGBA")
    result.alpha_composite(layer)
    result = result.convert("RGB")
    if hasattr(frames, "write_region"):
        frames.write_region(index, box, result)
    else:
        frame.paste(result, (left, top))
