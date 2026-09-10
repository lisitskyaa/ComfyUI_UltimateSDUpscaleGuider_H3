"""Lossless VIDEO transport, moved from ComfyUI-ContextAnchoredTile-videopath.

Frames cross the codec boundary one at a time; no CAT installation is required.
"""

import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path


def check_interrupt():
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()


def temporary_path(suffix):
    import folder_paths

    directory = folder_paths.get_temp_directory()
    os.makedirs(directory, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="usdu_video_", suffix=suffix, dir=directory)
    os.close(fd)
    return Path(path)


def from_path(path):
    from comfy_api.latest import InputImpl

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No video file at {path}")
    return InputImpl.VideoFromFile(str(path))


def source_path(video):
    # Calling the base VideoInput.get_stream_source() can itself materialize a
    # tensor-created video. Require the native file implementation explicitly.
    from comfy_api.latest import InputImpl

    if not isinstance(video, InputImpl.VideoFromFile):
        raise ValueError("Connect a file-backed VIDEO from Load Video or Video From Path.")
    start, duration = video.get_active_trim_window()
    if start or duration:
        raise ValueError("Save and reload a trimmed VIDEO before file processing; lazy trims are not supported.")
    # Core currently has no public crop accessor. Failing closed here prevents
    # get_stream_source() from silently discarding a native Crop Video operation.
    if getattr(video, "_VideoFromFile__crop", None) is not None:
        raise ValueError("Save and reload a cropped VIDEO before file processing; lazy crops are not supported.")
    source = video.get_stream_source()
    if not isinstance(source, str):
        raise ValueError("VIDEO must reference a file on disk, not an in-memory encoded buffer.")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"No video file at {path}")
    return path


def probe(path):
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if not rate or rate <= 0:
            raise ValueError(f"No valid frame rate in {path}")
        return {
            "width": stream.width, "height": stream.height,
            "rate": Fraction(rate), "time_base": stream.time_base,
            "audio": bool(container.streams.audio),
            "metadata": dict(container.metadata),
        }


def frames(path):
    """Yield (one BHWC float32 RGB frame, PTS seconds), with bounded residency."""
    import av
    import numpy as np
    import torch

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        previous = None
        for frame in container.decode(stream):
            check_interrupt()
            if frame.pts is None:
                raise ValueError("Video frames must have presentation timestamps.")
            timestamp = frame.pts * frame.time_base
            if previous is not None and timestamp <= previous:
                raise ValueError("Video presentation timestamps must be strictly increasing.")
            previous = timestamp
            if frame.rotation:
                raise ValueError("Apply the video's rotation before file processing.")
            pixels = frame.to_ndarray(format="rgb48le").astype(np.float32) / 65535.0
            yield torch.from_numpy(pixels).unsqueeze(0), timestamp


def timestamps(path):
    """Inspect the presentation clock without allocating RGB arrays/tensors."""
    import av

    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.video[0]):
            check_interrupt()
            if frame.pts is None:
                raise ValueError("Video frames must have presentation timestamps.")
            yield frame.pts * frame.time_base


class RGBWriter:
    """Lossless FFV1 RGB16 intermediate, avoiding YUV chroma subsampling."""

    def __init__(self, path, rate, time_base=None, metadata=None):
        import av

        self.container = av.open(str(path), mode="w")
        self.container.metadata.update(metadata or {})
        self.stream = None
        self.rate = Fraction(rate)
        self.time_base = time_base or Fraction(1, 1_000_000)
        self.count = 0
        self.previous = None

    def write(self, image, timestamp=None):
        import av
        import torch

        check_interrupt()
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Video processing must produce one RGB frame at a time.")
        height, width = image.shape[:2]
        if self.stream is None:
            self.stream = self.container.add_stream("ffv1", rate=self.rate)
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = "gbrp16le"
            self.stream.time_base = self.time_base
            self.stream.codec_context.time_base = self.time_base
            self.stream.options = {"level": "3"}
        elif (width, height) != (self.stream.width, self.stream.height):
            raise ValueError("Video processing changed dimensions between frames.")
        if timestamp is None:
            timestamp = Fraction(self.count, 1) / self.rate
        if self.previous is not None and timestamp <= self.previous:
            raise ValueError("Output timestamps must increase.")
        self.previous = timestamp
        pixels = (image.detach().to(device="cpu", dtype=torch.float32).clamp(0, 1)
                  * 65535).round().to(torch.uint16).contiguous().numpy()
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb48le")
        frame.pts = round(timestamp / self.time_base)
        frame.time_base = self.time_base
        self.container.mux(self.stream.encode(frame))
        self.count += 1

    def close(self):
        try:
            if self.stream is not None:
                self.container.mux(self.stream.encode())
        finally:
            self.container.close()


def copy_audio(rendered, source, output):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to preserve the source audio.")
    # Both inputs retain their source PTS; copyts keeps their relative offset.
    command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-copyts",
               "-i", str(rendered), "-i", str(source),
               "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
               "-map_chapters", "1", "-c", "copy", "-avoid_negative_ts", "disabled", str(output)]
    run_ffmpeg(command)


def run_ffmpeg(command):
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            while True:
                check_interrupt()
                try:
                    result = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if result:
                errors.seek(0)
                raise RuntimeError(f"Video encoding/mux failed: {errors.read().decode(errors='replace')[-4000:]}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


@contextmanager
def output_video(rate, *, source=None, time_base=None):
    """Keep the completed file for Comfy's cache; remove partial files on failure."""
    info = probe(source) if source else {}
    if info.get("audio") and not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to preserve the source audio.")
    path = temporary_path(".mkv")
    muxed = None
    writer = None
    try:
        writer = RGBWriter(path, rate, time_base, info.get("metadata"))
        yield writer, path
        if not writer.count:
            raise ValueError("The video contains no frames.")
        writer.close()
        writer = None
        if info.get("audio"):
            muxed = temporary_path(".mkv")
            copy_audio(path, source, muxed)
            os.replace(muxed, path)
    except BaseException:
        if writer is not None:
            writer.container.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        if muxed is not None:
            muxed.unlink(missing_ok=True)
