"""Compare peak process RSS for IMAGE, old VIDEO adapter and native VIDEO.

Set COMFYUI_ROOT and USDU_VIDEO_BENCH_DIR; run each mode in a fresh process:
USDU_VIDEO_BENCH_MODE=make|image|legacy_video|video|video_disk python -B test/benchmark_video_memory.py
This isolates pixel transport (mode_type=None); it does not measure model VRAM.
"""
import gc
import hashlib
import os
from pathlib import Path
import resource
import time
import json
import numpy as np
from video_unit_test import video, usdu, io, torch, folder_paths, settings

root = Path(os.environ['USDU_VIDEO_BENCH_DIR'])
root.mkdir(parents=True, exist_ok=True)
folder_paths.set_temp_directory(str(root))
torch.set_num_threads(4)
mode = os.environ['USDU_VIDEO_BENCH_MODE']
source = root / 'source.mkv'
frame_count = int(os.environ.get('USDU_VIDEO_BENCH_FRAMES', '22'))
if mode == 'make':
    writer = io.RGBWriter(source, 24)
    frame = torch.linspace(0.05, 0.95, 1920).view(1, 1920, 1).expand(1088, 1920, 3)
    for i in range(frame_count):
        writer.write(frame.roll(i * 13, dims=1))
    writer.close()
else:
    gc.collect()
    baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    options = settings(tile_width=960, tile_height=544)
    if mode in {'video', 'video_disk'}:
        result = video.UltimateSDUpscaleNoUpscaleGuiderVideo().refine(io.from_path(source), canvas_storage='disk' if mode == 'video_disk' else 'ram', **options)
        path = result[1]
    else:
        if mode == 'image':
            images = torch.empty((frame_count, 1088, 1920, 3))
        elif mode == 'legacy_video':
            mapped = np.memmap(root / 'input.f32', mode='w+', dtype=np.float32, shape=(frame_count, 1088, 1920, 3))
            images = torch.from_numpy(mapped)
        else:
            raise ValueError(mode)
        for i, (frame, _) in enumerate(io.frames(source)):
            images[i:i+1].copy_(frame)
        result = usdu.UltimateSDUpscaleNoUpscaleGuider().upscale(images, **options)[0]
        with io.output_video(24, source=source) as (writer, path):
            for i, timestamp in enumerate(io.timestamps(source)):
                writer.write(result[i], timestamp)
        del result, images
        if mode == 'legacy_video':
            mapped._mmap.close()
            (root / 'input.f32').unlink()
    digest = hashlib.sha256()
    for frame, _ in io.frames(path):
        digest.update(frame.numpy().tobytes())
    print(json.dumps(dict(mode=mode, frames=frame_count, baseline_rss_mib=baseline/1024,
                         peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                         seconds=time.perf_counter()-start, pixel_sha256=digest.hexdigest())), flush=True)
    Path(path).unlink()
