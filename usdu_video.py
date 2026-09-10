"""Native VIDEO transport around the same Guider tile engine as IMAGE."""

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from . import usdu_video_io as video_io
from .usdu_nodes import UltimateSDUpscaleNoUpscaleGuider, shared
from .usdu_video_storage import DiskFrameCanvas
from usdu_utils import pil_to_tensor, tensor_to_pil


@dataclass
class VideoSource:
    path: Path
    info: dict
    canvas: object
    timestamps: list = field(default_factory=list)


class UltimateSDUpscaleNoUpscaleGuiderVideo(UltimateSDUpscaleNoUpscaleGuider):
    @classmethod
    def INPUT_TYPES(cls):
        schema = deepcopy(super().INPUT_TYPES())
        schema["required"] = {
            "video" if key == "upscaled_image" else key: ("VIDEO", *value[1:]) if key == "upscaled_image" else value
            for key, value in schema["required"].items()
        }
        schema.setdefault("optional", {})["canvas_storage"] = (["ram", "disk"], {
            "default": "ram",
            "tooltip": "Disk keeps RGB frames in temporary files and reads/writes tile regions. RAM keeps frames in memory. Sampling and pixels are identical; use fast local scratch storage for disk mode.",
        })
        schema["optional"]["canvas_directory"] = ("STRING", {
            "default": "",
            "tooltip": "Disk mode only: scratch folder on a local SSD. Empty uses ComfyUI's temp folder. Avoid RAM-backed folders (tmpfs) to reduce system RAM use. Temporary canvas files are removed after execution.",
        })
        return schema

    RETURN_TYPES = ("VIDEO", "STRING", "INT")
    RETURN_NAMES = ("video", "video_path", "frames")
    OUTPUT_TOOLTIPS = ("Lossless refined video.", "Temporary video file.", "Number of frames.")
    FUNCTION = "refine"
    CATEGORY = "video/upscaling"
    DESCRIPTION = ("Refines spatial tiles with the original USDU Guider engine. Streams VIDEO into and out of "
                   "its canvas without full float32 input/output clips. Disk canvas storage further reduces RAM use. "
                   "Keeps the whole clip in each tile; "
                   "sampling memory still depends on tile size and clip length. Lossless FFV1 output, with source audio.")

    def refine(self, video, canvas_storage="ram", canvas_directory="", **kwargs):
        import folder_paths

        if canvas_storage not in {"disk", "ram"}:
            raise ValueError("canvas_storage must be disk or ram.")
        path = video_io.source_path(video)
        info = video_io.probe(path)
        canvas = (DiskFrameCanvas(Path(canvas_directory).expanduser() if canvas_directory else folder_paths.get_temp_directory())
                  if canvas_storage == "disk" else nullcontext([]))
        with canvas as frames:
            source = VideoSource(path, info, frames)
            return super().upscale(upscaled_image=source, **kwargs)

    def _prepare_images(self, source):
        images = source.canvas
        expected = (source.info["height"], source.info["width"], 3)
        for frame, timestamp in video_io.frames(source.path):
            if tuple(frame.shape[1:]) != expected:
                raise ValueError("Video dimensions change within the scene.")
            images.append(tensor_to_pil(frame))
            source.timestamps.append(timestamp)
        if not images:
            raise ValueError("The video contains no frames.")
        return images, None

    def _finish_images(self, source):
        if len(shared.batch) != len(source.timestamps):
            raise ValueError("USDU changed the scene frame count.")
        with video_io.output_video(source.info["rate"], source=source.path,
                                   time_base=source.info["time_base"]) as (writer, path):
            for index, timestamp in enumerate(source.timestamps):
                writer.write(pil_to_tensor(shared.batch[index]), timestamp)
        return video_io.from_path(path), str(path), writer.count


NODE_CLASS_MAPPINGS = {"UltimateSDUpscaleNoUpscaleGuiderVideo": UltimateSDUpscaleNoUpscaleGuiderVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"UltimateSDUpscaleNoUpscaleGuiderVideo": "Ultimate SD Upscale (No Upscale, Guider, VIDEO)"}
