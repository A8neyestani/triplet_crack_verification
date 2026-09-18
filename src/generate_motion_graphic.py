"""Build the methodology GIF from the retained scene animations.

The scene GIFs are the final visual assets produced during the research
project. This script gives the repository one portable entry point for
rebuilding the full methodology timeline and, optionally, exporting it as an
MP4 when FFmpeg is installed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

from PIL import Image, ImageSequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MOTION_DIRECTORY = REPOSITORY_ROOT / "assets" / "motion"
DEFAULT_OUTPUT = MOTION_DIRECTORY / "full_methodology.gif"
DEFAULT_SIZE = (960, 540)
DEFAULT_FPS = 8

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9.1
    RESAMPLE_FILTER = Image.LANCZOS


def _fit_frame(frame: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Resize a frame to the requested canvas while preserving its aspect ratio."""

    target_width, target_height = size
    source_ratio = frame.width / frame.height
    target_ratio = target_width / target_height

    if source_ratio > target_ratio:
        crop_width = int(frame.height * target_ratio)
        left = (frame.width - crop_width) // 2
        frame = frame.crop((left, 0, left + crop_width, frame.height))
    elif source_ratio < target_ratio:
        crop_height = int(frame.width / target_ratio)
        top = (frame.height - crop_height) // 2
        frame = frame.crop((0, top, frame.width, top + crop_height))

    return frame.resize(size, RESAMPLE_FILTER).convert("RGB")


def load_scene_frames(filename: str, size: Tuple[int, int]) -> List[Image.Image]:
    """Load all frames from one scene GIF and normalize their canvas size."""

    path = MOTION_DIRECTORY / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing scene asset: {path}. Restore the canonical scene GIF before rendering."
        )

    with Image.open(path) as animation:
        frames = [_fit_frame(frame.convert("RGB"), size) for frame in ImageSequence.Iterator(animation)]

    if not frames:
        raise ValueError(f"Scene asset contains no frames: {path}")
    return frames


def render_problem_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("01_problem.gif", size)


def render_verification_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("02_verification.gif", size)


def render_triplet_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("03_triplet_training.gif", size)


def render_encoder_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("04_shared_encoder.gif", size)


def render_triplet_loss_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("05_triplet_loss.gif", size)


def render_embedding_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("06_embedding_space.gif", size)


def render_inference_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("07_inference.gif", size)


def render_result_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("08_verification_result.gif", size)


def render_monitoring_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("09_monitoring.gif", size)


def render_summary_scene(size: Tuple[int, int]) -> List[Image.Image]:
    return load_scene_frames("10_summary.gif", size)


SCENE_RENDERERS: Sequence[Callable[[Tuple[int, int]], List[Image.Image]]] = (
    render_problem_scene,
    render_verification_scene,
    render_triplet_scene,
    render_encoder_scene,
    render_triplet_loss_scene,
    render_embedding_scene,
    render_inference_scene,
    render_result_scene,
    render_monitoring_scene,
    render_summary_scene,
)


def render_timeline(size: Tuple[int, int]) -> List[Image.Image]:
    """Render the scenes in paper-methodology order."""

    frames: List[Image.Image] = []
    for render_scene in SCENE_RENDERERS:
        frames.extend(render_scene(size))
    return frames


def build_gif_palette(frames: Sequence[Image.Image]) -> Image.Image:
    """Build one palette from representative frames to reduce GIF dithering."""

    sample_width, sample_height = 240, 135
    sample = Image.new("RGB", (sample_width * 4, sample_height * 3))
    step = max(1, len(frames) // 12)
    for slot, frame in enumerate(frames[::step][:12]):
        sample.paste(frame.resize((sample_width, sample_height), RESAMPLE_FILTER),
                     ((slot % 4) * sample_width, (slot // 4) * sample_height))
    return sample.quantize(colors=256, method=Image.MEDIANCUT)


def export_gif(frames: Sequence[Image.Image], output: Path, fps: int) -> None:
    """Write RGB frames as a looping GIF."""

    output.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, round(1000 / fps))
    palette = build_gif_palette(frames)

    def quantize_frame(frame: Image.Image) -> Image.Image:
        quantized = frame.quantize(palette=palette, dither=Image.NONE)
        quantized.info.pop("transparency", None)
        return quantized

    first_frame = quantize_frame(frames[0])
    remaining = [quantize_frame(frame) for frame in frames[1:]]
    first_frame.save(
        output,
        format="GIF",
        save_all=True,
        append_images=remaining,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def export_mp4(frames: Sequence[Image.Image], output: Path, fps: int) -> None:
    """Pipe PNG frames to FFmpeg and write an H.264 MP4."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP4 export requires FFmpeg on PATH; use GIF output or install FFmpeg.")

    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]

    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        assert process.stdin is not None
        for frame in frames:
            buffer = BytesIO()
            frame.save(buffer, format="PNG")
            process.stdin.write(buffer.getvalue())
        process.stdin.close()
    except BrokenPipeError as error:
        process.kill()
        raise RuntimeError("FFmpeg stopped before all frames were written.") from error

    if process.wait() != 0:
        raise RuntimeError("FFmpeg failed while exporting the MP4.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output GIF or MP4 path")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Output frame rate (default: 8)")
    parser.add_argument("--width", type=int, default=DEFAULT_SIZE[0], help="Output width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_SIZE[1], help="Output height in pixels")
    parser.add_argument("--format", choices=("gif", "mp4"), help="Override the format inferred from --output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("fps, width, and height must be positive integers")

    output = args.output if args.output.is_absolute() else REPOSITORY_ROOT / args.output
    output_format = (args.format or output.suffix.lstrip(".") or "gif").lower()
    if output_format not in {"gif", "mp4"}:
        raise SystemExit("Output format must be gif or mp4")
    if output.suffix.lower() != f".{output_format}":
        output = output.with_suffix(f".{output_format}")

    frames = render_timeline((args.width, args.height))
    if output_format == "gif":
        export_gif(frames, output, args.fps)
    else:
        export_mp4(frames, output, args.fps)
    print(f"Wrote {len(frames)} frames to {output}")


if __name__ == "__main__":
    main()
