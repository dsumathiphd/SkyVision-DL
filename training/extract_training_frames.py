"""Extract annotation-ready training frames from facade inspection videos."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.video.frame_extractor import SUPPORTED_EXTENSIONS, extract_frames  # noqa: E402


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "datasets" / "raw" / "frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract training frames from MP4/AVI facade videos for YOLO annotation."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more video files or directories containing MP4/AVI videos.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where extracted frames and manifest.csv will be saved.",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
        help="Number of frames to sample per second from each video.",
    )
    parser.add_argument(
        "--image-extension",
        default=".jpg",
        choices=(".jpg", ".jpeg", ".png"),
        help="Output image format for extracted frames.",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100 when saving .jpg/.jpeg frames.",
    )
    return parser.parse_args()


def discover_videos(inputs: list[str]) -> list[Path]:
    videos: list[Path] = []
    for input_value in inputs:
        path = Path(input_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")

        if path.is_file():
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                videos.append(path)
            continue

        for extension in sorted(SUPPORTED_EXTENSIONS):
            videos.extend(sorted(path.rglob(f"*{extension}")))
            videos.extend(sorted(path.rglob(f"*{extension.upper()}")))

    unique_videos = sorted({video.resolve() for video in videos})
    if not unique_videos:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FileNotFoundError(f"No supported videos found. Supported formats: {supported}")
    return unique_videos


def write_manifest(manifest_path: Path, rows: list[dict[str, str | int | float]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "video_file",
        "video_path",
        "fps",
        "frame_count",
        "duration_seconds",
        "frame_index",
        "timestamp_seconds",
        "frame_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than zero.")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 1 and 100.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(args.inputs)
    manifest_rows: list[dict[str, str | int | float]] = []
    total_frames = 0

    for video_path in videos:
        video_output_dir = output_dir / video_path.stem
        metadata, frames = extract_frames(
            video_path=video_path,
            output_dir=video_output_dir,
            sample_fps=args.sample_fps,
            image_extension=args.image_extension,
            jpeg_quality=args.jpeg_quality,
        )
        total_frames += len(frames)

        for frame in frames:
            manifest_rows.append(
                {
                    "video_file": metadata.file_name,
                    "video_path": metadata.path,
                    "fps": metadata.fps,
                    "frame_count": metadata.frame_count,
                    "duration_seconds": round(metadata.duration_seconds, 4),
                    "frame_index": frame.frame_index,
                    "timestamp_seconds": round(frame.timestamp_seconds, 4),
                    "frame_path": frame.output_path,
                }
            )

        print(f"Extracted {len(frames)} frames from {video_path.name} -> {video_output_dir}")

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"Done. Extracted {total_frames} frames from {len(videos)} video(s).")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
