"""Video frame extraction utilities for SkyVision-DL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import cv2


SUPPORTED_EXTENSIONS = {".mp4", ".avi"}


class VideoExtractionError(RuntimeError):
    """Raised when video metadata or frames cannot be extracted."""


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    file_name: str
    extension: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float
    media_type: str = "video"

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedFrame:
    frame_index: int
    timestamp_seconds: float
    output_path: str

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def validate_video_path(video_path: str | Path) -> Path:
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")
    if not path.is_file():
        raise VideoExtractionError(f"Video path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise VideoExtractionError(f"Unsupported video format '{path.suffix}'. Supported formats: {supported}")
    return path


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    path = validate_video_path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoExtractionError(f"Unable to open video file: {path}")
    return capture


def extract_metadata(video_path: str | Path) -> VideoMetadata:
    path = validate_video_path(video_path)
    capture = open_video(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise VideoExtractionError(f"Invalid FPS reported for video: {path}")
        if frame_count <= 0:
            raise VideoExtractionError(f"Invalid frame count reported for video: {path}")
        if width <= 0 or height <= 0:
            raise VideoExtractionError(f"Invalid video dimensions reported for video: {path}")

        return VideoMetadata(
            path=str(path),
            file_name=path.name,
            extension=path.suffix.lower(),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=frame_count / fps,
        )
    finally:
        capture.release()


def iter_sampled_frame_indices(
    frame_count: int,
    fps: float,
    sample_every_n_frames: int | None = None,
    sample_fps: float | None = None,
) -> Iterator[int]:
    if frame_count <= 0:
        raise ValueError("frame_count must be greater than zero.")
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    if sample_every_n_frames is not None and sample_fps is not None:
        raise ValueError("Use either sample_every_n_frames or sample_fps, not both.")
    if sample_every_n_frames is not None and sample_every_n_frames <= 0:
        raise ValueError("sample_every_n_frames must be greater than zero.")
    if sample_fps is not None and sample_fps <= 0:
        raise ValueError("sample_fps must be greater than zero.")

    step = sample_every_n_frames
    if sample_fps is not None:
        step = max(1, round(fps / sample_fps))
    if step is None:
        step = 1

    yield from range(0, frame_count, step)


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_every_n_frames: int | None = None,
    sample_fps: float | None = None,
    image_extension: str = ".jpg",
    jpeg_quality: int = 95,
) -> tuple[VideoMetadata, list[ExtractedFrame]]:
    path = validate_video_path(video_path)
    metadata = extract_metadata(path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if not image_extension.startswith("."):
        image_extension = f".{image_extension}"

    capture = open_video(path)
    extracted_frames: list[ExtractedFrame] = []

    try:
        for frame_index in iter_sampled_frame_indices(
            frame_count=metadata.frame_count,
            fps=metadata.fps,
            sample_every_n_frames=sample_every_n_frames,
            sample_fps=sample_fps,
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success or frame is None:
                raise VideoExtractionError(f"Failed to read frame {frame_index} from video: {path}")

            timestamp_seconds = frame_index / metadata.fps
            output_path = destination / f"{path.stem}_frame_{frame_index:08d}{image_extension.lower()}"

            params: list[int] = []
            if image_extension.lower() in {".jpg", ".jpeg"}:
                params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

            if not cv2.imwrite(str(output_path), frame, params):
                raise VideoExtractionError(f"Failed to write frame to: {output_path}")

            extracted_frames.append(
                ExtractedFrame(
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    output_path=str(output_path),
                )
            )
    finally:
        capture.release()

    return metadata, extracted_frames
