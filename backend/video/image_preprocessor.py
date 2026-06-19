"""Image preprocessing utilities for SkyVision-DL video frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


InterpolationMode = Literal["nearest", "linear", "cubic", "area", "lanczos"]


INTERPOLATION_MAP: dict[InterpolationMode, int] = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


class ImagePreprocessingError(RuntimeError):
    """Raised when image preprocessing fails."""


@dataclass(frozen=True)
class PreprocessingConfig:
    target_width: int = 1280
    target_height: int = 1280
    keep_aspect_ratio: bool = True
    pad_color: tuple[int, int, int] = (114, 114, 114)
    interpolation: InterpolationMode = "linear"
    enhance_contrast: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    denoise: bool = True
    denoise_strength: int = 5
    reduce_blur: bool = True
    sharpen_amount: float = 0.7

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise ImagePreprocessingError(f"Image path is not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ImagePreprocessingError(f"Unable to read image: {path}")
    return image


def write_image(image_path: str | Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path = Path(image_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    params: list[int] = []
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]

    if not cv2.imwrite(str(path), image, params):
        raise ImagePreprocessingError(f"Unable to write image: {path}")


def validate_image(image: np.ndarray) -> None:
    if image is None:
        raise ImagePreprocessingError("Image is None.")
    if not isinstance(image, np.ndarray):
        raise TypeError("Image must be a numpy.ndarray.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ImagePreprocessingError("Image must be a BGR color image with shape HxWx3.")
    if image.size == 0:
        raise ImagePreprocessingError("Image is empty.")


def resize_image(
    image: np.ndarray,
    width: int,
    height: int,
    keep_aspect_ratio: bool = True,
    pad_color: tuple[int, int, int] = (114, 114, 114),
    interpolation: InterpolationMode = "linear",
) -> np.ndarray:
    validate_image(image)
    if width <= 0 or height <= 0:
        raise ValueError("Target width and height must be greater than zero.")

    interpolation_flag = INTERPOLATION_MAP[interpolation]
    original_height, original_width = image.shape[:2]

    if not keep_aspect_ratio:
        return cv2.resize(image, (width, height), interpolation=interpolation_flag)

    scale = min(width / original_width, height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation_flag)
    canvas = np.full((height, width, 3), pad_color, dtype=image.dtype)

    x_offset = (width - resized_width) // 2
    y_offset = (height - resized_height) // 2
    canvas[y_offset : y_offset + resized_height, x_offset : x_offset + resized_width] = resized
    return canvas


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    validate_image(image)
    if clip_limit <= 0:
        raise ValueError("clip_limit must be greater than zero.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_lightness = clahe.apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, channel_a, channel_b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def denoise_image(image: np.ndarray, strength: int = 5) -> np.ndarray:
    validate_image(image)
    if strength <= 0:
        raise ValueError("Denoise strength must be greater than zero.")

    return cv2.fastNlMeansDenoisingColored(
        image,
        None,
        h=float(strength),
        hColor=float(strength),
        templateWindowSize=7,
        searchWindowSize=21,
    )


def reduce_blur(image: np.ndarray, sharpen_amount: float = 0.7) -> np.ndarray:
    validate_image(image)
    if sharpen_amount < 0:
        raise ValueError("sharpen_amount must be non-negative.")
    if sharpen_amount == 0:
        return image.copy()

    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image, 1.0 + sharpen_amount, blurred, -sharpen_amount, 0)


def normalize_resolution(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    return resize_image(
        image=image,
        width=config.target_width,
        height=config.target_height,
        keep_aspect_ratio=config.keep_aspect_ratio,
        pad_color=config.pad_color,
        interpolation=config.interpolation,
    )


def preprocess_image(image: np.ndarray, config: PreprocessingConfig | None = None) -> np.ndarray:
    validate_image(image)
    config = config or PreprocessingConfig()

    processed = image.copy()
    if config.denoise:
        processed = denoise_image(processed, strength=config.denoise_strength)
    if config.enhance_contrast:
        processed = enhance_contrast(
            processed,
            clip_limit=config.clahe_clip_limit,
            tile_grid_size=config.clahe_tile_grid_size,
        )
    if config.reduce_blur:
        processed = reduce_blur(processed, sharpen_amount=config.sharpen_amount)

    return normalize_resolution(processed, config)


def preprocess_file(
    input_path: str | Path,
    output_path: str | Path,
    config: PreprocessingConfig | None = None,
    jpeg_quality: int = 95,
) -> Path:
    image = read_image(input_path)
    processed = preprocess_image(image, config=config)
    write_image(output_path, processed, jpeg_quality=jpeg_quality)
    return Path(output_path).expanduser().resolve()
