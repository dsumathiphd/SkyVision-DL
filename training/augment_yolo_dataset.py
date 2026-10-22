"""Create labeled YOLO dataset augmentations for SkyVision-DL training."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "models" / "datasets" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Augment YOLO images and labels for SkyVision-DL.")
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT), help="YOLO dataset root.")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="Dataset split to augment.")
    parser.add_argument("--prefix", default="aug", help="Filename prefix for generated augmented samples.")
    return parser.parse_args()


def read_labels(label_path: Path) -> list[list[float]]:
    if not label_path.exists():
        return []

    labels: list[list[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append([float(parts[0]), *[float(value) for value in parts[1:]]])
    return labels


def write_labels(label_path: Path, labels: list[list[float]]) -> None:
    lines = [
        f"{int(class_id)} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"
        for class_id, xc, yc, width, height in labels
    ]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clip_labels(labels: list[list[float]]) -> list[list[float]]:
    clipped = []
    for class_id, xc, yc, width, height in labels:
        xc = min(max(xc, 0.0), 1.0)
        yc = min(max(yc, 0.0), 1.0)
        width = min(max(width, 0.001), 1.0)
        height = min(max(height, 0.001), 1.0)
        clipped.append([class_id, xc, yc, width, height])
    return clipped


def horizontal_flip(image: np.ndarray, labels: list[list[float]]) -> tuple[np.ndarray, list[list[float]]]:
    flipped = cv2.flip(image, 1)
    flipped_labels = [[class_id, 1.0 - xc, yc, width, height] for class_id, xc, yc, width, height in labels]
    return flipped, clip_labels(flipped_labels)


def adjust_brightness_contrast(image: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_noise(image: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    noise = np.random.default_rng(42).normal(0, sigma, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def sharpen(image: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)


def augmentations(image: np.ndarray, labels: list[list[float]]) -> list[tuple[str, np.ndarray, list[list[float]]]]:
    flipped_image, flipped_labels = horizontal_flip(image, labels)
    return [
        ("flip", flipped_image, flipped_labels),
        ("bright", adjust_brightness_contrast(image, alpha=1.12, beta=12), labels),
        ("dark", adjust_brightness_contrast(image, alpha=0.88, beta=-10), labels),
        ("contrast", adjust_brightness_contrast(image, alpha=1.28, beta=0), labels),
        ("blur", cv2.GaussianBlur(image, (3, 3), 0), labels),
        ("noise", add_noise(image), labels),
        ("sharp", sharpen(image), labels),
    ]


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    image_dir = dataset_root / "images" / args.split
    label_dir = dataset_root / "labels" / args.split

    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"Missing image or label directory for split: {args.split}")

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and not path.stem.startswith(f"{args.prefix}_")
    )
    created = 0

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        labels = read_labels(label_path)
        if not labels:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        for aug_name, aug_image, aug_labels in augmentations(image, labels):
            output_stem = f"{args.prefix}_{aug_name}_{image_path.stem}"
            output_image_path = image_dir / f"{output_stem}{image_path.suffix.lower()}"
            output_label_path = label_dir / f"{output_stem}.txt"

            cv2.imwrite(str(output_image_path), aug_image)
            write_labels(output_label_path, clip_labels(aug_labels))
            created += 1

    print(f"Created {created} augmented labeled samples in split '{args.split}'.")


if __name__ == "__main__":
    main()
