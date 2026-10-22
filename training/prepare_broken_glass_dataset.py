"""Prepare the labeled broken-glass dataset for YOLOv8 detection training.

The source dataset uses Roboflow/YOLO segmentation labels for many samples.
SkyVision-DL's detector expects YOLO detection labels, so this
script converts polygon labels into bounding boxes while preserving the
train/validation/test split.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "Training Data"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "models" / "datasets" / "processed_broken_glass_bbox"
DEFAULT_YAML_PATH = PROJECT_ROOT / "database" / "schemas" / "skyvision_yolov8.yaml"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert broken-glass segmentation labels to YOLO detection labels."
    )
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT), help="Input dataset root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Prepared dataset root.")
    parser.add_argument("--yaml-path", default=str(DEFAULT_YAML_PATH), help="Dataset YAML to write.")
    parser.add_argument("--overwrite", action="store_true", help="Replace the prepared dataset folder.")
    return parser.parse_args()


def safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return stem or "sample"


def unique_path(directory: Path, stem: str, suffix: str) -> Path:
    candidate = directory / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter:03d}{suffix}"
        counter += 1
    return candidate


def polygon_to_bbox(parts: list[str]) -> str | None:
    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
    except ValueError:
        return None

    if len(parts) == 5:
        xc, yc, width, height = values
    else:
        if len(values) < 6 or len(values) % 2 != 0:
            return None
        xs = values[0::2]
        ys = values[1::2]
        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        xc = (x_min + x_max) / 2
        yc = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    width = min(max(width, 0.001), 1.0)
    height = min(max(height, 0.001), 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}"


def convert_label_file(source: Path, destination: Path) -> tuple[int, int]:
    converted_lines: list[str] = []
    skipped = 0

    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            converted = polygon_to_bbox(stripped.split())
            if converted is None:
                skipped += 1
            else:
                converted_lines.append(converted)

    destination.write_text("\n".join(converted_lines) + ("\n" if converted_lines else ""), encoding="utf-8")
    return len(converted_lines), skipped


def prepare_split(source_root: Path, output_root: Path, source_split: str, target_split: str) -> dict[str, int]:
    source_images = source_root / source_split / "images"
    source_labels = source_root / source_split / "labels"
    target_images = output_root / "images" / target_split
    target_labels = output_root / "labels" / target_split
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    object_count = 0
    skipped_labels = 0
    missing_labels = 0

    for image_path in sorted(source_images.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        destination_image = unique_path(target_images, safe_stem(image_path.stem), image_path.suffix.lower())
        shutil.copy2(image_path, destination_image)

        source_label = source_labels / f"{image_path.stem}.txt"
        destination_label = target_labels / f"{destination_image.stem}.txt"
        if not source_label.exists():
            missing_labels += 1

        converted_count, skipped_count = convert_label_file(source_label, destination_label)
        image_count += 1
        object_count += converted_count
        skipped_labels += skipped_count

    return {
        "images": image_count,
        "objects": object_count,
        "skipped_labels": skipped_labels,
        "missing_labels": missing_labels,
    }


def write_dataset_yaml(yaml_path: Path, output_root: Path) -> None:
    relative_root = output_root.relative_to(PROJECT_ROOT).as_posix()
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {relative_root}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                "  0: broken_glass",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    yaml_path = Path(args.yaml_path).resolve()

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_root}")

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    summary = {}
    for source_split, target_split in SPLIT_MAP.items():
        summary[target_split] = prepare_split(source_root, output_root, source_split, target_split)

    write_dataset_yaml(yaml_path, output_root)

    print(f"Prepared dataset: {output_root}")
    print(f"Dataset YAML: {yaml_path}")
    for split, values in summary.items():
        print(
            f"{split}: images={values['images']} objects={values['objects']} "
            f"missing_labels={values['missing_labels']} skipped_labels={values['skipped_labels']}"
        )


if __name__ == "__main__":
    main()
