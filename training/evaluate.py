"""Evaluate SkyVision-DL YOLOv8 defect detection models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_YAML = PROJECT_ROOT / "database" / "schemas" / "skyvision_yolov8.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models" / "runs" / "yolov8_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv8 model for SkyVision-DL.")
    parser.add_argument("--model", required=True, help="Path to trained YOLOv8 weights, e.g. best.pt.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET_YAML), help="Dataset YAML path.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--device", default="0", help="CUDA device index, 'cpu', or comma-separated GPU IDs.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold for validation.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold for NMS.")
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT_DIR), help="Evaluation output directory.")
    parser.add_argument("--name", default="skyvision_eval", help="Evaluation run name.")
    return parser.parse_args()


def harmonic_mean(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator == 0:
        return 0.0
    return 2 * precision * recall / denominator


def extract_metrics(results: Any) -> dict[str, float]:
    box = results.box
    precision = float(box.mp)
    recall = float(box.mr)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": harmonic_mean(precision, recall),
        "map50": float(box.map50),
        "map50_95": float(box.map),
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")


def validate_paths(model_path: Path, data_path: Path) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    validate_paths(model_path, data_path)

    model = YOLO(str(model_path))
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        project=args.project,
        name=args.name,
        plots=True,
        save_json=True,
        exist_ok=True,
    )

    metrics = extract_metrics(results)
    run_dir = Path(results.save_dir)
    save_metrics(metrics, run_dir / "metrics.json")

    print(json.dumps(metrics, indent=2))
    print(f"Confusion matrix saved in: {run_dir}")


if __name__ == "__main__":
    main()
