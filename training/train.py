"""Train SkyVision-DL YOLOv8 defect detection models.

Requirements:
    pip install ultralytics mlflow tensorboard
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import mlflow
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_YAML = PROJECT_ROOT / "database" / "schemas" / "skyvision_yolov8.yaml"
DEFAULT_RUN_DIR = PROJECT_ROOT / "models" / "runs" / "yolov8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLOv8 model for SkyVision-DL.")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model checkpoint or YAML.")
    parser.add_argument("--data", default=str(DEFAULT_DATASET_YAML), help="Dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=1280, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default="0", help="CUDA device index, 'cpu', or comma-separated GPU IDs.")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument("--patience", type=int, default=35, help="Early stopping patience.")
    parser.add_argument("--project", default=str(DEFAULT_RUN_DIR), help="YOLO output project directory.")
    parser.add_argument("--name", default="skyvision_yolov8_facade_defects", help="Experiment run name.")
    parser.add_argument("--optimizer", default="AdamW", help="Optimizer name.")
    parser.add_argument("--lr0", type=float, default=0.002, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005, help="Weight decay.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest checkpoint for this run.")
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI.")
    parser.add_argument("--mlflow-experiment", default="SkyVision-DL-YOLOv8", help="MLflow experiment name.")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("ULTRALYTICS_SETTINGS", "sync=False")
    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        str(PROJECT_ROOT / "models" / "runs" / "mlflow"),
    )


def train_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "patience": args.patience,
        "project": args.project,
        "name": args.name,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "save": True,
        "save_period": 10,
        "exist_ok": True,
        "resume": args.resume,
        "plots": True,
        "verbose": True,
        "cos_lr": True,
        "close_mosaic": 20,
        "box": 7.5,
        "cls": 0.7,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.35,
        "hsv_v": 0.25,
        "degrees": 5.0,
        "translate": 0.08,
        "scale": 0.35,
        "shear": 1.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.6,
        "mixup": 0.05,
        "erasing": 0.1,
    }


def main() -> None:
    args = parse_args()
    configure_environment(args)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")

    DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment(args.mlflow_experiment)

    params = train_args(args)
    model = YOLO(args.model)

    with mlflow.start_run(run_name=args.name):
        mlflow.log_params(
            {
                "model": args.model,
                "data": str(data_path),
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "device": args.device,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "weight_decay": args.weight_decay,
            }
        )

        results = model.train(**params)
        run_dir = Path(results.save_dir)
        weights_dir = run_dir / "weights"

        for weight_name in ("best.pt", "last.pt"):
            weight_path = weights_dir / weight_name
            if weight_path.exists():
                mlflow.log_artifact(str(weight_path), artifact_path="weights")

        tensorboard_dir = run_dir
        if tensorboard_dir.exists():
            mlflow.log_artifacts(str(tensorboard_dir), artifact_path="tensorboard")

        metrics = getattr(results, "results_dict", None)
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key.replace("(", "_").replace(")", ""), float(value))


if __name__ == "__main__":
    main()
