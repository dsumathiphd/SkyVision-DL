# SkyVision-DL Annotation Preparation

This folder stores training images and YOLOv8 labels for the custom facade glass defect model.

## Step 1: Extract Training Frames

From the project root:

```powershell
python training/extract_training_frames.py "C:\path\to\facade_video.mp4" --sample-fps 1.0
```

For a folder containing MP4/AVI videos:

```powershell
python training/extract_training_frames.py "C:\path\to\videos" --sample-fps 1.0
```

Frames are saved to:

```text
models/datasets/raw/frames/
```

A manifest is saved to:

```text
models/datasets/raw/frames/manifest.csv
```

## Step 2: Annotate Frames

Use CVAT or Label Studio.

Create bounding boxes for the SkyVision-DL classes:

```text
0: broken_glass
1: crack
2: fracture
3: water_leakage
4: surface_stain
5: delamination
6: repair_patch
```

For the immediate broken-glass model, focus first on accurate `broken_glass` boxes.

## Step 3: Export YOLOv8 Format

Export annotations in YOLO format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized between `0` and `1`.

## Step 4: Place Processed Dataset

Use this final training structure:

```text
models/datasets/processed/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each image must have a matching label file:

```text
models/datasets/processed/images/train/facade_000001.jpg
models/datasets/processed/labels/train/facade_000001.txt
```

## Step 5: Train After Annotation

Do not train until the processed dataset contains annotated image/label pairs.

Training command:

```powershell
python training/train.py --data database/schemas/skyvision_yolov8.yaml --model yolov8s.pt --epochs 100 --imgsz 1280 --batch 8 --device 0 --name skyvision_broken_glass
```

For CPU-only training:

```powershell
python training/train.py --data database/schemas/skyvision_yolov8.yaml --model yolov8n.pt --epochs 50 --imgsz 960 --batch 4 --device cpu --name skyvision_broken_glass_cpu
```
