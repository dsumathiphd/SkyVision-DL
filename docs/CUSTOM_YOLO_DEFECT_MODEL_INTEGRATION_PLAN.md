# SkyVision-DL Custom YOLO Defect Model Integration Plan

## Purpose

This document defines the custom YOLO defect model integration plan for replacing a generic YOLO model with a trained facade glass defect detection model.

The current SkyVision-DL application pipeline is functional:

```text
Upload Video
-> Extract Frames
-> Run YOLO
-> Show Defects
-> Save History
-> Generate Reports
```

However, real facade glass defect detection requires a domain-trained model. The generic `yolov8n.pt` model should be treated only as a fallback for validating the processing pipeline.

## Why `yolov8n.pt` Cannot Detect Facade Glass Defects

`yolov8n.pt` is a general-purpose Ultralytics YOLOv8 model trained on common object categories such as people, vehicles, furniture, traffic objects, animals, and household objects.

It is not trained to recognize SkyVision-DL facade defect classes:

```text
0: broken_glass
1: crack
2: fracture
3: water_leakage
4: surface_stain
5: delamination
6: repair_patch
```

Therefore, when `yolov8n.pt` processes facade video frames, it tries to detect ordinary objects in the scene instead of glass defects. If many windows are broken, the model may still miss them because it has never learned the visual patterns of broken facade glass.

For correct broken-glass detection, SkyVision-DL must train a custom YOLOv8 model using facade inspection imagery annotated with the project defect classes.

## Dataset Requirements For Broken Glass Detection

### Minimum Dataset Scope

For a custom broken-glass detector:

```text
Minimum target: 500-1,000 annotated images
Recommended target: 2,000-5,000 annotated images
Production target: 10,000+ annotated images across buildings, lighting, weather, and camera types
```

### Required Image Sources

The dataset should include:

- Drone facade inspection video frames
- Ground camera facade images
- High-rise glass curtain wall images
- Close-up glass panel damage images
- Long-range facade inspection images
- Reflections, shadows, glare, dirt, and partial occlusion cases

### Broken Glass Positive Examples

Include broken glass cases with:

- Fully shattered panels
- Partially broken panels
- Impact-centered breakage
- Missing glass regions
- Spider-web fracture patterns
- Broken corners or edge damage
- Broken glass behind reflections
- Broken glass visible only in part of the frame

### Negative Examples

Include non-defective examples to reduce false positives:

- Clean glass windows
- Strong sky/building reflections
- Window frames and mullions
- Curtains/blinds behind glass
- Dirt and stains that are not broken glass
- Open windows
- Dark interiors
- Construction scaffolding

### Class Balance

For broken-glass improvement, prioritize:

```text
broken_glass: high priority
crack: high priority
fracture: high priority
water_leakage: medium priority
surface_stain: medium priority
delamination: medium priority
repair_patch: medium priority
```

If the initial goal is stronger broken-window accuracy, train first with strong coverage for `broken_glass`, then expand the remaining six defect classes.

## YOLOv8 Annotation Format

SkyVision-DL uses YOLOv8 object detection format.

Each image must have a matching `.txt` label file with one row per defect object:

```text
class_id x_center y_center width height
```

All coordinates are normalized between `0` and `1`.

Example:

```text
0 0.5124 0.4381 0.1820 0.2445
2 0.3810 0.5220 0.0905 0.1642
```

Meaning:

```text
0 -> broken_glass
2 -> fracture
```

### Class Mapping

The class mapping must match `database/schemas/skyvision_yolov8.yaml`:

```yaml
names:
  0: broken_glass
  1: crack
  2: fracture
  3: water_leakage
  4: surface_stain
  5: delamination
  6: repair_patch
```

### Dataset Folder Structure

Use this structure:

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

Example image-label pair:

```text
models/datasets/processed/images/train/facade_000123.jpg
models/datasets/processed/labels/train/facade_000123.txt
```

## Broken Glass Annotation Rules

Use bounding boxes around the visible damaged glass region only.

### Annotate As `broken_glass`

Annotate a bounding box when the image shows:

- Shattered glass
- Missing glass panel area
- Broken pane section
- Large impact damage with visible broken surface
- Window panel visibly destroyed or structurally compromised

### Do Not Annotate As `broken_glass`

Do not label:

- Normal window reflections
- Dark interior behind glass
- Dirt-only marks
- Window frame intersections
- Intact glass with no damage
- Open windows unless the glass itself is broken

### Multiple Broken Windows

If multiple window panels are broken, annotate each broken panel separately.

### Partial Broken Windows

If only part of a broken panel is visible, annotate the visible broken region and keep the box tight.

### Broken Glass vs Crack vs Fracture

Use this distinction:

```text
broken_glass: shattered, missing, or visibly destroyed glass area
crack: thin linear crack without large destroyed region
fracture: impact-centered damage or radiating crack cluster
```

## Training Command

Use the existing training script:

```powershell
python training/train.py --data database/schemas/skyvision_yolov8.yaml --model yolov8s.pt --epochs 150 --imgsz 1280 --batch 8 --device 0
```

For CPU-only training:

```powershell
python training/train.py --data database/schemas/skyvision_yolov8.yaml --model yolov8n.pt --epochs 50 --imgsz 960 --batch 4 --device cpu
```

Recommended first training run:

```powershell
python training/train.py --data database/schemas/skyvision_yolov8.yaml --model yolov8s.pt --epochs 100 --imgsz 1280 --batch 8 --device 0 --name skyvision_broken_glass
```

## Trained Weights Location

The training script saves runs under:

```text
models/runs/yolov8/
```

Typical trained weights path:

```text
models/runs/yolov8/skyvision_broken_glass/weights/best.pt
```

After validation, copy the selected best model to:

```text
models/weights/best.pt
```

This is the application model path.

## App Integration

Once `models/weights/best.pt` exists, update the inspection form Model Path field to:

```text
models/weights/best.pt
```

Or set the environment variable before starting the server:

```powershell
$env:SKYVISION_MODEL_PATH="models/weights/best.pt"
python -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

Expected app behavior after integration:

```text
Upload facade video
-> Extract frames
-> Run custom SkyVision-DL YOLO model
-> Detect broken_glass and other trained defect classes
-> Save detections
-> Show annotated frames
-> Generate PDF report with defect-specific results
```

## Validation Checklist

Before using the trained model for operational review:

- [ ] `models/weights/best.pt` exists
- [ ] Dataset YAML points to the correct processed dataset path
- [ ] Class order matches the SkyVision-DL class mapping
- [ ] Validation mAP50 is acceptable for operational review
- [ ] Broken-glass false negatives are reviewed manually
- [ ] Clean-window false positives are reviewed manually
- [ ] Inference runs with `device=cpu` for machines without GPU
- [ ] PDF report shows `broken_glass` detections instead of general object labels

## Recommended Next Work

The immediate next practical task is dataset preparation:

```text
Collect facade images
-> Annotate broken_glass boxes
-> Export YOLOv8 format
-> Place dataset under models/datasets/processed
-> Train with training/train.py
-> Copy best.pt to models/weights/best.pt
-> Run SkyVision-DL inspection again
```

No billing, digital twin, or enterprise admin dashboard changes are required for this step.
