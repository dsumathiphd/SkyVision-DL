# SkyVision-DL YOLOv8 Training Architecture

## Target

Real-time facade defect detection for architectural glass inspection.

Primary model family:

- YOLOv8 detection

Primary deployment goal:

- High recall for safety-critical glass defects
- Low false positives under reflection, glare, rain, and long-range inspection conditions
- Real-time inference on GPU-enabled edge or workstation hardware

## 1. Dataset YAML

Recommended location:

```text
database/schemas/skyvision_yolov8.yaml
```

Dataset layout expected by YOLOv8:

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

Dataset YAML:

```yaml
path: ../../models/datasets/processed
train: images/train
val: images/val
test: images/test

names:
  0: broken_glass
  1: crack
  2: fracture
  3: water_leakage
  4: surface_stain
  5: delamination
  6: repair_patch
```

Class mapping:

| YOLO ID | Class | Code |
|---:|---|---|
| 0 | Broken Glass | `BRG` |
| 1 | Crack | `CRK` |
| 2 | Fracture | `FRC` |
| 3 | Water Leakage | `WTL` |
| 4 | Surface Stain | `STS` |
| 5 | Delamination | `DLM` |
| 6 | Repair Patch | `RPT` |

YOLO label format per image:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates must be normalized to `[0, 1]`.

Example:

```text
1 0.5124 0.3821 0.0712 0.0148
3 0.6410 0.5587 0.0925 0.2440
```

## 2. Training Configuration

Recommended training entry:

```text
backend/yolo/train_yolov8.py
```

Recommended experiment directory:

```text
models/runs/yolov8/
```

Baseline training command:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=database/schemas/skyvision_yolov8.yaml \
  epochs=150 \
  imgsz=1280 \
  batch=8 \
  device=0 \
  project=models/runs/yolov8 \
  name=skyvision_yolov8s_facade_defects
```

Recommended model scale:

| Model | Use Case |
|---|---|
| `yolov8n.pt` | Edge prototype, fastest inference, lower recall |
| `yolov8s.pt` | Recommended real-time baseline |
| `yolov8m.pt` | Higher accuracy workstation baseline |
| `yolov8l.pt` | Offline benchmark or high-accuracy inspection pipeline |

Recommended baseline:

```text
yolov8s.pt
```

Reason:

- Good balance between speed and accuracy
- Suitable for real-time facade inspection
- Better small-defect capacity than nano while still deployable

Training phases:

| Phase | Purpose | Epochs |
|---|---|---:|
| Warm-up baseline | Validate labels and pipeline | 20 |
| Main training | Learn defect detector | 150 |
| High-resolution fine-tuning | Improve small crack and stain recall | 30-50 |
| Hard-negative fine-tuning | Reduce reflection false positives | 20-30 |

Recommended two-stage training:

```bash
yolo detect train \
  model=yolov8s.pt \
  data=database/schemas/skyvision_yolov8.yaml \
  epochs=150 \
  imgsz=960 \
  batch=12 \
  project=models/runs/yolov8 \
  name=skyvision_yolov8s_stage1
```

```bash
yolo detect train \
  model=models/runs/yolov8/skyvision_yolov8s_stage1/weights/best.pt \
  data=database/schemas/skyvision_yolov8.yaml \
  epochs=50 \
  imgsz=1280 \
  batch=8 \
  lr0=0.001 \
  mosaic=0.2 \
  project=models/runs/yolov8 \
  name=skyvision_yolov8s_stage2_highres
```

## 3. Hyperparameters

Recommended baseline hyperparameters:

```yaml
model: yolov8s.pt
task: detect
mode: train

epochs: 150
patience: 35
batch: 8
imgsz: 1280
device: 0
workers: 8
seed: 42
deterministic: true

optimizer: AdamW
lr0: 0.002
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 5
warmup_momentum: 0.8
warmup_bias_lr: 0.1

box: 7.5
cls: 0.7
dfl: 1.5

close_mosaic: 20
overlap_mask: true
single_cls: false
rect: false
cos_lr: true
```

Small-defect tuning:

| Parameter | Recommended Value | Purpose |
|---|---:|---|
| `imgsz` | `1280-1536` | Preserve small cracks and thin defects |
| `batch` | GPU-dependent | Use largest stable batch |
| `box` | `7.5-9.0` | Improve localization precision |
| `cls` | `0.7-1.0` | Improve rare-class separation |
| `close_mosaic` | `15-25` | Reduce mosaic artifacts late in training |
| `patience` | `30-50` | Allow rare classes to stabilize |

Class imbalance strategy:

- Oversample rare classes in training image selection.
- Add more hard negatives for reflection-heavy clean glass.
- Monitor per-class AP rather than only aggregate mAP.
- Use active learning to add missed cracks, delamination, and repair patches.

Recommended inference thresholds:

```yaml
conf: 0.25
iou: 0.50
max_det: 300
agnostic_nms: false
```

Deployment threshold tuning:

| Use Case | Confidence Threshold |
|---|---:|
| Safety-first inspection | `0.15-0.25` |
| Balanced dashboard detection | `0.25-0.35` |
| Low false-positive reporting | `0.40-0.55` |

## 4. Data Augmentation Strategy

Facade defect detection requires augmentation that improves robustness without destroying subtle visual evidence.

### 4.1 Recommended Augmentations

```yaml
hsv_h: 0.015
hsv_s: 0.35
hsv_v: 0.25
degrees: 5.0
translate: 0.08
scale: 0.35
shear: 1.0
perspective: 0.0005
flipud: 0.0
fliplr: 0.5
mosaic: 0.6
mixup: 0.05
copy_paste: 0.0
erasing: 0.1
crop_fraction: 1.0
```

### 4.2 Augmentation Rationale

| Augmentation | Recommendation | Reason |
|---|---|---|
| Horizontal flip | Use | Facade defects are generally orientation-tolerant |
| Vertical flip | Avoid | Gravity-driven leakage would become unrealistic |
| Mild rotation | Use | Drone and handheld images may be tilted |
| Perspective | Use lightly | Simulates oblique facade inspection |
| HSV brightness | Use moderately | Handles sun, shadow, overcast, and dusk |
| Mosaic | Use early, reduce late | Improves robustness but may distort tiny cracks |
| MixUp | Use lightly | Helps generalization, but too much weakens fine defects |
| Copy-paste | Avoid by default | Artificial crack placement can create unrealistic defects |
| Random erasing | Use lightly | Simulates occlusion, glare, and partial visibility |

### 4.3 Domain-Specific Augmentations

Recommended custom augmentations:

- Mild glare simulation
- Reflection-like brightness streaks
- Motion blur for drone imagery
- Defocus blur for long-range telephoto images
- JPEG compression artifacts
- Rain droplet hard negatives
- Low-light brightness and contrast variation

Use cautiously:

- Strong blur, because small cracks may disappear
- Heavy color shifts, because stain and leakage appearance matters
- Aggressive random crop, because defects may be cut into ambiguous fragments

### 4.4 Class-Specific Considerations

| Class | Augmentation Sensitivity |
|---|---|
| Broken Glass | Robust to moderate augmentation |
| Crack | Sensitive to blur, compression, and downscaling |
| Fracture | Sensitive to excessive mosaic and resizing |
| Water Leakage | Sensitive to vertical flip and heavy color shift |
| Surface Stain | Sensitive to excessive brightness and contrast changes |
| Delamination | Sensitive to glare simulation and blur |
| Repair Patch | Robust, but avoid unrealistic copy-paste |

### 4.5 Hard Negative Strategy

Include clean or non-target images containing:

- Reflected cables
- Tree branch reflections
- Railings
- Mullion lines
- Strong glare
- Cloud reflection
- Rain droplets
- Dirt patterns that are not labeled defects
- Interior objects behind glass

Hard negatives are essential for reducing false positives in real facade deployments.

## 5. Evaluation Metrics

### 5.1 Core YOLO Metrics

Report:

- Precision
- Recall
- F1 score
- mAP@0.50
- mAP@0.50:0.95
- Per-class AP
- Per-class recall
- Confusion matrix

Primary model-selection metric:

```text
mAP@0.50:0.95 + high-severity defect recall
```

Safety-critical monitoring metric:

```text
Recall for Broken Glass, Crack, and Fracture
```

### 5.2 Real-Time Deployment Metrics

Report:

- FPS
- Mean inference latency
- P95 inference latency
- GPU memory usage
- Model size
- False positives per image
- False positives per facade elevation
- Miss rate for high-severity defects

Recommended deployment targets:

| Metric | Target |
|---|---:|
| Inference speed | `>= 25 FPS` on target GPU |
| Image size | `960-1280 px` |
| Broken Glass recall | `>= 0.95` |
| Crack recall | `>= 0.85` |
| Fracture recall | `>= 0.90` |
| False positives | Tune per deployment risk tolerance |

### 5.3 Subset Evaluation

Evaluate separately on:

- Standard test set
- Low-light images
- High-reflection images
- Long-range images
- Wet-weather images
- Rare-defect images
- Hard-negative clean images

Subset metrics:

```text
AP_class_subset
Recall_class_subset
False positives per image_subset
```

### 5.4 Small-Object Evaluation

For cracks and small delamination regions, report:

- AP-small
- Recall-small
- Miss rate for boxes below 32 px
- Localization error for thin defects

Small-object performance should be tracked independently because aggregate mAP can hide poor crack detection.

### 5.5 Acceptance Criteria

A model is acceptable for controlled pilot deployment only if:

- Test set mAP@0.50 is stable across all seven classes.
- High-severity Broken Glass recall is near-perfect.
- Crack and Fracture recall remain acceptable under reflection.
- Hard-negative false positives are reviewed and below operational tolerance.
- Real-time latency meets the target hardware requirement.

