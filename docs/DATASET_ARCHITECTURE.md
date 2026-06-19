# SkyVision-DL Dataset Architecture

## Research-Grade Dataset Design Document

Project: AI-powered facade glass defect detection system

Defect classes:

1. Broken Glass
2. Crack
3. Fracture
4. Water Leakage
5. Surface Stain
6. Delamination
7. Repair Patch

## 1. Dataset Specification

### 1.1 Dataset Objective

SkyVision-DL is a computer vision dataset for automated inspection of architectural facade glass. The dataset is designed to train, validate, and benchmark AI models that detect and localize visible defects on glass facade systems under realistic inspection conditions.

The dataset must support:

- Object detection
- Instance segmentation
- Semantic segmentation
- Multi-label image classification
- Defect severity estimation
- Temporal defect tracking
- Inspection reporting and maintenance prioritization

Primary research task:

> Detect, classify, and localize facade glass defects from drone, ground, telephoto, and inspection-archive imagery.

### 1.2 Image Sources

Accepted image sources:

- Drone facade inspection imagery
- Ground-level inspection cameras
- Telephoto facade survey cameras
- Mobile maintenance inspection photos
- Fixed monitoring cameras
- Historical inspection archives

### 1.3 Capture Diversity

The dataset should include variation across:

- Building type: commercial towers, residential towers, institutional buildings, malls, airports, industrial structures
- Facade system: curtain wall, window wall, storefront glazing, skylight, atrium glass
- Viewpoint: frontal, oblique, upward-looking, downward-looking
- Distance: close-range, mid-range, long-range
- Lighting: direct sun, overcast, shadow, dusk, low light, night lighting
- Weather: dry, wet, post-rain, humid
- Reflection level: low reflection, high reflection, sky reflection, urban reflection
- Glass type: clear, tinted, coated, laminated, insulated glass units
- Defect scale: tiny, small, medium, large

### 1.4 Dataset Entities

The dataset should be organized around the following entities:

| Entity | Description |
|---|---|
| Building | Physical inspection site |
| Inspection | Capture session for a building |
| Image | Individual inspection frame or photograph |
| Panel | Individual glass panel or facade module, when identifiable |
| Defect instance | Localized visible defect occurrence |

Each defect instance should be traceable to:

- `building_id`
- `inspection_id`
- `image_id`
- `panel_id`, when available
- `defect_tracking_id`, when tracked across time
- annotation version
- reviewer status

### 1.5 Primary Annotation Format

Canonical format:

- COCO-style JSON with detection and segmentation extensions

Supported export formats:

- YOLO detection labels
- Pascal VOC XML
- PNG semantic masks
- PNG instance masks
- CSV metadata tables

The COCO-style annotation file should remain the source of truth for official releases.

### 1.6 Required Metadata

Each image record should include:

- `image_id`
- `file_name`
- `width`
- `height`
- `building_id`
- `inspection_id`
- `camera_type`
- `capture_timestamp`
- `facade_elevation`
- `weather_condition`
- `lighting_condition`
- `reflection_level`
- `distance_range`
- `source_type`
- `split`

Each annotation record should include:

- `annotation_id`
- `image_id`
- `category_id`
- `category_name`
- `bbox`
- `segmentation`, when available
- `area`
- `severity`
- `occlusion`
- `visibility_confidence`
- `panel_id`, when available
- `defect_tracking_id`, when available
- `review_status`

## 2. Class Definitions

### 2.1 Broken Glass

Definition:

Broken Glass is a visible failure state where the glass panel has shattered, lost material, opened, collapsed, or developed severe physical discontinuity.

Visual indicators:

- Missing glass area
- Shattered fragments
- Splintered or jagged boundary
- Hole through the panel
- Detached glass pieces
- Severe panel collapse

Include:

- Fully shattered glass
- Partially missing glass
- Broken corner with material loss
- Exposed void caused by glass failure

Exclude:

- Single thin line damage; classify as `Crack`
- Dense crack network without missing material; classify as `Fracture`
- Reflection artifacts that resemble holes
- Visible repair material; classify as `Repair Patch`

Severity:

| Severity | Definition |
|---|---|
| Low | Small chip or broken corner |
| Medium | Localized broken area, panel mostly intact |
| High | Large missing or shattered region, safety-critical condition |

### 2.2 Crack

Definition:

Crack is a narrow linear discontinuity on or within the glass where the panel remains mostly intact.

Visual indicators:

- Thin dark or bright line
- Linear, curved, or slightly branching path
- Edge-originating line
- Sharp contrast against glass
- No major material loss

Include:

- Hairline crack
- Single long crack
- Edge crack
- Sparse branching crack

Exclude:

- Dense impact pattern; classify as `Fracture`
- Sealant, gasket, mullion, or panel boundary line
- Reflections of cables, trees, rails, or facade edges
- Cosmetic scratch without evidence of glass failure

Severity:

| Severity | Definition |
|---|---|
| Low | Short localized hairline crack |
| Medium | Long crack across part of the panel |
| High | Crack spanning most of the panel or reaching multiple edges |

### 2.3 Fracture

Definition:

Fracture is a complex structural damage pattern consisting of multiple cracks, radial lines, branching failures, or stress networks.

Visual indicators:

- Spiderweb crack pattern
- Radial crack network
- Multiple intersecting cracks
- Dense crack cluster
- Impact-centered break pattern

Include:

- Impact fracture
- Multi-branch crack network
- Thermal stress fracture pattern
- Structural crack cluster

Exclude:

- Single isolated line; classify as `Crack`
- Missing or collapsed glass; classify as `Broken Glass`
- Reflection patterns resembling crack networks

Severity:

| Severity | Definition |
|---|---|
| Low | Small localized fracture cluster |
| Medium | Fracture covering moderate panel area |
| High | Extensive fracture network across most of the panel |

### 2.4 Water Leakage

Definition:

Water Leakage is visible moisture ingress, seepage, runoff, dripping, or moisture-related marking associated with facade glass, glazing joints, seals, or panel boundaries.

Visual indicators:

- Vertical water trail
- Wet patch near seal or joint
- Drip line
- Moisture accumulation near panel edge
- Mineral deposit following a leakage path
- Repeated runoff pattern

Include:

- Active water seepage
- Water trail from glazing joint
- Wet stain below a facade opening
- Moisture residue when the leakage path is visually clear

Exclude:

- Uniform rain droplets across glass
- Sky or cloud reflections
- General dirt without leakage pattern
- Dry contamination; classify as `Surface Stain`

Severity:

| Severity | Definition |
|---|---|
| Low | Minor localized water mark |
| Medium | Clear leakage trail or repeated moisture path |
| High | Heavy leakage, multiple trails, or large wet region |

### 2.5 Surface Stain

Definition:

Surface Stain is visible exterior contamination, discoloration, grime, mineral residue, pollution deposit, biological growth, or cleaning residue on glass.

Visual indicators:

- Cloudy dirty area
- Hard-water spotting
- Brown, black, white, or green discoloration
- Smudge or residue
- Pollution streak
- Irregular exterior deposit

Include:

- Dirt buildup
- Hard-water stain
- Rust-colored runoff residue
- Cleaning residue
- Biological growth on glass surface

Exclude:

- Active or path-like moisture defect; classify as `Water Leakage`
- Internal layer haze; classify as `Delamination`
- Repair material; classify as `Repair Patch`
- Reflection or shadow without physical contamination

Severity:

| Severity | Definition |
|---|---|
| Low | Light visible stain |
| Medium | Clearly visible localized staining |
| High | Large or opaque stain affecting inspection visibility |

### 2.6 Delamination

Definition:

Delamination is internal layer separation, bubbling, edge whitening, haze, coating separation, or film failure in laminated, coated, or insulated glass.

Visual indicators:

- Milky internal patch
- Edge whitening
- Bubble-like pattern
- Internal film separation
- Cloudy non-surface haze
- Laminated layer distortion

Include:

- Laminated glass layer separation
- Coating peel
- Edge delamination
- Internal bubbles
- Layer failure in insulated or laminated glass

Exclude:

- External dirt; classify as `Surface Stain`
- Weather condensation without material failure
- Reflection or glare
- Applied repair material unless visibly delaminating

Severity:

| Severity | Definition |
|---|---|
| Low | Small edge haze or bubble |
| Medium | Localized delaminated region |
| High | Large cloudy or separated layer region |

### 2.7 Repair Patch

Definition:

Repair Patch is visible material applied to glass or adjacent facade areas as temporary or permanent repair, sealing, reinforcement, protection, or replacement marking.

Visual indicators:

- Adhesive tape
- Resin-filled patch
- Silicone or sealant smear
- Temporary plastic covering
- Applied film
- Rectangular or irregular repair material

Include:

- Tape over cracked glass
- Sealant patch
- Resin repair
- Temporary protective cover
- Localized replacement film

Exclude:

- Original tint or architectural film
- Stickers or signage unrelated to repair
- Normal gaskets, mullions, or facade components
- Dirt or stain

Severity:

| Severity | Definition |
|---|---|
| Low | Small localized repair |
| Medium | Visible repair covering a defect area |
| High | Large temporary patch indicating serious prior damage |

## 3. Annotation Standards

### 3.1 General Rules

Annotate visible evidence only. Do not annotate inferred damage outside the visible region.

Annotators must:

- Mark only visible defects.
- Avoid labeling reflections as defects.
- Use the most specific class possible.
- Annotate each distinct defect as a separate instance.
- Include the full visible extent of each defect.
- Use uncertainty metadata when visibility is poor.

### 3.2 Bounding Box Standards

Every defect instance must have a bounding box.

Bounding boxes should:

- Tightly enclose the visible defect.
- Include all visible crack or fracture branches.
- Avoid unnecessary surrounding glass.
- Cover only the visible stain, leakage, delamination, broken region, or patch.
- Remain inside image boundaries.

Minimum recommended box size:

- `8 x 8 px` for standard training use
- Smaller defects may be retained only when visually unambiguous

### 3.3 Segmentation Standards

Instance masks are strongly recommended for all defect classes.

Mask rules:

- Follow visible defect boundaries.
- Use narrow masks for cracks.
- Use area masks for broken glass, stains, leakage, delamination, fractures, and patches.
- For very thin cracks, store an optional polyline skeleton where supported.
- Do not include unaffected background or full panels unless fully affected.

### 3.4 Class Conflict Rules

| Situation | Annotation Rule |
|---|---|
| Single linear defect | Label as `Crack` |
| Dense network of cracks | Label as `Fracture` |
| Missing glass with surrounding cracks | Label missing region as `Broken Glass`; label surrounding network as `Fracture` if distinct |
| Water trail with dry residue | Use `Water Leakage` if moisture path is primary; otherwise use `Surface Stain` |
| Patch covering crack | Label visible patch as `Repair Patch`; label crack only if visible outside patch |
| Delamination plus surface stain | Use separate annotations if visually separable |

### 3.5 Occlusion Metadata

Use:

```json
"occlusion": "none | partial | heavy"
```

| Value | Meaning |
|---|---|
| `none` | Defect fully visible |
| `partial` | Defect partly blocked by frame, object, glare, reflection, or shadow |
| `heavy` | Defect identifiable but visibility is poor |

### 3.6 Visibility Confidence

Use:

```json
"visibility_confidence": "high | medium | low"
```

| Value | Meaning |
|---|---|
| `high` | Clear defect and class |
| `medium` | Visible defect with some ambiguity |
| `low` | Likely defect requiring expert review |

Low-confidence annotations should not be included in the locked benchmark test set.

### 3.7 Severity Metadata

Use:

```json
"severity": "low | medium | high"
```

Severity should consider:

- Physical extent
- Safety consequence
- Structural implication
- Maintenance urgency
- Progression risk

Severity must not be assigned from pixel size alone because camera distance and zoom vary.

### 3.8 Negative Images

The dataset must include defect-free facade images.

Hard negatives should include:

- Reflections
- Shadows
- Mullions
- Sealant lines
- Stickers and signage
- Rain droplets
- Cleaning marks
- Interior objects visible through glass
- Dirt-like patterns that are not target defects

## 4. Dataset Folder Structure

Dataset assets should live inside the existing `docs/`, `models/`, `backend/`, `frontend/`, and `database/` project organization while raw and processed dataset files remain clearly separated.

Recommended dataset structure:

```text
docs/
├── DATASET_ARCHITECTURE.md
├── annotation_guide.md
├── dataset_card.md
├── edge_cases.md
└── reviewer_protocol.md

database/
├── metadata/
│   ├── buildings.csv
│   ├── inspections.csv
│   ├── images.csv
│   ├── panels.csv
│   └── annotations_review.csv
└── schemas/
    ├── annotation_schema.json
    ├── class_map.yaml
    └── split_config.yaml

models/
├── datasets/
│   ├── raw/
│   │   ├── drone/
│   │   ├── ground/
│   │   ├── telephoto/
│   │   └── archive/
│   ├── processed/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── masks/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── annotations/
│   │   ├── coco/
│   │   │   ├── instances_train.json
│   │   │   ├── instances_val.json
│   │   │   └── instances_test.json
│   │   ├── yolo/
│   │   │   ├── train/
│   │   │   ├── val/
│   │   │   └── test/
│   │   └── segmentation/
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── splits/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── test.txt
│   │   └── split_manifest.json
│   ├── review/
│   │   ├── pending/
│   │   ├── corrected/
│   │   ├── rejected/
│   │   └── adjudicated/
│   └── active_learning_pool/
│       ├── unlabeled/
│       ├── weak_labels/
│       ├── model_disagreements/
│       └── human_review_candidates/
```

## 5. Naming Conventions

### 5.1 Image File Names

Format:

```text
SV_<building_id>_<inspection_id>_<camera_type>_<elevation>_<timestamp>_<frame_id>.jpg
```

Example:

```text
SV_BLDG023_INSP004_DRONE_NORTH_20260618T143022_F000128.jpg
```

Fields:

| Field | Example | Description |
|---|---|---|
| `SV` | `SV` | SkyVision prefix |
| `building_id` | `BLDG023` | Building identifier |
| `inspection_id` | `INSP004` | Inspection session identifier |
| `camera_type` | `DRONE` | Capture source |
| `elevation` | `NORTH` | Facade side or zone |
| `timestamp` | `20260618T143022` | Local capture timestamp |
| `frame_id` | `F000128` | Sequential frame number |

### 5.2 Class Codes

| Class | Code |
|---|---|
| Broken Glass | `BRG` |
| Crack | `CRK` |
| Fracture | `FRC` |
| Water Leakage | `WTL` |
| Surface Stain | `STS` |
| Delamination | `DLM` |
| Repair Patch | `RPT` |

### 5.3 Annotation IDs

Format:

```text
ANN_<image_id>_<class_code>_<instance_number>
```

Example:

```text
ANN_SV_BLDG023_INSP004_DRONE_NORTH_20260618T143022_F000128_CRK_001
```

### 5.4 Panel IDs

Format:

```text
PNL_<building_id>_<elevation>_<floor>_<bay>_<panel_number>
```

Example:

```text
PNL_BLDG023_NORTH_FL12_BAY08_P03
```

### 5.5 Defect Tracking IDs

Format:

```text
DEF_<building_id>_<defect_number>
```

Example:

```text
DEF_BLDG023_000742
```

### 5.6 Release Names

Format:

```text
skyvision-dl-dataset-v<major>.<minor>.<patch>
```

Example:

```text
skyvision-dl-dataset-v1.0.0
```

## 6. Dataset Versioning Strategy

### 6.1 Version Format

Use semantic dataset versioning:

```text
MAJOR.MINOR.PATCH
```

Example:

```text
v1.2.0
```

Version meanings:

| Version | Meaning |
|---|---|
| MAJOR | Class taxonomy, annotation schema, or locked split changes |
| MINOR | New images, new annotations, expanded metadata, or new benchmark subsets |
| PATCH | Label corrections, metadata fixes, checksum repairs, or documentation updates |

### 6.2 Immutable Releases

Official dataset releases must be immutable. Corrections should produce a new patch version rather than modifying a released artifact.

Each release should include:

- Dataset version
- Annotation schema version
- Class map version
- Split manifest
- Data checksum manifest
- Dataset statistics
- Changelog
- Known limitations

### 6.3 Release Layout

Recommended release package:

```text
releases/
└── v1.0.0/
    ├── manifest.json
    ├── checksums.sha256
    ├── class_map.yaml
    ├── annotation_schema.json
    ├── split_manifest.json
    ├── dataset_statistics.json
    └── changelog.md
```

### 6.4 Data Lineage

Every processed image should retain linkage to:

- Raw source file or source asset ID
- Capture device
- Capture timestamp
- Inspection session
- Preprocessing pipeline version
- Annotation batch ID
- Reviewer ID
- Release version

### 6.5 Annotation Lifecycle

Annotation status values:

```text
draft
reviewed
corrected
adjudicated
approved
deprecated
```

Recommended lifecycle:

```text
Initial Label -> Reviewer Check -> Correction -> Adjudication -> Approved Release
```

## 7. Train/Validation/Test Split Strategy

### 7.1 Default Split Ratio

Recommended split:

```text
Train:      70%
Validation: 15%
Test:       15%
```

For early-stage datasets with limited samples:

```text
Train:      80%
Validation: 10%
Test:       10%
```

### 7.2 Split Unit

Splitting must be performed by grouped physical context, not random image assignment.

Preferred hierarchy:

1. Building-level split
2. Inspection-session split
3. Facade-elevation split
4. Panel-level split
5. Image-level split only when no higher-level metadata exists

### 7.3 Leakage Prevention

The same physical defect must not appear across train, validation, and test.

Group by:

- `building_id`
- `inspection_id`
- `facade_elevation`
- `panel_id`
- `defect_tracking_id`
- video sequence
- timestamp neighborhood

Near-duplicate frames from drone video should remain in the same split.

### 7.4 Stratification Requirements

Splits should preserve distribution across:

- Defect class
- Severity
- Camera type
- Lighting condition
- Weather condition
- Reflection level
- Distance range
- Facade type
- Building type
- Negative image ratio

Validation and test sets must contain every class unless a class has insufficient verified examples. Rare classes should be deliberately allocated to ensure meaningful evaluation.

### 7.5 Test Set Design

The test set should be locked, expert-reviewed, and excluded from model selection.

Recommended test subsets:

```text
test/
├── standard/
├── low_light/
├── high_reflection/
├── long_range/
├── wet_weather/
├── rare_defects/
└── hard_negatives/
```

The test set must not be used for:

- Hyperparameter tuning
- Threshold selection
- Active learning selection
- Annotation guideline iteration
- Repeated informal model debugging

### 7.6 Validation Set Design

The validation set should represent deployment conditions and support:

- Hyperparameter tuning
- Model selection
- Confidence threshold selection
- Class imbalance diagnosis
- Annotation guideline refinement
- Error analysis

Validation should include enough rare-class examples for reliable model development.

### 7.7 Active Learning Holdout

Maintain a separate active learning pool outside train, validation, and test:

```text
active_learning_pool/
├── unlabeled/
├── weak_labels/
├── model_disagreements/
└── human_review_candidates/
```

Prioritize future labeling for:

- Model false positives
- Model false negatives
- Low-confidence detections
- Rare classes
- High-reflection scenes
- Low-light scenes
- New facade systems

