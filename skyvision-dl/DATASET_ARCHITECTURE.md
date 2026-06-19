# SkyVision-DL Dataset Architecture

## Research-Grade Dataset Design Document

Project: AI-powered facade glass defect detection system

## 1. Dataset Specification

### 1.1 Objective

SkyVision-DL is a computer vision dataset for detecting, localizing, segmenting, and analyzing facade glass defects from drone, ground, telephoto, and inspection-archive imagery.

The dataset is designed to support:

- Object detection
- Instance segmentation
- Semantic segmentation
- Multi-label image classification
- Defect severity estimation
- Temporal defect tracking
- Inspection prioritization for facade maintenance

Primary benchmark task:

> Detect and localize visible defects on architectural facade glass panels under real inspection conditions.

### 1.2 Target Classes

The dataset uses seven primary defect classes:

| ID | Class | Code |
|---:|---|---|
| 1 | Broken Glass | `BRG` |
| 2 | Crack | `CRK` |
| 3 | Fracture | `FRC` |
| 4 | Water Leakage | `WTL` |
| 5 | Surface Stain | `STS` |
| 6 | Delamination | `DLM` |
| 7 | Repair Patch | `RPT` |

### 1.3 Data Sources

Recommended image sources:

- UAV or drone facade inspection imagery
- Ground-level inspection cameras
- Telephoto facade survey images
- Mobile-device maintenance images
- Fixed monitoring camera frames
- Historical inspection archives

### 1.4 Capture Diversity Requirements

The dataset should include variation across:

- Building height: low-rise, mid-rise, high-rise
- Facade type: curtain wall, window wall, storefront, skylight, atrium glass
- Viewpoint: frontal, oblique, upward-looking, downward-looking
- Distance: close-range, medium-range, long-range
- Weather: dry, wet, post-rain, humid
- Lighting: direct sunlight, overcast, dusk, night, shadowed facade
- Reflection: low reflection, high reflection, sky reflection, urban reflection
- Glass type: clear, tinted, coated, laminated, insulated glass units
- Defect scale: tiny, small, medium, large

### 1.5 Core Data Unit

The dataset is organized around four related entities:

| Entity | Description |
|---|---|
| Building | Physical building or site under inspection |
| Inspection | A specific capture event for a building |
| Image | Individual frame or photograph |
| Defect instance | Localized visible occurrence of a target defect |

Each annotation should be traceable to:

- `building_id`
- `inspection_id`
- `image_id`
- `panel_id`, when available
- `defect_tracking_id`, when the same physical defect is tracked over time
- annotator and reviewer metadata

### 1.6 Primary Annotation Format

Primary format:

- COCO-style JSON with detection and segmentation extensions

Secondary export formats:

- YOLO detection labels
- Pascal VOC XML
- PNG semantic masks
- PNG instance masks
- CSV metadata tables

The canonical source of truth should remain the COCO-style JSON plus metadata tables.

## 2. Class Definitions

### 2.1 Broken Glass

Definition:

Broken Glass refers to visibly shattered, missing, collapsed, or severely damaged glass where the panel has lost physical continuity.

Visual indicators:

- Missing glass region
- Shattered or splintered glass
- Large opening through the panel
- Detached fragments
- Sharp irregular break boundary
- Severe local collapse

Include:

- Fully shattered glass panel
- Partially missing glass
- Broken corner with visible material loss
- Hole or exposed void caused by glass failure

Exclude:

- Thin line damage without missing material; use `Crack`
- Dense crack network with intact glass; use `Fracture`
- Reflection artifacts resembling holes
- Temporary repair material; use `Repair Patch`

Severity:

| Severity | Rule |
|---|---|
| Low | Small broken corner or chip-like break |
| Medium | Localized broken area, panel mostly intact |
| High | Large shattered or missing region, likely safety-critical |

### 2.2 Crack

Definition:

Crack refers to a narrow linear discontinuity on or within the glass where the panel remains mostly intact.

Visual indicators:

- Thin bright or dark line
- Linear, curved, or branching path
- Sharp local contrast
- Edge-originating line
- No major missing material

Include:

- Hairline crack
- Single long crack
- Edge crack
- Small branching crack when still sparse and line-like

Exclude:

- Dense impact pattern; use `Fracture`
- Normal mullion, gasket, or sealant line
- Reflections of cables, trees, railings, or facade edges
- Cosmetic scratch when not a structural glass defect

Severity:

| Severity | Rule |
|---|---|
| Low | Short, localized hairline crack |
| Medium | Long crack across part of panel |
| High | Crack spanning most of panel or reaching multiple edges |

### 2.3 Fracture

Definition:

Fracture refers to a complex damage pattern with multiple cracks, radial lines, branching failures, or stress networks.

Visual indicators:

- Spiderweb pattern
- Radial cracks from an impact center
- Multiple intersecting cracks
- Dense crack cluster
- Thermal or stress fracture network

Include:

- Impact-centered fracture
- Multi-branch crack system
- Structural crack cluster
- Crack network covering a meaningful panel region

Exclude:

- Single isolated crack; use `Crack`
- Missing or shattered glass region; use `Broken Glass`
- Reflection patterns that only resemble crack networks

Severity:

| Severity | Rule |
|---|---|
| Low | Small localized fracture cluster |
| Medium | Fracture covering moderate panel area |
| High | Extensive fracture network across the panel |

### 2.4 Water Leakage

Definition:

Water Leakage refers to visible water ingress, seepage, wetness, runoff paths, drip marks, or moisture staining associated with glazing edges, seals, or panel joints.

Visual indicators:

- Vertical water trail
- Wet patch near seal or joint
- Drip line
- Moisture accumulation at panel edge
- Mineral deposit following a leakage path
- Repeated runoff pattern

Include:

- Active seepage
- Water trail from glazing joint
- Wet stain below a facade opening
- Moisture-related residue when the leakage path is clear

Exclude:

- Uniform rain droplets across glass
- Reflections of clouds or sky
- General dirt with no leakage pattern
- Dry non-moisture contamination; use `Surface Stain`

Severity:

| Severity | Rule |
|---|---|
| Low | Minor localized water mark |
| Medium | Clear leakage trail or repeated moisture path |
| High | Heavy leakage, multiple trails, or large wet region |

### 2.5 Surface Stain

Definition:

Surface Stain refers to visible exterior contamination, discoloration, grime, mineral residue, pollution deposit, biological growth, or cleaning residue on glass.

Visual indicators:

- Cloudy dirty region
- Mineral spotting
- Brown, black, white, or green discoloration
- Smudge or residue
- Pollution streak
- Irregular external deposit

Include:

- Dirt buildup
- Hard-water stain
- Rust-colored runoff residue
- Cleaning residue
- Biological growth on glass surface

Exclude:

- Active or path-like moisture defect; use `Water Leakage`
- Internal layer haze; use `Delamination`
- Repair material; use `Repair Patch`
- Reflection or shadow without physical contamination

Severity:

| Severity | Rule |
|---|---|
| Low | Light visible stain |
| Medium | Clearly visible localized staining |
| High | Large or opaque stain affecting inspection quality |

### 2.6 Delamination

Definition:

Delamination refers to internal layer separation, bubbling, edge whitening, haze, coating separation, or film failure in laminated, coated, or insulated glass.

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
- Layer failure in insulated or laminated glazing

Exclude:

- External dirt; use `Surface Stain`
- Weather condensation without material failure
- Reflection or glare
- Applied repair material unless visibly delaminating

Severity:

| Severity | Rule |
|---|---|
| Low | Small edge haze or bubble |
| Medium | Localized delaminated region |
| High | Large cloudy or separated layer region |

### 2.7 Repair Patch

Definition:

Repair Patch refers to visible material applied to glass or adjacent facade areas as temporary or permanent repair, sealing, reinforcement, protection, or replacement marking.

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
- Stains or dirt

Severity:

| Severity | Rule |
|---|---|
| Low | Small localized repair |
| Medium | Visible repair covering a defect area |
| High | Large temporary patch indicating serious prior damage |

## 3. Annotation Standards

### 3.1 General Principles

Annotate visible evidence, not inferred hidden damage.

Annotators must:

- Mark only visible defects.
- Avoid labeling reflections as defects.
- Use the most specific applicable class.
- Annotate each distinct defect as a separate instance.
- Include the full visible extent of the defect.
- Flag uncertainty instead of forcing a confident label.

### 3.2 Bounding Boxes

Every defect instance requires a bounding box.

Rules:

- Box tightly around the visible defect.
- Include all visible branches for cracks and fractures.
- Do not include an entire panel unless the defect covers the entire panel.
- For leakage and stains, box only the visible affected region.
- For repair patches, box the visible repair material.

Minimum recommended box size:

- `8 x 8 px` for reliable training labels
- Smaller defects may be retained only when visually unambiguous

### 3.3 Instance Masks

Instance masks are strongly recommended for all defects.

Rules:

- Follow visible defect boundaries.
- Use thin polygons or brush masks for cracks.
- Use area masks for stains, delamination, leakage, broken regions, and patches.
- For very thin cracks, store an optional polyline skeleton in addition to the mask.

### 3.4 Semantic Masks

Semantic masks may be generated from instance masks for segmentation models.

Rules:

- Pixel value `0` is background.
- Pixel values `1-7` map to the class IDs.
- Overlapping instances should preserve instance annotations in the canonical COCO file.
- Semantic masks are derived artifacts, not the primary source of truth.

### 3.5 Multi-Class Conflict Rules

| Situation | Rule |
|---|---|
| Single line damage | Label as `Crack` |
| Dense crack network | Label as `Fracture` |
| Broken area with surrounding cracks | Label broken region as `Broken Glass`; label surrounding network as `Fracture` if distinct |
| Water trail with dry residue | Use `Water Leakage` if path suggests moisture ingress; otherwise use `Surface Stain` |
| Patch covering crack | Label visible patch as `Repair Patch`; label crack only if visible outside patch |
| Delamination plus exterior stain | Use separate annotations if both are visually distinct |

### 3.6 Occlusion Metadata

Each annotation should include:

```json
"occlusion": "none | partial | heavy"
```

Definitions:

| Value | Meaning |
|---|---|
| `none` | Defect is fully visible |
| `partial` | Defect is partly blocked by frame, object, glare, reflection, or shadow |
| `heavy` | Defect is identifiable but visibility is poor |

### 3.7 Visibility Confidence

Each annotation should include:

```json
"visibility_confidence": "high | medium | low"
```

Definitions:

| Value | Meaning |
|---|---|
| `high` | Clear defect and class |
| `medium` | Visible defect with some ambiguity |
| `low` | Likely defect requiring expert review |

Low-confidence annotations should be excluded from locked benchmark test sets.

### 3.8 Severity Metadata

Each annotation should include:

```json
"severity": "low | medium | high"
```

Severity should consider:

- Physical extent
- Safety consequence
- Structural implication
- Maintenance urgency
- Defect progression risk

Severity must not be assigned only from pixel size, because camera distance and zoom vary.

### 3.9 Negative Images

The dataset must include defect-free facade images.

Hard negative content should include:

- Reflections
- Shadows
- Mullions
- Sealant lines
- Stickers and signage
- Rain droplets
- Cleaning marks
- Dirt-like non-target patterns
- Interior objects seen through glass

## 4. Dataset Folder Structure

Recommended repository layout:

```text
skyvision-dl/
├── README.md
├── DATASET_ARCHITECTURE.md
├── DATASET_CARD.md
├── LICENSE.md
├── CHANGELOG.md
├── class_map.yaml
├── annotation_schema.json
├── split_config.yaml
├── quality_report.md
│
├── metadata/
│   ├── buildings.csv
│   ├── inspections.csv
│   ├── images.csv
│   ├── panels.csv
│   └── annotations_review.csv
│
├── raw/
│   ├── drone/
│   ├── ground/
│   ├── telephoto/
│   └── archive/
│
├── processed/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── masks/
│       ├── train/
│       ├── val/
│       └── test/
│
├── annotations/
│   ├── coco/
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   ├── yolo/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── segmentation/
│       ├── train/
│       ├── val/
│       └── test/
│
├── splits/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── split_manifest.json
│
├── review/
│   ├── pending/
│   ├── rejected/
│   ├── corrected/
│   └── adjudicated/
│
├── benchmarks/
│   ├── detection/
│   ├── segmentation/
│   └── severity/
│
├── active_learning_pool/
│   ├── unlabeled/
│   ├── weak_labels/
│   ├── model_disagreements/
│   └── human_review_candidates/
│
└── docs/
    ├── annotation_guide.md
    ├── labeling_examples/
    ├── edge_cases.md
    └── reviewer_protocol.md
```

## 5. Naming Conventions

### 5.1 Image Naming

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

### 5.2 Annotation Naming

Format:

```text
ANN_<image_id>_<class_code>_<instance_number>
```

Example:

```text
ANN_SV_BLDG023_INSP004_DRONE_NORTH_20260618T143022_F000128_CRK_001
```

### 5.3 Panel Naming

Format:

```text
PNL_<building_id>_<elevation>_<floor>_<bay>_<panel_number>
```

Example:

```text
PNL_BLDG023_NORTH_FL12_BAY08_P03
```

### 5.4 Defect Tracking ID

Format:

```text
DEF_<building_id>_<defect_number>
```

Example:

```text
DEF_BLDG023_000742
```

### 5.5 File Extension Rules

Recommended extensions:

- Images: `.jpg` for lossy camera images, `.png` for lossless processed masks
- COCO annotations: `.json`
- Metadata tables: `.csv`
- Configuration files: `.yaml`
- Documentation: `.md`

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
| MINOR | New images, new labels, expanded metadata, new benchmark subset |
| PATCH | Label corrections, metadata fixes, checksum repairs, documentation updates |

### 6.2 Immutable Releases

Official releases must be immutable. If a defect is found in a release, create a new patch version rather than modifying the released artifact.

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

- Raw source path or source asset ID
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

### 6.6 Dataset Card

Each release should include a dataset card documenting:

- Intended use
- Non-recommended use
- Collection method
- Building and geographic diversity
- Camera systems
- Class taxonomy
- Annotation process
- Quality controls
- Known biases
- Privacy considerations
- Benchmark metrics
- Version history

## 7. Train, Validation, and Test Split Strategy

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
5. Image-level split only when no higher metadata exists

### 7.3 Leakage Prevention

The same physical defect must not appear across train, validation, and test.

Group by:

- `building_id`
- `inspection_id`
- `elevation`
- `panel_id`
- `defect_tracking_id`
- video frame sequence
- timestamp neighborhood

Near-duplicate frames from drone video should remain in the same split.

### 7.4 Stratification

Splits should preserve distribution across:

- Defect class
- Severity
- Camera type
- Lighting condition
- Weather
- Reflection level
- Distance range
- Facade type
- Building type
- Negative image ratio

Validation and test sets must contain every class unless a class has insufficient verified examples. Rare classes should be deliberately allocated to ensure meaningful evaluation.

### 7.5 Test Set Design

The test set should be locked and expert-reviewed.

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

The test set must not be used for model selection, threshold tuning, active learning selection, or annotation guideline iteration.

### 7.6 Validation Set Design

The validation set is used for:

- Hyperparameter tuning
- Model selection
- Confidence threshold selection
- Class imbalance diagnosis
- Annotation guideline refinement
- Error analysis

Validation should resemble deployment conditions while retaining enough rare-class examples for reliable model development.

### 7.7 Active Learning Pool

Maintain a separate pool for future labeling:

```text
active_learning_pool/
├── unlabeled/
├── weak_labels/
├── model_disagreements/
└── human_review_candidates/
```

Prioritize:

- Model false positives
- Model false negatives
- Low-confidence detections
- Rare classes
- High-reflection scenes
- Low-light scenes
- New facade systems

## 8. Recommended Annotation Schema

COCO-style annotation extension:

```json
{
  "annotation_id": "ANN_SV_BLDG023_INSP004_DRONE_NORTH_20260618T143022_F000128_CRK_001",
  "image_id": "SV_BLDG023_INSP004_DRONE_NORTH_20260618T143022_F000128",
  "category_id": 2,
  "category_name": "Crack",
  "bbox": [412, 281, 96, 18],
  "segmentation": [],
  "area": 1225,
  "iscrowd": 0,
  "severity": "medium",
  "occlusion": "none",
  "visibility_confidence": "high",
  "panel_id": "PNL_BLDG023_NORTH_FL12_BAY08_P03",
  "defect_tracking_id": "DEF_BLDG023_000742",
  "annotator_id": "A014",
  "review_status": "approved"
}
```

## 9. Quality Control Protocol

### 9.1 Automated Checks

Run automated checks for:

- Missing image files
- Missing annotation files
- Empty boxes
- Out-of-bounds boxes
- Invalid masks
- Class ID mismatch
- Duplicate image IDs
- Duplicate annotations
- Near-duplicate leakage across splits
- Missing metadata
- Tiny annotations below minimum threshold

### 9.2 Human Review

Recommended review coverage:

- 100% review for test set
- 100% review for rare classes
- 100% review for high-severity defects
- At least 20% review for standard training annotations
- Double annotation for ambiguous samples
- Expert adjudication for disputed labels

### 9.3 Release Metrics

Each release should report:

- Number of images
- Number of buildings
- Number of inspection sessions
- Number of panels
- Number of annotations
- Instances per class
- Images per class
- Negative image count
- Severity distribution
- Camera distribution
- Lighting distribution
- Weather distribution
- Reflection distribution
- Mean and median defect size
- Small-object percentage
- Annotation confidence distribution
- Inter-annotator agreement

## 10. Benchmark Evaluation Protocol

### 10.1 Detection Metrics

Report:

- mAP@0.50
- mAP@0.50:0.95
- Per-class AP
- AP for small, medium, and large defects
- Recall at fixed false positives per image

### 10.2 Segmentation Metrics

Report:

- Mask mAP
- Mean IoU
- Boundary F-score
- Per-class mask AP

For cracks and fractures, boundary-sensitive metrics are especially important because thin structures may have poor IoU despite useful localization.

### 10.3 Classification Metrics

Report:

- Macro F1
- Micro F1
- Per-class precision and recall
- Multi-label AUROC
- Confusion matrix

### 10.4 Deployment Metrics

Report:

- False positives per facade image
- Miss rate for high-severity defects
- Inference latency
- Performance under high reflection
- Performance under low light
- Performance at long range
- Performance in wet-weather scenes

## 11. Final Recommendation

SkyVision-DL should be built as a metadata-rich, multi-task facade inspection dataset with COCO-style detection and segmentation as the canonical label format. The dataset should preserve building, inspection, panel, and defect identity so that model evaluation reflects true deployment generalization rather than near-duplicate image memorization.

The highest-priority design decisions are:

- Use instance-level annotations as the source of truth.
- Split by building or inspection session, not by random image.
- Lock and expert-review the test set.
- Include hard negatives that resemble facade defects.
- Track severity, occlusion, visibility confidence, and data lineage from the first release.

