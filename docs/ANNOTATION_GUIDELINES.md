# SkyVision-DL Annotation Guidelines

## Tool

Recommended annotation tools:

- CVAT
- Label Studio

Primary annotation mode:

- Bounding boxes for all defect instances
- Instance masks where precise defect boundaries are required
- Optional polylines for very thin cracks when supported by the annotation tool

## 1. Bounding Box Rules

### 1.1 General Box Rule

Draw one tight bounding box around each visible defect instance.

The box must:

- Enclose the complete visible defect.
- Stay inside the image boundary.
- Avoid excessive background.
- Avoid including unaffected glass panel area.
- Use the most specific defect class.

### 1.2 Correct Box Placement

For each class:

| Class | Bounding Box Rule |
|---|---|
| Broken Glass | Box the broken, missing, shattered, or collapsed visible region |
| Crack | Box the full visible crack path, including branches if sparse |
| Fracture | Box the full crack network or impact pattern |
| Water Leakage | Box the visible wet area, seepage trail, drip path, or moisture stain |
| Surface Stain | Box the visible stained or contaminated region |
| Delamination | Box the cloudy, bubbled, whitened, or separated internal layer region |
| Repair Patch | Box the visible repair material, tape, sealant, resin, or film patch |

### 1.3 Box Tightness Standard

Acceptable:

- Box margin is small and follows the defect extent.
- Long thin cracks are boxed as narrow regions.
- Irregular stains are boxed around the full affected area.

Not acceptable:

- Box covers the entire glass panel when only a small defect is visible.
- Box includes large reflection areas unrelated to the defect.
- Box merges two clearly separate defects into one instance.
- Box includes facade mullions, frames, or gaskets unless they are part of the defect area.

## 2. Occlusion Handling

### 2.1 Occlusion Labels

Each annotation should include an occlusion attribute:

```text
none
partial
heavy
```

| Value | Definition |
|---|---|
| `none` | Defect is fully visible |
| `partial` | Defect is partly blocked but class and extent are still understandable |
| `heavy` | Defect is mostly blocked but still identifiable |

### 2.2 Annotation Rule for Occlusion

Annotate only the visible part of the defect.

Do not extend the bounding box over hidden or inferred regions unless the visible defect clearly continues behind a thin obstruction and the tool protocol explicitly allows estimated boxes.

### 2.3 Examples

Correct:

- A crack partly hidden behind a mullion is boxed only around visible crack segments.
- A water trail partly covered by shadow is boxed over the visible wet trail.
- A repair patch partly cropped by the image boundary is boxed up to the image edge.

Incorrect:

- Extending a crack box behind a solid frame where no crack is visible.
- Labeling a defect that is completely hidden.
- Guessing the full size of a delamination region behind glare.

## 3. Reflection Handling

### 3.1 General Reflection Rule

Do not annotate reflections as defects.

Facade glass often reflects:

- Sky
- Clouds
- Trees
- Nearby buildings
- Cables
- Railings
- Interior lights
- People or equipment

These should not be labeled unless a real physical defect is visible.

### 3.2 Reflection vs Crack

Annotate as `Crack` only when the line:

- Has sharp local discontinuity on the glass.
- Remains fixed relative to the panel surface.
- Does not follow reflected object geometry.
- Shows physical branching, edge origin, or stress pattern.

Do not annotate when the line is likely:

- Reflected cable
- Reflected tree branch
- Reflected railing
- Interior object edge
- Mullion or gasket line

### 3.3 Reflection vs Water Leakage

Annotate as `Water Leakage` only when:

- The mark follows gravity-like flow.
- It starts near a joint, seal, edge, or opening.
- It has wetness, residue, or drip characteristics.

Do not annotate:

- Cloud reflections resembling stains.
- Bright glare streaks.
- Reflected building edges.

### 3.4 Edge Cases

If a defect is visible under reflection:

- Annotate the visible defect.
- Set visibility confidence to `medium` or `low`.
- Add reviewer comment if available.

If reflection prevents confident classification:

- Do not label for training.
- Send to review queue.

## 4. Multiple Defect Handling

### 4.1 Separate Instances

Annotate separate defects as separate instances when they are spatially distinct.

Example:

- One panel has a crack at the top and a water stain at the bottom.
- Create one `Crack` annotation and one `Water Leakage` or `Surface Stain` annotation.

### 4.2 Overlapping Defects

When defects overlap, annotate each visible defect separately if both are visually distinguishable.

Examples:

- Tape covers part of a crack: annotate visible tape as `Repair Patch`; annotate crack only where visible outside the tape.
- Water leakage passes over surface staining: annotate leakage if the moisture path is clear; annotate stain separately only if it is visually distinct.
- Broken glass has surrounding fracture lines: annotate missing or shattered area as `Broken Glass`; annotate surrounding crack network as `Fracture` if separable.

### 4.3 Class Priority Rules

Use these rules when one visual pattern could fit multiple classes:

| Visual Situation | Preferred Label |
|---|---|
| Single isolated line | `Crack` |
| Dense network of crack lines | `Fracture` |
| Missing or shattered glass | `Broken Glass` |
| Moisture trail from joint | `Water Leakage` |
| Dry dirt, grime, or residue | `Surface Stain` |
| Internal cloudy/bubbled layer failure | `Delamination` |
| Applied tape, sealant, resin, or cover | `Repair Patch` |

## 5. Partial Defects

### 5.1 Image Boundary Cropping

If a defect is cut off by the image boundary:

- Annotate the visible portion only.
- Set occlusion to `partial` if the defect continues outside the frame.
- Do not estimate the hidden extent outside the image.

### 5.2 Panel Boundary Cropping

If a defect continues across adjacent panels:

- Create separate annotations per visible defect instance when panel boundaries clearly divide the damage.
- Use the same `defect_tracking_id` if the dataset supports temporal or cross-panel tracking.

### 5.3 Partial Visibility from Blur or Low Resolution

Annotate only if:

- The defect class is visually identifiable.
- The bounding box can be placed consistently.
- The defect is not merely suspected.

If uncertain:

- Set visibility confidence to `low`.
- Send to reviewer.
- Exclude from benchmark test data.

## 6. Small Crack Annotations

### 6.1 Definition

Small cracks include hairline or short cracks that may be thin, low-contrast, or only a few pixels wide.

### 6.2 Annotation Rule

For small cracks:

- Draw the tightest possible bounding box around the visible line.
- Include the full visible length.
- Keep box height narrow but usable.
- Do not expand the box to cover the whole panel.
- Use `Crack`, not `Fracture`, unless the pattern has multiple intersecting branches.

### 6.3 Minimum Size

Recommended minimum:

```text
8 x 8 px
```

Cracks smaller than this may be annotated only if:

- The image is high-resolution.
- The crack is unambiguous.
- Multiple reviewers can verify it.

### 6.4 Optional Polyline

When using CVAT or Label Studio configurations that support polylines:

- Add a polyline along the crack centerline.
- Keep the bounding box as the required detection annotation.
- Use the polyline as supplemental geometry for thin-structure learning or QA.

### 6.5 Edge Cases

Do annotate:

- A short edge-originating hairline crack with clear physical continuity.
- A thin crack crossing a small part of a pane.
- A sparse two-branch crack as one `Crack` if the branches are minor.

Do not annotate:

- Reflected tree branch.
- Reflected cable.
- Scratches without evidence of glass failure.
- JPEG compression artifacts.

## 7. Water Leakage Annotations

### 7.1 Definition

Water Leakage is visible moisture ingress, seepage, wetness, runoff, dripping, or moisture-related marking associated with facade glass, joints, seals, or panel boundaries.

### 7.2 Annotation Rule

Annotate the visible leakage region, not the entire panel.

Include:

- Wet trail
- Drip mark
- Seepage line
- Water accumulation
- Moisture stain connected to a joint or edge
- Mineral deposit following a repeated leakage path

Exclude:

- Uniform rain droplets on the whole glass surface
- Cloud or sky reflection
- Cleaning streaks with no moisture source
- Dry general dirt without leakage pattern

### 7.3 Leakage vs Surface Stain

Use `Water Leakage` when:

- The pattern follows gravity.
- The mark originates near a joint, seal, edge, or opening.
- The path suggests active or repeated moisture movement.

Use `Surface Stain` when:

- The mark is dry contamination.
- There is no visible moisture path.
- The discoloration is broad, static, and not associated with a joint or seal.

### 7.4 Examples

Correct `Water Leakage`:

- A vertical wet trail starting from a window gasket.
- Drip marks below a facade joint.
- Mineral residue following a repeated runoff path.

Correct `Surface Stain`:

- Broad dusty haze on glass.
- Irregular dirt patch with no flow path.
- Cleaning residue spread across a pane.

Incorrect:

- Labeling every raindrop as leakage.
- Boxing an entire wet facade when only one trail is visible.
- Labeling cloud reflection as water damage.

## 8. Examples and Edge Cases

### 8.1 Broken Glass with Fracture Around It

Scenario:

A panel has a missing corner and several cracks radiating from the missing region.

Annotation:

- Box missing or shattered corner as `Broken Glass`.
- Box surrounding crack network as `Fracture` only if the network is visible and distinct.

### 8.2 Crack Under Strong Reflection

Scenario:

A thin diagonal line appears across reflective glass with tree reflections nearby.

Annotation:

- Label as `Crack` only if the line has physical crack characteristics.
- If uncertain, set confidence to `low` or send to review.
- Do not label if it follows the reflected tree branch structure.

### 8.3 Repair Tape Over Crack

Scenario:

Clear tape covers a cracked glass panel, and the crack is visible outside the tape.

Annotation:

- Box tape as `Repair Patch`.
- Box visible crack segments as `Crack`.
- Do not annotate hidden crack portions under opaque tape.

### 8.4 Water Trail Over Dirt

Scenario:

A vertical dark path runs through an already dirty pane.

Annotation:

- Use `Water Leakage` if the path starts near a joint and follows a drip pattern.
- Add `Surface Stain` separately only if a distinct dry stain region exists outside the leakage path.

### 8.5 Delamination vs Surface Stain

Scenario:

A milky cloudy region appears inside laminated glass near the edge.

Annotation:

- Use `Delamination` if the clouding appears internal or layer-like.
- Use `Surface Stain` if the mark is clearly exterior dirt or residue.

### 8.6 Partial Crack at Image Edge

Scenario:

A crack enters from the left image boundary and continues out of frame.

Annotation:

- Box the visible crack segment only.
- Set occlusion to `partial`.
- Do not estimate the off-image portion.

### 8.7 Multiple Small Cracks

Scenario:

Several small cracks are separated across the same panel.

Annotation:

- Use separate `Crack` boxes if the cracks are spatially separate.
- Use one `Fracture` box if they form a connected network.

### 8.8 Glare Hiding a Defect

Scenario:

Bright glare covers part of a suspected stain or crack.

Annotation:

- Annotate only clearly visible parts.
- Set occlusion to `partial` or `heavy`.
- Set visibility confidence to `medium` or `low`.
- Route to review if classification is uncertain.

## 9. Reviewer Checklist

Before approval, verify:

- Correct class label is used.
- Bounding box is tight.
- Separate defects are not merged unnecessarily.
- Reflection artifacts are not labeled.
- Partial defects are not overestimated.
- Small cracks are real and not reflected objects.
- Water leakage is not confused with rain droplets or surface dirt.
- Occlusion and visibility confidence attributes are set correctly.

