# Traffic Speed Monitoring — YOLO Version

## `monitor_traffic.py`

---

## 1. Idea

The goal is to **measure vehicle speeds on a motorway** from an overhead bridge camera, using a **deep-learning object detector** (YOLOv8) combined with a modern multi-object tracker (ByteTrack).

This version replaces the classical background-subtraction detection of the OpenCV script with **YOLOv8x inference**, providing more accurate vehicle detection, built-in vehicle classification, and robust tracking through occlusions.

Speed estimation follows the same two-method approach:

1. **Tripwire speed** — time between crossing two virtual "green lines".
2. **BEV rolling-window speed** — continuous displacement in BEV metre space.

---

## 2. Theory

### 2.1 Perspective Transform (Homography)

Identical to the OpenCV version. A 3×3 **homography matrix H** maps a road trapezoid in the 1280×720 image to a 28 m × 102 m rectangle in the real-world BEV plane. This corrects for the perspective distortion introduced by the elevated camera viewpoint.

| Point        | Pixel (u, v) | Real-world (m) |
| ------------ | ------------ | -------------- |
| Top-left     | (490, 295)   | (0, 0)         |
| Top-right    | (790, 295)   | (28, 0)        |
| Bottom-right | (1260, 700)  | (28, 102)      |
| Bottom-left  | (20, 700)    | (0, 102)       |

### 2.2 YOLOv8 Object Detection

**YOLO** (You Only Look Once) is a single-stage object detector that predicts bounding boxes and class probabilities in a single forward pass through a convolutional neural network.

The model used is **YOLOv8x** (extra-large), pre-trained on the **COCO dataset** (80 object classes). We filter detections to keep only vehicle classes:

| COCO ID | Class      |
| ------- | ---------- |
| 2       | car        |
| 3       | motorcycle |
| 5       | bus        |
| 7       | truck      |

This gives us:

- **Accurate bounding boxes** even for partially visible or overlapping vehicles.
- **Built-in classification** — no need for area-based heuristics (as in the OpenCV version).

### 2.3 ByteTrack Multi-Object Tracking

**ByteTrack** is an online multi-object tracker that associates every detection — including low-confidence ones — with existing tracks using the **Hungarian algorithm** on IoU (Intersection over Union) costs.

Key advantages over the simple centroid tracker in the OpenCV version:

- Handles occlusions by retaining "lost" tracks and recovering them via low-confidence detections.
- Maintains consistent track IDs across longer time periods.
- Works well with varying detection confidence.

The tracker is provided by the **Supervision** library (`sv.ByteTrack`).

### 2.4 Speed Estimation — Tripwire Method

Identical formula to the OpenCV version:

$$
\text{speed (km/h)} = \frac{d_{\text{corrected}}}{(f_2 - f_1) / \text{fps}} \times 3.6
$$

Where $d_{\text{corrected}}$ is the perspective-corrected distance between the two green lines (≈30.1 m), and $(f_2 - f_1)$ is the frame count between the two crossings.

The **bottom-centre anchor** of each YOLO bounding box is used as the crossing point (rather than the centroid used in the OpenCV version), because it better represents the vehicle's contact point with the road.

### 2.5 Speed Estimation — BEV Rolling Window

A sliding window of 1 second (= `fps` frames) stores the BEV Y-coordinate of each track's bottom-centre anchor. The displacement over the full window gives the speed:

$$
\text{speed} = \frac{|y_{\text{newest}} - y_{\text{oldest}}|}{n_{\text{frames}} / \text{fps}} \times 3.6
$$

This provides a real-time speed estimate for vehicles that haven't yet completed both tripwire crossings.

### 2.6 Tripwire Zone vs Full-Frame Tracking

A **critical design decision**: speed is computed for **all** vehicles across the full frame (for accurate BEV displacement), but bounding boxes are **only drawn** for vehicles inside the tripwire zone (between Line 1 and Line 2, ± 30 px margin).

This ensures:

- The BEV speed calculation has a long displacement baseline (more accurate).
- The video output shows only the region of interest.

### 2.7 Naive vs Corrected Distance

The script computes both distances:

- **Corrected**: The mid-point of each green line is projected through the homography. The Y-difference gives the true distance (≈30.1 m).
- **Naive**: Assumes a uniform pixel-to-metre scale derived from the average pixel width of both lines. This yields ≈4.8 m — a massive underestimate that demonstrates why perspective correction is essential.

---

## 3. Implementation

### 3.1 Architecture

The script is **fully functional** (no classes). Mutable state is stored in plain `dict` and `list` objects.

```
┌──────────────────────────────────┐
│  Configuration (constants)       │
│  YOLO class IDs, paths, params   │
├──────────────────────────────────┤
│  Perspective calibration         │
│  Tripwire distance computation   │
├──────────────────────────────────┤
│  Tripwire Crossing (dict)        │  make_tripwire_state() / update_tripwire()
│  Point Transform (function)      │  transform_points()
│  Drawing helpers                 │  draw_tripwires() / draw_hud()
├──────────────────────────────────┤
│  main()  — frame loop            │
│   1. YOLO detection              │
│   2. ByteTrack tracking          │
│   3. Perspective transform       │
│   4. Class name recording        │
│   5. Speed calculation (all)     │
│   6. Zone filtering (draw only)  │
│   7. Annotation & write          │
│   8. CSV export                  │
└──────────────────────────────────┘
```

### 3.2 Key Functions

| Function                                                    | Purpose                                                               |
| ----------------------------------------------------------- | --------------------------------------------------------------------- |
| `_transform_pt(pt)`                                         | Map one pixel point → BEV metres (used at module level)               |
| `transform_points(points)`                                  | Batch-transform an (N, 2) array of pixels → BEV metres                |
| `make_tripwire_state()`                                     | Create tripwire state dict (line positions, distances, fps)           |
| `update_tripwire(state, tracker_ids, anchor_ys, frame_idx)` | Detect Line 1/2 crossings per track; compute speed on Line 2 crossing |
| `get_tripwire_speed(state, tid)`                            | Query the corrected speed for a given track, or None                  |
| `draw_tripwires(frame)`                                     | Draw two green lines with labels                                      |
| `draw_hud(frame, ...)`                                      | Draw semi-transparent status overlay                                  |
| `main()`                                                    | Full YOLO pipeline: detect → track → speed → annotate → export        |

### 3.3 Differences from the OpenCV Version

| Aspect               | OpenCV version                                     | YOLO version                                   |
| -------------------- | -------------------------------------------------- | ---------------------------------------------- |
| **Detection**        | MOG2 background subtraction + contour extraction   | YOLOv8x neural network                         |
| **Tracking**         | Custom centroid tracker (greedy nearest-neighbour) | ByteTrack (Hungarian + IoU)                    |
| **Vehicle type**     | Bounding-box area threshold (car vs truck)         | YOLO class label (car, motorcycle, bus, truck) |
| **Anchor point**     | Bounding-box centroid                              | Bottom-centre of bounding box                  |
| **BEV speed method** | Median per-frame displacement                      | Start/end displacement over window             |
| **Annotation**       | Custom `cv2.rectangle` + `cv2.putText`             | Supervision `BoxAnnotator` + `LabelAnnotator`  |
| **Processing speed** | ~4 seconds for 250 frames                          | ~230 seconds (GPU-dependent)                   |

### 3.4 Outputs

| Output                     | Description                                                                                       |
| -------------------------- | ------------------------------------------------------------------------------------------------- |
| `traffic_speed_output.mp4` | Annotated video with YOLO bounding boxes, class labels, speeds, green lines, and HUD              |
| `vehicle_speeds_yolo.csv`  | One row per vehicle: YOLO class name, tripwire speed (corrected + naive), BEV speed (avg/min/max) |

### 3.5 Dependencies

| Package         | Role                                                                                           |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `ultralytics`   | YOLOv8 model loading, inference, and result parsing                                            |
| `supervision`   | ByteTrack tracker, box/label annotators, video I/O (`VideoSink`, `get_video_frames_generator`) |
| `opencv-python` | Perspective transform, image operations, drawing                                               |
| `numpy`         | Numerical array operations                                                                     |

---

## 4. Code Walkthrough

### Module-level setup (lines 1–66)

1. **Imports** — `os`, `csv`, `cv2`, `numpy`, `supervision`, `ultralytics`, `collections`.
2. **Configuration** — Paths, YOLO class IDs and names, green-line coordinates, zone margin.
3. **Perspective matrix** — Same calibration as the OpenCV version.
4. **Tripwire distance** — Green-line midpoints projected through the homography → corrected distance (≈30.1 m). Naive distance computed for comparison (≈4.8 m).

### Tripwire crossing (lines 68–128)

- `make_tripwire_state()` creates a dict with line positions, distances, FPS, and per-track state dicts.
- `update_tripwire()` iterates over `(tracker_id, anchor_y)` pairs. It detects crossings by checking if the previous Y and current Y straddle a line's Y-coordinate. Line 1 must be crossed first.
- `get_tripwire_speed()` returns the corrected km/h speed, or `None`.

> **Note:** Unlike the OpenCV version (which takes an `active` dict of centroids), this version takes separate `tracker_ids` and `anchor_ys` arrays — matching the Supervision detection format.

### Utility and drawing (lines 132–161)

- `transform_points()` — batch version of the perspective transform for N points at once.
- `draw_tripwires()` — two green `cv2.line()` calls with text labels.
- `draw_hud()` — semi-transparent overlay with vehicle count, frame progress, and tripwire distance.

### Main pipeline (lines 167–328)

1. **Load YOLO model** — `YOLO("yolov8x.pt")` downloads/loads the extra-large model.
2. **Open video** — Supervision's `VideoInfo` and `get_video_frames_generator`.
3. **Initialise** — ByteTrack, annotators, coordinate history, tripwire state.
4. **Frame loop** (lines 199–281):
   - **Step 1:** Run YOLO inference → filter to vehicle class IDs only.
   - **Step 2:** `byte_track.update_with_detections()` assigns persistent track IDs.
   - **Step 3:** Extract bottom-centre anchors → project to BEV via `transform_points()`.
   - **Step 4:** Map each track ID to its YOLO class name.
   - **Step 5:** Compute tripwire speed (crossing check) and BEV rolling-window speed. Choose the best available for the label.
   - **Step 6:** Create a zone mask (`LINE_1_Y ± margin` to `LINE_2_Y ± margin`) and filter detections/labels for drawing only.
   - **Step 7:** Annotate with Supervision's `BoxAnnotator` and `LabelAnnotator`, plus green lines and HUD.
5. **CSV export** (lines 283–306): one row per vehicle, merging both speed methods.
6. **Summary** (lines 308–320): average corrected/naive speeds and the percentage effect of perspective correction.

---

## 5. Performance

| Metric               | Value                                      |
| -------------------- | ------------------------------------------ |
| **Detection model**  | YOLOv8x (68.7M params)                     |
| **Processing time**  | ~230 s for 250 frames (1280×720) on CPU    |
| **GPU acceleration** | Recommended for real-time or longer videos |

The YOLO version is ~50× slower than the OpenCV version but produces more reliable detections and classifications.

---

## 6. Limitations & Possible Improvements

| Limitation                                                  | Improvement                                               |
| ----------------------------------------------------------- | --------------------------------------------------------- |
| Very slow on CPU (~4 min for 10 s of video)                 | Use GPU or a lighter model (yolov8n/s/m)                  |
| Tripwire speed only captures vehicles that cross both lines | Extend window or use BEV speed as the primary metric      |
| Single camera view                                          | Multi-camera fusion for wider coverage                    |
| Fixed green-line coordinates                                | Interactive calibration GUI                               |
| No traffic rule detection (e.g. speeding alerts)            | Add threshold-based alerts with configurable speed limits |
