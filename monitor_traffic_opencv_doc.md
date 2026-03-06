# Traffic Speed Monitoring — OpenCV Version

## `monitor_traffic_opencv.py`

---

## 1. Idea

The goal is to **measure vehicle speeds on a motorway** from an overhead bridge camera, using **only classical computer-vision** techniques (no deep learning).

The camera records at a fixed position and frame rate (25 fps). Two virtual "green lines" (tripwires) are drawn across the road at known positions. When a vehicle crosses both lines, the time between the crossings — combined with the real-world distance between the lines — yields its speed.

A secondary, continuous speed estimate is also computed by tracking each vehicle through a **Bird's-Eye-View (BEV)** coordinate system.

---

## 2. Theory

### 2.1 Perspective Transform (Homography)

A camera observing a flat road from an elevated viewpoint introduces **perspective distortion**: objects far away appear smaller and closer together. A pixel displacement near the horizon represents a much larger real-world distance than the same pixel displacement at the bottom of the frame.

To correct for this, we define a **homography** — a 3×3 matrix **H** that maps a trapezoid in image space to a rectangle in real-world metre space.

$$
\begin{bmatrix} X' \\ Y' \\ W' \end{bmatrix}
= H \cdot
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
\quad,\qquad
x_{\text{m}} = \frac{X'}{W'} \;,\;\; y_{\text{m}} = \frac{Y'}{W'}
$$

Where $(u, v)$ are pixel coordinates and $(x_m, y_m)$ are real-world metres.

**Calibration** was performed using visible lane-dash markings:

| Point        | Pixel (u, v) | Real-world (m) |
| ------------ | ------------ | -------------- |
| Top-left     | (490, 295)   | (0, 0)         |
| Top-right    | (790, 295)   | (28, 0)        |
| Bottom-right | (1260, 700)  | (28, 102)      |
| Bottom-left  | (20, 700)    | (0, 102)       |

This gives a road patch of **28 m wide × 102 m long**.

### 2.2 Speed Estimation — Tripwire Method

Two lines are drawn at pixel rows `y=400` (Line 1) and `y=560` (Line 2). Each represents 25 m of real-world road width.

The **real-world distance** between the lines is found by projecting their centres through the calibrated homography:

```
distance_m = |BEV_y(centre_2) − BEV_y(centre_1)|
```

When a vehicle's centroid crosses Line 1 at frame $f_1$ and Line 2 at frame $f_2$:

$$
\text{speed (km/h)} = \frac{\text{distance\_m}}{(f_2 - f_1) / \text{fps}} \times 3.6
$$

A **naive distance** is also computed for comparison:

$$
d_{\text{naive}} = \frac{(y_2 - y_1) \times 25\,\text{m}}{(\text{px\_width}_{L1} + \text{px\_width}_{L2}) / 2}
$$

This ignores perspective and significantly underestimates the true distance.

### 2.3 Speed Estimation — BEV Rolling Window

For vehicles that haven't completed both crossings (or as a real-time display), the script computes a **rolling-window BEV speed**:

1. Each frame, the centroid is projected to BEV → Y-coordinate stored.
2. Over the last `SPEED_WINDOW` frames, compute per-frame displacements.
3. Take the **median** displacement (robust to outliers).
4. Convert to km/h: `speed = median_disp_m × fps × 3.6`.

### 2.4 Vehicle Detection — Background Subtraction

Since there is no deep-learning model, vehicles are detected using **MOG2 background subtraction**:

1. **MOG2** learns a Gaussian mixture model of the static background.
2. Moving foreground pixels are extracted.
3. **Morphological operations** clean up the mask:
   - **Opening** (erode + dilate) removes small noise.
   - **Closing** (dilate + erode) fills internal gaps.
   - **Dilation** grows blobs so nearby detections merge.
4. **Contour extraction** (`cv2.findContours`) produces bounding boxes.
5. Contours smaller than `MIN_CONTOUR_AREA` (1500 px²) are discarded.

### 2.5 Centroid Tracking

A lightweight **greedy nearest-neighbour** tracker assigns persistent IDs:

1. Compute the centroid of each bounding box.
2. Build a pairwise distance matrix between existing and new centroids.
3. Sort by distance; greedily match closest unmatched pairs.
4. Unmatched existing tracks are aged (`lost_count += 1`).
5. Tracks unseen for `max_lost` frames are deleted.
6. Unmatched new detections create fresh track IDs.

### 2.6 Vehicle Classification

Vehicles are classified as **"truck"** or **"car"** based on bounding-box area:

- `TRUCK_MIN_AREA = 9000 px²` → box area ≥ 9000 → "truck"
- Smaller boxes → "car"

---

## 3. Implementation

### 3.1 Architecture

The entire script is **functional** (no classes). All mutable state is stored in plain Python `dict` objects.

```
┌──────────────────────────────┐
│  Configuration (constants)   │
├──────────────────────────────┤
│  Perspective calibration     │
│  Tripwire distance compute   │
├──────────────────────────────┤
│  Centroid Tracker (dict)     │  make_tracker_state() / update_tracker()
│  Speed Stats (list)          │  ss_add() / ss_avg() / ss_min() / ...
│  Tripwire Crossing (dict)    │  make_tripwire_state() / update_tripwire()
├──────────────────────────────┤
│  Drawing helpers             │  draw_detection() / draw_tripwires() / draw_hud()
├──────────────────────────────┤
│  main()  — frame loop        │
│   1. Background subtraction  │
│   2. Contour extraction      │
│   3. Centroid tracking       │
│   4a. BEV speed              │
│   4b. Tripwire crossing      │
│   5. Annotation & write      │
│   6. CSV export              │
└──────────────────────────────┘
```

### 3.2 Key Functions

| Function                                    | Purpose                                                          |
| ------------------------------------------- | ---------------------------------------------------------------- |
| `transform_point(pt)`                       | Map one pixel (x, y) → BEV metres via `cv2.perspectiveTransform` |
| `make_tracker_state()`                      | Create the tracker state dict (next_id, centroids, lost_count)   |
| `update_tracker(state, rects)`              | Greedy nearest-neighbour matching, returns `{tid: (cx,cy)}`      |
| `ss_add/avg/min/max/count()`                | Aggregate speed statistics stored as `list[float]`               |
| `make_tripwire_state()`                     | Create tripwire state dict (line positions, distances, fps)      |
| `update_tripwire(state, active, frame_idx)` | Detect Line 1/2 crossings; compute speed on Line 2 crossing      |
| `get_tripwire_speed(state, tid)`            | Query the corrected speed for a given track                      |
| `draw_detection()`                          | Draw bounding box + speed label on the frame                     |
| `draw_tripwires()`                          | Draw two green lines across the road                             |
| `draw_hud()`                                | Draw semi-transparent status overlay                             |
| `main()`                                    | Frame-by-frame processing loop + CSV export                      |

### 3.3 Outputs

| Output                            | Description                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------- |
| `traffic_speed_output_opencv.mp4` | Annotated video with boxes, speeds, green lines, and HUD                               |
| `vehicle_speeds.csv`              | One row per vehicle: type, tripwire speed (corrected + naive), BEV speed (avg/min/max) |

### 3.4 Dependencies

| Package         | Role                                                              |
| --------------- | ----------------------------------------------------------------- |
| `opencv-python` | Video I/O, background subtraction, perspective transform, drawing |
| `numpy`         | Numerical operations, distance matrices, statistics               |

---

## 4. Code Walkthrough

### Module-level setup (lines 1–99)

1. **Imports** — `os`, `csv`, `cv2`, `numpy`, `collections`.
2. **Configuration** — Paths, FPS, detection/tracking thresholds, green-line coordinates.
3. **Perspective matrix** — `cv2.getPerspectiveTransform(SOURCE, TARGET)` computes the 3×3 homography.
4. **Tripwire distance** — Midpoints of both lines are projected through the homography; the Y-difference gives the corrected distance (≈30.1 m). The naive distance (≈4.8 m) is computed for comparison.

### Centroid tracker (lines 113–206)

- `make_tracker_state()` creates a dict with `next_id=0`, empty `centroids`, and `lost_count`.
- `update_tracker()` computes `new_cents` (centroids of new rects), builds a distance matrix, sorts pairs by ascending distance, and greedily assigns matches. Unmatched existing tracks are aged; unmatched new detections spawn new IDs.

### Speed stats (lines 209–229)

- Each track's speed history is a plain `list[float]` in a `defaultdict(list)`.
- `ss_add()`, `ss_avg()`, `ss_min()`, `ss_max()`, `ss_count()` operate on these lists.

### Tripwire crossing (lines 232–285)

- `make_tripwire_state()` stores line Y-positions, distances, FPS, and per-track dicts for `prev_y`, `crossed_1`, `speed_corr`, `speed_naive`.
- `update_tripwire()` iterates over active tracks, checks if the centroid Y crossed a line between this frame and the previous, and records the crossing frame or computes the speed.
- `get_tripwire_speed()` returns the corrected speed for a given track ID, or `None`.

### Main pipeline (lines 349–534)

1. Open video, read metadata (width, height, total frames).
2. Create MOG2 background subtractor with morphological kernels.
3. Initialise tracker, tripwire, and per-track data dicts.
4. **Frame loop** (lines 393–477):
   - Apply MOG2 → threshold → open → close → dilate → clean foreground mask.
   - Extract contours → bounding boxes (filter by area ≥ 1500 px²).
   - Update centroid tracker → get active tracks.
   - Re-associate each track with its nearest bounding box (for drawing).
   - Compute BEV speed: project centroid → store Y → median displacement → km/h.
   - Update tripwire crossings.
   - Draw only vehicles between Line 1 and Line 2 (zone filtering at draw time).
   - Write annotated frame.
5. **CSV export** (lines 482–525): merge all track IDs from both methods, write one row per vehicle.
6. Print summary with average speeds and perspective correction effect.

---

## 5. Performance

Processing the 10-second video (250 frames at 1280×720) takes approximately **4–5 seconds** on a modern CPU — no GPU required.

---

## 6. Limitations & Possible Improvements

| Limitation                                                       | Improvement                                               |
| ---------------------------------------------------------------- | --------------------------------------------------------- |
| MOG2 struggles in slow traffic / stopped vehicles                | Use optical flow or deep-learning detector                |
| Vehicle type based on box area only                              | Train a classifier or use YOLO (see `monitor_traffic.py`) |
| Single-direction tripwire (both directions captured identically) | Add direction detection via sign of Y-displacement        |
| Fixed green-line coordinates                                     | Implement interactive calibration GUI                     |
