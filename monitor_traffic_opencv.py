"""
monitor_traffic_opencv.py
=========================
Traffic speed monitoring using **pure OpenCV** (no deep learning).

This script processes a motorway video recorded from an overhead bridge
camera and estimates per-vehicle speeds using two complementary methods:

1. **BEV rolling-window speed** – Each vehicle's centroid is mapped to a
   Bird's-Eye-View (BEV) coordinate system via a calibrated perspective
   transform.  The median per-frame displacement over a sliding window is
   converted to km/h.

2. **Tripwire speed** – Two virtual "green lines" (tripwires) span the
   road at known positions.  When a tracked vehicle crosses Line 1 and
   then Line 2, the elapsed time yields a speed estimate.
   The physical distance between the lines is derived from the same
   calibrated perspective transform (perspective-corrected) and also
   computed naively (uniform scale) so both can be compared.

Vehicle detection is performed with background subtraction
(MOG2 + morphological cleanup) and contour extraction.
Tracking uses a lightweight greedy centroid tracker.

Outputs
-------
- An annotated MP4 video with bounding boxes, speed labels, and green
  tripwire lines overlaid.
- A CSV file with per-vehicle speed data (both methods, plus vehicle type).

Dependencies
------------
- Python 3.10+
- opencv-python (cv2)
- numpy
"""

import os
import csv
import cv2
import numpy as np
from collections import defaultdict, deque

# ============================================================================
# CONFIGURATION
# ============================================================================

# -- Paths -------------------------------------------------------------------
VIDEO_NAME  = "Road traffic video for object recognition_part_1.mp4"
VIDEO_PATH  = os.path.join("Downloads", "Segments", VIDEO_NAME)
OUTPUT_PATH = "traffic_speed_output_opencv.mp4"    # annotated output video
CSV_PATH    = "vehicle_speeds.csv"                 # per-vehicle speed report

# -- Video -------------------------------------------------------------------
FPS: float = 25.0   # known frame rate of the source recording

# -- Detection ---------------------------------------------------------------
# Minimum contour area (px^2) after morphological cleanup to be accepted
# as a vehicle.  Smaller blobs (noise, pedestrians) are discarded.
MIN_CONTOUR_AREA = 1_500

# -- BEV speed smoothing -----------------------------------------------------
# Number of recent BEV positions kept per track for the rolling-window
# speed estimate.  At 25 fps, 8 frames = 0.32 s.
SPEED_WINDOW: int = 8

# Minimum number of BEV positions required before reporting a speed for a
# track.  Prevents noisy estimates from very short tracks.
MIN_FRAMES_FOR_SPEED: int = 5

# -- Vehicle classification ---------------------------------------------------
# Bounding-box area threshold (px^2).  Boxes whose area >= this value are
# labelled "truck"; smaller boxes are labelled "car".
TRUCK_MIN_AREA: int = 9_000

# -- Zone margin --------------------------------------------------------------
# Extra pixel margin above Line 1 and below Line 2 so the tracker can
# "see" vehicles a few frames before they actually cross the line.
# This ensures the crossing event is reliably detected.
ZONE_MARGIN: int = 30

# ============================================================================
# GREEN-LINE TRIPWIRE POSITIONS  (pixel coordinates in the 1280x720 frame)
# ============================================================================
# Two virtual lines span the full road width.  Each line represents a
# real-world length of 25 m (given by the assignment specification).
# The user must adjust these coordinates to match the exact lines visible
# in their specific video.

GREEN_LINE_LENGTH_M = 25.0   # real-world length of each green line (metres)

# Upper green line (Line 1) — further from the camera
LINE_1_LEFT  = np.array([340, 400], dtype=np.float32)   # left endpoint  (x, y)
LINE_1_RIGHT = np.array([940, 400], dtype=np.float32)   # right endpoint (x, y)

# Lower green line (Line 2) — closer to the camera
LINE_2_LEFT  = np.array([100, 560], dtype=np.float32)
LINE_2_RIGHT = np.array([1180, 560], dtype=np.float32)

# Convenience: integer Y-coordinates for crossing detection
LINE_1_Y = int(LINE_1_LEFT[1])   # 400
LINE_2_Y = int(LINE_2_LEFT[1])   # 560

# ============================================================================
# PERSPECTIVE CALIBRATION
# ============================================================================
# A homography (3x3 matrix) that maps pixel coordinates in the camera
# frame to real-world metre coordinates on a flat road surface.
#
# SOURCE defines a trapezoid in the 1280x720 image that corresponds to
# a rectangular patch of road.  TARGET defines that rectangle in metres.
#
# These values were calibrated from lane-dash markings:
#   - Road width  ~28 m  (4 lanes + median + shoulders)
#   - Road length ~102 m  (6 dash-groups x 17 m each)

SOURCE = np.array([
    [ 490, 295],   # 0: top-left     (far left edge of road near horizon)
    [ 790, 295],   # 1: top-right    (far right edge near horizon)
    [1260, 700],   # 2: bottom-right (near right shoulder)
    [  20, 700],   # 3: bottom-left  (near left shoulder)
], dtype=np.float32)

TARGET = np.array([
    [ 0,   0],     # 0 -> (0 m, 0 m)
    [28,   0],     # 1 -> (28 m, 0 m)
    [28, 102],     # 2 -> (28 m, 102 m)
    [ 0, 102],     # 3 -> (0 m, 102 m)
], dtype=np.float32)

PERSPECTIVE_MATRIX = cv2.getPerspectiveTransform(SOURCE, TARGET)

# ============================================================================
# TRIPWIRE DISTANCE COMPUTATION
# ============================================================================
# The real-world distance between the two green lines is computed by
# projecting their midpoints through the calibrated perspective matrix.
# This gives the "perspective-corrected" distance.
#
# A "naive" distance is also computed for comparison: it assumes a
# uniform scale (average of the two known green-line pixel widths)
# and simply converts the pixel gap between the lines.

def _transform_pt(pt):
    """Transform a single 2-D point through the perspective matrix."""
    src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    return cv2.perspectiveTransform(src, PERSPECTIVE_MATRIX)[0, 0]

# Midpoints of each green line (pixel space)
_centre_1 = (LINE_1_LEFT + LINE_1_RIGHT) / 2.0   # centre of Line 1
_centre_2 = (LINE_2_LEFT + LINE_2_RIGHT) / 2.0   # centre of Line 2

# Project midpoints to BEV (metre space)
_bev_1 = _transform_pt(_centre_1)
_bev_2 = _transform_pt(_centre_2)

# Perspective-corrected distance (metres, along road)
TRIPWIRE_DISTANCE_M = abs(float(_bev_2[1]) - float(_bev_1[1]))

# Naive distance: uniform scale from the average pixel width of both lines
_L1 = float(np.linalg.norm(LINE_1_RIGHT - LINE_1_LEFT))  # px width Line 1
_L2 = float(np.linalg.norm(LINE_2_RIGHT - LINE_2_LEFT))  # px width Line 2
_y1 = float(LINE_1_LEFT[1])
_y2 = float(LINE_2_LEFT[1])
NAIVE_DISTANCE_M = (_y2 - _y1) * GREEN_LINE_LENGTH_M / ((_L1 + _L2) / 2.0)

# Print calibration summary at startup
print(f"BEV Line 1 centre            : y = {_bev_1[1]:.1f} m")
print(f"BEV Line 2 centre            : y = {_bev_2[1]:.1f} m")
print(f"Perspective-corrected dist.  : {TRIPWIRE_DISTANCE_M:.2f} m")
print(f"Naive (avg-scale) dist.      : {NAIVE_DISTANCE_M:.2f} m")
print(f"Difference                   : {(TRIPWIRE_DISTANCE_M / NAIVE_DISTANCE_M - 1) * 100:+.1f} %")
print()


# ============================================================================
# UTILITY: POINT TRANSFORM
# ============================================================================

def transform_point(pt):
    """
    Map a single pixel coordinate (x, y) to the BEV plane in metres.

    Parameters
    ----------
    pt : array-like of shape (2,)
        Pixel coordinate [x, y].

    Returns
    -------
    numpy.ndarray of shape (2,)
        Real-world coordinate [x_m, y_m] in metres.
    """
    src = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, PERSPECTIVE_MATRIX)
    return dst[0, 0]


# ============================================================================
# CENTROID TRACKER  (functional / dict-based)
# ============================================================================
# A lightweight greedy nearest-neighbour tracker.
# Each frame, new bounding-box centroids are matched to existing tracks.
# Unmatched tracks are aged out after `max_lost` consecutive frames.
# Unmatched detections spawn new track IDs.

def make_tracker_state(max_lost=8, max_distance=120):
    """
    Create a fresh tracker state dictionary.

    Parameters
    ----------
    max_lost : int
        How many consecutive frames a track can go unmatched before
        it is deleted.
    max_distance : int
        Maximum pixel distance between an existing centroid and a new
        detection for them to be matched.

    Returns
    -------
    dict
        Mutable state dictionary used by ``update_tracker()``.
    """
    return {
        "next_id":      0,                    # next track ID to assign
        "centroids":    {},                    # tid -> (cx, cy)
        "lost_count":   defaultdict(int),      # tid -> frames unseen
        "max_lost":     max_lost,
        "max_distance": max_distance,
    }


def update_tracker(state, rects):
    """
    Match new bounding boxes to existing tracks using greedy
    nearest-neighbour on centroid distance.

    Parameters
    ----------
    state : dict
        Tracker state created by ``make_tracker_state()``.
    rects : list of (x, y, w, h)
        Bounding boxes detected in the current frame.

    Returns
    -------
    dict
        Mapping ``tid -> (cx, cy)`` for every active track after this
        frame's update.
    """
    centroids    = state["centroids"]
    lost_count   = state["lost_count"]
    max_lost     = state["max_lost"]
    max_distance = state["max_distance"]

    # -- No detections: age all existing tracks ---
    if not rects:
        for tid in list(centroids):
            lost_count[tid] += 1
            if lost_count[tid] > max_lost:
                del centroids[tid]
                del lost_count[tid]
        return dict(centroids)

    # Compute centroids of new detections
    new_cents = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]

    # -- First frame (no existing tracks): register all detections ---
    if not centroids:
        for nc in new_cents:
            centroids[state["next_id"]] = nc
            lost_count[state["next_id"]] = 0
            state["next_id"] += 1
        return dict(centroids)

    ex_ids  = list(centroids)
    ex_pts  = [centroids[i] for i in ex_ids]
    used_ex = set()   # indices of matched existing tracks
    used_nw = set()   # indices of matched new detections

    # Build pairwise distance matrix  (n_existing x n_new)
    dist_mat = np.linalg.norm(
        np.array(ex_pts)[:, None, :] - np.array(new_cents)[None, :, :],
        axis=-1,
    )

    # Sort all (distance, existing_idx, new_idx) triples by distance
    pairs = sorted(
        [(dist_mat[i, j], i, j)
         for i in range(len(ex_pts))
         for j in range(len(new_cents))],
        key=lambda t: t[0],
    )

    # Greedy assignment: pick the closest unmatched pair each iteration
    for d, ei, ni in pairs:
        if ei in used_ex or ni in used_nw:
            continue
        if d > max_distance:
            break                          # all remaining pairs are too far
        tid = ex_ids[ei]
        centroids[tid]  = new_cents[ni]    # update track position
        lost_count[tid] = 0                # reset lost counter
        used_ex.add(ei)
        used_nw.add(ni)

    # Age out unmatched existing tracks
    for ei, tid in enumerate(ex_ids):
        if ei not in used_ex:
            lost_count[tid] += 1
            if lost_count[tid] > max_lost:
                del centroids[tid]
                del lost_count[tid]

    # Register unmatched new detections as fresh tracks
    for ni, nc in enumerate(new_cents):
        if ni not in used_nw:
            centroids[state["next_id"]] = nc
            lost_count[state["next_id"]] = 0
            state["next_id"] += 1

    return dict(centroids)


# ============================================================================
# SPEED STATS HELPERS  (functional / list-based)
# ============================================================================
# Instead of a class, each track's speed history is a plain ``list[float]``
# stored in a ``defaultdict(list)``.  These helper functions provide
# aggregate statistics over those samples.

def ss_add(samples, kmh):
    """Append a speed sample (km/h) to the per-track list."""
    samples.append(float(kmh))

def ss_avg(samples):
    """Return the mean speed, or None if no samples."""
    return float(np.mean(samples)) if samples else None

def ss_min(samples):
    """Return the minimum speed, or None if no samples."""
    return float(np.min(samples)) if samples else None

def ss_max(samples):
    """Return the maximum speed, or None if no samples."""
    return float(np.max(samples)) if samples else None

def ss_count(samples):
    """Return the number of speed samples recorded."""
    return len(samples)


# ============================================================================
# TRIPWIRE CROSSING  (functional / dict-based)
# ============================================================================
# The tripwire system records the frame at which a vehicle's centroid
# crosses Line 1, then Line 2.  The elapsed frame count yields
# a time, and dividing the known inter-line distance by that time
# gives the average speed of the vehicle across that section.

def make_tripwire_state(line_1_y, line_2_y, corrected_dist_m, naive_dist_m, fps):
    """
    Create a fresh tripwire state dictionary.

    Parameters
    ----------
    line_1_y, line_2_y : int
        Pixel Y-coordinates of the two tripwire lines.
    corrected_dist_m : float
        Perspective-corrected real-world distance between the lines (m).
    naive_dist_m : float
        Naive (uniform-scale) distance between the lines (m).
    fps : float
        Video frame rate (frames per second).

    Returns
    -------
    dict
        Mutable state dictionary used by ``update_tripwire()`` and
        ``get_tripwire_speed()``.
    """
    return {
        "line_1_y":         line_1_y,
        "line_2_y":         line_2_y,
        "corrected_dist_m": corrected_dist_m,
        "naive_dist_m":     naive_dist_m,
        "fps":              fps,
        "prev_y":           {},   # tid -> previous centroid Y
        "crossed_1":        {},   # tid -> frame index when Line 1 was crossed
        "speed_corr":       {},   # tid -> perspective-corrected speed (km/h)
        "speed_naive":      {},   # tid -> naive speed (km/h)
    }


def update_tripwire(state, active, frame_idx):
    """
    Check every active track for a tripwire line crossing.

    A vehicle must cross Line 1 first, then Line 2.  When the second
    crossing is detected, speeds are computed from the elapsed time and
    the known inter-line distances.

    Parameters
    ----------
    state : dict
        State dictionary from ``make_tripwire_state()``.
    active : dict
        ``tid -> (cx, cy)`` mapping from the centroid tracker.
    frame_idx : int
        Current frame number (1-based).
    """
    for tid, (cx, cy) in active.items():
        prev = state["prev_y"].get(tid)
        state["prev_y"][tid] = cy

        if prev is None:
            continue    # first observation for this track; nothing to compare

        # --- Check crossing of Line 1 (either direction) ---
        if tid not in state["crossed_1"]:
            if prev <= state["line_1_y"] < cy or cy <= state["line_1_y"] < prev:
                state["crossed_1"][tid] = frame_idx

        # --- Check crossing of Line 2 (only after Line 1 was crossed) ---
        elif tid not in state["speed_corr"]:
            if prev <= state["line_2_y"] < cy or cy <= state["line_2_y"] < prev:
                elapsed_frames = frame_idx - state["crossed_1"][tid]
                if elapsed_frames > 0:
                    elapsed_s = elapsed_frames / state["fps"]
                    # speed = distance / time, converted m/s -> km/h (* 3.6)
                    state["speed_corr"][tid]  = (state["corrected_dist_m"] / elapsed_s) * 3.6
                    state["speed_naive"][tid] = (state["naive_dist_m"]     / elapsed_s) * 3.6


def get_tripwire_speed(state, tid):
    """
    Return the perspective-corrected tripwire speed for a track, or None.

    Parameters
    ----------
    state : dict
        Tripwire state dictionary.
    tid : int
        Track ID.

    Returns
    -------
    float or None
        Speed in km/h if the vehicle completed both crossings, else None.
    """
    return state["speed_corr"].get(tid)


# ============================================================================
# DRAWING / ANNOTATION HELPERS
# ============================================================================

# Colour palette for bounding boxes (one per track ID, cycled)
PALETTE = [
    (255,  85,  85), ( 85, 255,  85), ( 85,  85, 255), (255, 200,   0),
    (  0, 220, 220), (220,   0, 220), (255, 160,  10), (160, 255,  10),
]

def id_color(tid):
    """Return a deterministic colour for a given track ID."""
    return PALETTE[tid % len(PALETTE)]


def draw_detection(frame, rect, tid, speed_bev, speed_trip, vtype="car"):
    """
    Draw a bounding box and speed label for one vehicle.

    The label shows the vehicle type (CAR / TRUCK), track ID, and speed.
    If a tripwire speed is available it is shown directly; otherwise the
    BEV speed is shown prefixed with '~' (approximate).

    Parameters
    ----------
    frame : numpy.ndarray
        The video frame to draw on (modified in-place).
    rect : tuple (x, y, w, h)
        Bounding box in pixel coordinates.
    tid : int
        Track ID.
    speed_bev : float or None
        BEV rolling-window speed (km/h), or None.
    speed_trip : float or None
        Tripwire speed (km/h), or None.
    vtype : str
        Vehicle type label ('car' or 'truck').
    """
    x, y, w, h = rect
    color = id_color(tid)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Compose label text
    tag = vtype.upper()
    label = f"{tag} #{tid}"
    if speed_trip is not None:
        label += f"  {int(speed_trip)} km/h"       # tripwire speed (exact)
    elif speed_bev is not None:
        label += f"  ~{int(speed_bev)} km/h"       # BEV speed (approximate)
    else:
        label += "  --"                             # no speed yet

    # Draw label background and text
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg_y1 = max(y - th - 8, 0)
    cv2.rectangle(frame, (x, bg_y1), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, max(y - 4, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tripwires(frame):
    """Draw the two green tripwire lines with labels on the frame."""
    green = (0, 255, 0)
    cv2.line(frame, tuple(LINE_1_LEFT.astype(int)), tuple(LINE_1_RIGHT.astype(int)), green, 2)
    cv2.line(frame, tuple(LINE_2_LEFT.astype(int)), tuple(LINE_2_RIGHT.astype(int)), green, 2)
    cv2.putText(frame, "Line 1 (25 m)", (int(LINE_1_RIGHT[0]) + 10, int(LINE_1_RIGHT[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1, cv2.LINE_AA)
    cv2.putText(frame, "Line 2 (25 m)", (int(LINE_2_RIGHT[0]) + 10, int(LINE_2_RIGHT[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, green, 1, cv2.LINE_AA)


def draw_hud(frame, vehicle_count, frame_idx, total):
    """
    Draw a semi-transparent heads-up display showing vehicle count,
    progress, and tripwire distance info.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (440, 100), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Vehicles in frame: {vehicle_count}",
                (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)
    pct = frame_idx / total * 100 if total else 0
    cv2.putText(frame, f"Frame {frame_idx}/{total}  ({pct:.0f}%)",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Tripwire dist: {TRIPWIRE_DISTANCE_M:.1f}m (corrected)"
                       f"  |  {NAIVE_DISTANCE_M:.1f}m (naive)",
                (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 220, 100), 1, cv2.LINE_AA)


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main():
    """
    Run the full traffic monitoring pipeline:

    1. Open the input video and initialise the background subtractor,
       morphological kernels, tracker, and tripwire state.
    2. For each frame:
       a. Extract foreground mask (MOG2 + morphology).
       b. Find contours -> bounding boxes.
       c. Update the centroid tracker with new detections.
       d. Compute BEV rolling-window speed for every active track.
       e. Check for tripwire line crossings and compute crossing speed.
       f. Annotate the frame (boxes, labels, green lines, HUD).
    3. Write the annotated frame to the output video.
    4. After all frames: export per-vehicle CSV and print summary.
    """
    # -- Open video ----------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    fps    = FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video  : {VIDEO_PATH}")
    print(f"Size   : {width}x{height}  |  FPS: {fps}  |  Frames: {total}  ({total/fps:.1f}s)")
    print(f"Output : {OUTPUT_PATH}  |  CSV: {CSV_PATH}")

    # -- Background subtractor (MOG2) ----------------------------------------
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=40, detectShadows=True
    )

    # Morphological kernels for foreground mask cleanup
    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))    # remove noise
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # fill gaps
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,    (15, 15))  # expand blobs

    # -- Initialise tracker and tripwire state --------------------------------
    tracker = make_tracker_state(max_lost=4, max_distance=100)

    tripwire = make_tripwire_state(
        LINE_1_Y, LINE_2_Y,
        TRIPWIRE_DISTANCE_M, NAIVE_DISTANCE_M,
        fps,
    )

    # Per-track data stores
    bev_history   = defaultdict(lambda: deque(maxlen=SPEED_WINDOW))  # BEV Y history
    speed_stats   = defaultdict(list)   # tid -> [speed samples (km/h)]
    tid_to_rect   = {}                  # tid -> last known bounding box
    instant_speed = {}                  # tid -> current-frame BEV speed
    tid_to_type   = {}                  # tid -> "car" or "truck"

    # -- Video writer --------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_idx = 0

    # ========================================================================
    # FRAME LOOP
    # ========================================================================
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- Step 1: Foreground mask (background subtraction) ----------------
        fg = bg_sub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)  # remove shadow pixels
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open)      # erode then dilate (noise)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)     # dilate then erode (gaps)
        fg = cv2.dilate(fg, k_dilate, iterations=1)              # grow blobs for merging

        # --- Step 2: Contour extraction -> bounding boxes --------------------
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours
                 if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

        # --- Step 3: Update centroid tracker ---------------------------------
        active = update_tracker(tracker, rects)

        # Re-associate each track ID with its closest bounding box
        new_tid_to_rect = {}
        used_ri = set()
        for tid, (cx, cy) in active.items():
            best_d, best_i = float("inf"), None
            for ri, (rx, ry, rw, rh) in enumerate(rects):
                if ri in used_ri:
                    continue
                d = ((cx - (rx + rw//2))**2 + (cy - (ry + rh//2))**2) ** 0.5
                if d < best_d:
                    best_d, best_i = d, ri
            if best_i is not None and best_d < 100:
                new_tid_to_rect[tid] = rects[best_i]
                used_ri.add(best_i)
            elif tid in tid_to_rect:
                new_tid_to_rect[tid] = tid_to_rect[tid]   # keep last known rect
        tid_to_rect = new_tid_to_rect

        # --- Step 4a: BEV rolling-window speed (fallback / real-time) --------
        instant_speed.clear()

        for tid, (cx, cy) in active.items():
            # Project centroid to BEV and store the Y-coordinate (along-road)
            bev = transform_point(np.array([cx, cy], dtype=np.float32))
            bev_history[tid].append(float(bev[1]))

            h = list(bev_history[tid])
            if len(h) >= MIN_FRAMES_FOR_SPEED:
                # Compute per-frame displacements and take the median
                diffs = [abs(h[i + 1] - h[i]) for i in range(len(h) - 1)]
                median_disp_m = float(np.median(diffs))
                spd = median_disp_m * fps * 3.6   # m/frame -> km/h
                if 1.0 < spd < 180.0:             # sanity filter
                    instant_speed[tid] = spd
                    ss_add(speed_stats[tid], spd)

        # --- Step 4b: Tripwire crossing detection ----------------------------
        update_tripwire(tripwire, active, frame_idx)

        # --- Step 5: Annotate and write frame --------------------------------
        annotated = frame.copy()
        draw_tripwires(annotated)

        # Only draw bounding boxes for vehicles inside the tripwire zone
        for tid, (cx, cy) in active.items():
            if not (LINE_1_Y <= cy <= LINE_2_Y):
                continue   # centroid outside the zone -> skip drawing
            rect = tid_to_rect.get(tid)
            if rect is None:
                continue

            # Classify vehicle by bounding box area
            _, _, rw, rh = rect
            vtype = "truck" if (rw * rh) >= TRUCK_MIN_AREA else "car"
            tid_to_type[tid] = vtype

            draw_detection(annotated, rect, tid,
                           instant_speed.get(tid),
                           get_tripwire_speed(tripwire, tid),
                           vtype)

        # Count vehicles strictly between the two lines for the HUD
        in_zone = sum(1 for _, (_, cy) in active.items()
                      if LINE_1_Y <= cy <= LINE_2_Y)
        draw_hud(annotated, in_zone, frame_idx, total)
        writer.write(annotated)

        # Progress indicator
        if frame_idx % 50 == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total else 0
            print(f"  Frame {frame_idx}/{total}  ({pct:.0f}%)", end="\r")

    cap.release()
    writer.release()

    # ========================================================================
    # CSV EXPORT
    # ========================================================================
    # Merge track IDs from both speed methods
    all_tids = sorted(set(
        [tid for tid, s in speed_stats.items() if ss_count(s) > 0]
        + list(tripwire["speed_corr"].keys())
    ))

    with open(CSV_PATH, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "vehicle_id",
            "vehicle_type",
            "tripwire_speed_corrected_kmh",
            "tripwire_speed_naive_kmh",
            "bev_avg_speed_kmh",
            "bev_min_speed_kmh",
            "bev_max_speed_kmh",
        ])
        for tid in all_tids:
            s = speed_stats.get(tid)
            writer_csv.writerow([
                tid,
                tid_to_type.get(tid, "unknown"),
                f"{tripwire['speed_corr'][tid]:.1f}"
                    if tid in tripwire["speed_corr"] else "",
                f"{tripwire['speed_naive'][tid]:.1f}"
                    if tid in tripwire["speed_naive"] else "",
                f"{ss_avg(s):.1f}"   if s and ss_count(s) else "",
                f"{ss_min(s):.1f}"   if s and ss_count(s) else "",
                f"{ss_max(s):.1f}"   if s and ss_count(s) else "",
            ])

    # ========================================================================
    # SUMMARY
    # ========================================================================
    n_trip = len(tripwire["speed_corr"])
    n_bev  = sum(1 for s in speed_stats.values() if ss_count(s) > 0)

    print(f"\nDone!")
    print(f"  Video : {OUTPUT_PATH}")
    print(f"  CSV   : {CSV_PATH}  ({n_trip} tripwire + {n_bev} BEV measurements)")

    if n_trip > 0:
        avg_corr  = np.mean(list(tripwire["speed_corr"].values()))
        avg_naive = np.mean(list(tripwire["speed_naive"].values()))
        print(f"\n  Tripwire avg (perspective-corrected): {avg_corr:.1f} km/h")
        print(f"  Tripwire avg (naive / no correction): {avg_naive:.1f} km/h")
        print(f"  Perspective correction effect:         {((avg_corr/avg_naive)-1)*100:+.1f} %")


if __name__ == "__main__":
    import time
    start_time = time.time()
    print(f'start time: {start_time}')
    main()
    print(f'total duration: {time.time() - start_time}')
