"""
monitor_traffic.py
============================
Lean traffic speed monitor — pure OpenCV, no deep learning.

Pipeline
--------
1. MOG2 background subtraction + morphological cleanup → foreground mask.
2. Contour extraction → bounding boxes.
3. Greedy centroid tracker → stable track IDs.
4. Tripwire crossing (Line 1 → Line 2) → speed in km/h.
5. CSV export of per-vehicle results.

Dependencies: opencv-python, numpy
"""

import os
import csv
import cv2
import numpy as np
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

VIDEO_PATH  = os.path.join("Downloads", "Segments",
                           "Road traffic video for object recognition_part_1.mp4")
OUTPUT_PATH = "traffic_speed_output_opencv_v2.mp4"
CSV_PATH    = "vehicle_speeds_v2.csv"

FPS             = 25.0
MIN_AREA        = 1_500   # px² — minimum contour area to be treated as a vehicle
TRUCK_MIN_AREA  = 9_000   # px² — bounding-box area threshold for car/truck classification
MAX_LOST        = 12      # frames a track survives without a matching detection
MAX_DIST        = 150     # px — max centroid jump between frames to count as same vehicle

# BEV rolling-window fallback speed
BEV_WINDOW      = 8       # recent frames kept per track
BEV_MIN_FRAMES  = 5       # minimum observations before reporting a BEV speed

# ---------------------------------------------------------------------------
# TRIPWIRE LINES  (pixel Y-coordinates, 1280×720 frame)
# ---------------------------------------------------------------------------

LINE_1_Y = 400
LINE_2_Y = 560

# ---------------------------------------------------------------------------
# PERSPECTIVE CALIBRATION  (pixel → metres)
# ---------------------------------------------------------------------------
# SOURCE: trapezoid in camera frame that maps to a known road rectangle.
# TARGET: that rectangle in real-world metres (width ≈ 28 m, length ≈ 102 m).

SOURCE = np.array([[490, 295], [790, 295], [1260, 700], [20, 700]], dtype=np.float32)
TARGET = np.array([[0, 0],    [28, 0],    [28, 102],   [0, 102]],  dtype=np.float32)

M = cv2.getPerspectiveTransform(SOURCE, TARGET)

# ---------------------------------------------------------------------------
# TRIPWIRE DISTANCE (perspective-corrected)
# ---------------------------------------------------------------------------

def _bev_y(pix_y, pix_x=640):
    """Return the BEV Y-coordinate (metres) for a given pixel position."""
    pt = np.array([[[float(pix_x), float(pix_y)]]], dtype=np.float32)
    return float(cv2.perspectiveTransform(pt, M)[0, 0, 1])

DIST_M = abs(_bev_y(LINE_2_Y) - _bev_y(LINE_1_Y))
print(f"Tripwire distance (perspective-corrected): {DIST_M:.2f} m\n")

# ---------------------------------------------------------------------------
# CENTROID TRACKER
# ---------------------------------------------------------------------------

def make_tracker():
    return {"next_id": 0, "centroids": {}, "lost": defaultdict(int)}


def update_tracker(t, rects):
    cents    = t["centroids"]
    lost     = t["lost"]
    new_pts  = [(int(x + w / 2), int(y + h / 2)) for x, y, w, h in rects]

    if not rects:
        for tid in list(cents):
            lost[tid] += 1
            if lost[tid] > MAX_LOST:
                del cents[tid]; del lost[tid]
        return dict(cents)

    if not cents:
        for pt in new_pts:
            cents[t["next_id"]] = pt
            lost[t["next_id"]]  = 0
            t["next_id"] += 1
        return dict(cents)

    ex_ids = list(cents)
    ex_pts = [cents[i] for i in ex_ids]
    dist   = np.linalg.norm(
        np.array(ex_pts)[:, None] - np.array(new_pts)[None, :], axis=-1)
    pairs  = sorted(
        [(dist[i, j], i, j) for i in range(len(ex_pts)) for j in range(len(new_pts))],
        key=lambda t: t[0])

    used_e, used_n = set(), set()
    for d, ei, ni in pairs:
        if ei in used_e or ni in used_n: continue
        if d > MAX_DIST: break
        tid = ex_ids[ei]
        cents[tid] = new_pts[ni]
        lost[tid]  = 0
        used_e.add(ei); used_n.add(ni)

    for ei, tid in enumerate(ex_ids):
        if ei not in used_e:
            lost[tid] += 1
            if lost[tid] > MAX_LOST:
                del cents[tid]; del lost[tid]

    for ni, pt in enumerate(new_pts):
        if ni not in used_n:
            cents[t["next_id"]] = pt
            lost[t["next_id"]]  = 0
            t["next_id"] += 1

    return dict(cents)

# ---------------------------------------------------------------------------
# TRIPWIRE STATE
# ---------------------------------------------------------------------------

def make_tripwire():
    return {"prev_y": {}, "crossed_1": {}, "speed": {}}


def _crossed(prev_y, cur_y, line_y):
    """True if centroid moved from one side of line_y to the other."""
    return (prev_y < line_y <= cur_y) or (cur_y < line_y <= prev_y)


def update_tripwire(tw, active, frame_idx):
    for tid, (_, cy) in active.items():
        prev = tw["prev_y"].get(tid)
        tw["prev_y"][tid] = cy
        if prev is None:
            continue
        if tid not in tw["crossed_1"]:
            if _crossed(prev, cy, LINE_1_Y):
                tw["crossed_1"][tid] = frame_idx
        elif tid not in tw["speed"]:
            if _crossed(prev, cy, LINE_2_Y):
                elapsed = (frame_idx - tw["crossed_1"][tid]) / FPS
                if elapsed > 0:
                    tw["speed"][tid] = (DIST_M / elapsed) * 3.6  # km/h

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {VIDEO_PATH}")

    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO_PATH}  |  {w}x{h}  |  {total} frames")

    bg  = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=True)
    k3  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k15 = cv2.getStructuringElement(cv2.MORPH_RECT,   (15, 15))

    tracker   = make_tracker()
    tripwire  = make_tripwire()
    vtype     = {}                                              # tid → "car" | "truck"
    bev_hist  = defaultdict(lambda: deque(maxlen=BEV_WINDOW))  # tid → recent BEV-Y values
    bev_speed = {}                                              # tid → latest BEV speed (km/h)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (w, h))

    for frame_idx in range(1, total + 1):
        ret, frame = cap.read()
        if not ret:
            break

        # -- Foreground mask --------------------------------------------------
        fg = bg.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k3)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k11)
        fg = cv2.dilate(fg, k15)

        # -- Detections -------------------------------------------------------
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= MIN_AREA]

        # -- Track update, BEV speed & tripwire -------------------------------
        active = update_tracker(tracker, rects)
        update_tripwire(tripwire, active, frame_idx)

        # BEV rolling-window speed for every active track
        for tid, (cx, cy) in active.items():
            bev_pt = cv2.perspectiveTransform(
                np.array([[[float(cx), float(cy)]]], dtype=np.float32), M)[0, 0]
            bev_hist[tid].append(float(bev_pt[1]))
            h = list(bev_hist[tid])
            if len(h) >= BEV_MIN_FRAMES:
                diffs = [abs(h[i+1] - h[i]) for i in range(len(h)-1)]
                spd = float(np.median(diffs)) * FPS * 3.6
                if 1.0 < spd < 200.0:
                    bev_speed[tid] = spd

        # -- Annotate ---------------------------------------------------------
        ann = frame.copy()
        cv2.line(ann, (340, LINE_1_Y), (940,  LINE_1_Y), (0, 255, 0), 2)
        cv2.line(ann, (100, LINE_2_Y), (1180, LINE_2_Y), (0, 255, 0), 2)

        for tid, (cx, cy) in active.items():
            # Find closest rect to this centroid (all detections, not just in-zone)
            rect = min(
                rects,
                key=lambda r: abs(cx - (r[0] + r[2]//2)) + abs(cy - (r[1] + r[3]//2)),
                default=None,
            )
            if rect is not None:
                rx, ry, rw, rh = rect
                vtype[tid] = "truck" if rw * rh >= TRUCK_MIN_AREA else "car"

            # Only draw boxes and labels while in the measurement zone
            if not (LINE_1_Y <= cy <= LINE_2_Y):
                continue
            if rect is None:
                continue
            rx, ry, rw, rh = rect
            spd_trip = tripwire["speed"].get(tid)
            spd_bev  = bev_speed.get(tid)
            if spd_trip is not None:
                spd_label = f"{int(spd_trip)} km/h"
            elif spd_bev is not None:
                spd_label = f"~{int(spd_bev)} km/h"
            else:
                spd_label = "--"
            label = f"{vtype[tid].upper()} #{tid}  {spd_label}"
            cv2.rectangle(ann, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)
            cv2.putText(ann, label, (rx, max(ry - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

        out.write(ann)
        if frame_idx % 50 == 0 or frame_idx == total:
            print(f"  {frame_idx}/{total}", end="\r")

    cap.release()
    out.release()

    # -- CSV export -----------------------------------------------------------
    # Union of every tid ever seen in the zone (vtype) and every tid that
    # completed a tripwire crossing — so no vehicle is silently dropped.
    all_tids = sorted(set(vtype) | set(tripwire["speed"]))
    with open(CSV_PATH, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["vehicle_id", "vehicle_type", "speed_kmh", "speed_source"])
        for tid in all_tids:
            spd_trip = tripwire["speed"].get(tid)
            spd_bev  = bev_speed.get(tid)
            if spd_trip is not None:
                wr.writerow([tid, vtype.get(tid, "unknown"), f"{spd_trip:.1f}", "tripwire"])
            elif spd_bev is not None:
                wr.writerow([tid, vtype.get(tid, "unknown"), f"{spd_bev:.1f}", "bev_avg"])
            else:
                wr.writerow([tid, vtype.get(tid, "unknown"), "", "none"])

    n_trip = len(tripwire["speed"])
    n_bev  = sum(1 for tid in all_tids if tid not in tripwire["speed"] and tid in bev_speed)
    print(f"\nDone.  Output: {OUTPUT_PATH}  |  CSV: {CSV_PATH}")
    print(f"  Tripwire speeds : {n_trip}")
    print(f"  BEV fallback    : {n_bev}")
    print(f"  No speed        : {len(all_tids) - n_trip - n_bev}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    main()
    print(f"Total time: {time.time() - t0:.1f} s")
