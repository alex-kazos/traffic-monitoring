"""
Traffic speed estimation – TWO-LINE TRIPWIRE with perspective correction.
=========================================================================

Method
------
Two horizontal "virtual lines" (green) are drawn across the road at known
pixel-Y positions.  Each line spans 25 m in real-world length (given by the
assignment).

When a vehicle's centroid crosses Line 1, we record the frame number.
When the same vehicle later crosses Line 2, we compute:

    speed = perspective_corrected_distance / elapsed_time

The perspective correction uses the known 25 m green-line lengths at two
different Y positions to derive the road's vanishing point and, from that,
the true metric distance between the lines.

Naive (uniform-scale) speed is also computed for comparison.
"""

import os
import csv
import cv2
import numpy as np
from collections import defaultdict, deque

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
VIDEO_NAME  = "Road traffic video for object recognition_part_1.mp4"
VIDEO_PATH  = os.path.join("Downloads", "Segments", VIDEO_NAME)
OUTPUT_PATH = "traffic_speed_tripwire.mp4"
CSV_PATH    = "vehicle_speeds_tripwire.csv"

FPS: float       = 25.0
MIN_CONTOUR_AREA = 1_500          # px²  – ignore smaller blobs

# ---------------------------------------------------------------------------
# GREEN-LINE TRIPWIRE POSITIONS  (pixel coordinates in 1280×720 frame)
# ---------------------------------------------------------------------------
# *** Adjust these to match the EXACT green lines visible in YOUR video ***

# Upper green line (Line 1 – further from camera)
LINE_1_LEFT  = np.array([340, 400], dtype=np.float32)
LINE_1_RIGHT = np.array([940, 400], dtype=np.float32)

# Lower green line (Line 2 – closer to camera)
LINE_2_LEFT  = np.array([100, 560], dtype=np.float32)
LINE_2_RIGHT = np.array([1180, 560], dtype=np.float32)

GREEN_LINE_LENGTH_M = 25.0        # each green line = 25 m (given)

# Derived pixel-Y shortcuts
LINE_1_Y = int(LINE_1_LEFT[1])
LINE_2_Y = int(LINE_2_LEFT[1])

# ---------------------------------------------------------------------------
# PERSPECTIVE-CORRECTED DISTANCE  (derived only from the green lines)
# ---------------------------------------------------------------------------
#
# Camera model (flat road, pinhole camera at height H):
#     Road width at pixel row y :  L(y) = W·(y − y_v) / H
#     Distance along road       :  Z(y) = H·f / (y − y_v)
#
# Where y_v = vanishing-point Y and f = focal length (px).
#
# Given L1, L2  (pixel widths of the two 25 m lines) and y1, y2:
#     y_v = (L1·y2 − L2·y1) / (L1 − L2)
#     D   = W·(y2 − y1) / [ L1·(y2 − y_v) ]      (metres)
# ---------------------------------------------------------------------------

_L1 = float(np.linalg.norm(LINE_1_RIGHT - LINE_1_LEFT))   # pixel width line 1
_L2 = float(np.linalg.norm(LINE_2_RIGHT - LINE_2_LEFT))   # pixel width line 2
_y1 = float(LINE_1_LEFT[1])
_y2 = float(LINE_2_LEFT[1])

# Vanishing-point Y (where road edges converge)
_y_v = (_L1 * _y2 - _L2 * _y1) / (_L1 - _L2)

# Perspective-corrected real-world distance between the two green lines
TRIPWIRE_DISTANCE_M = GREEN_LINE_LENGTH_M * (_y2 - _y1) / (_L1 * (_y2 - _y_v))

# Naive distance (uniform scale from the average of the two lines)
NAIVE_DISTANCE_M = GREEN_LINE_LENGTH_M * (_y2 - _y1) / ((_L1 + _L2) / 2.0)

print(f"Vanishing-point Y            : {_y_v:.1f} px")
print(f"Pixel widths                 : L1 = {_L1:.0f} px, L2 = {_L2:.0f} px")
print(f"Perspective-corrected dist.  : {TRIPWIRE_DISTANCE_M:.2f} m")
print(f"Naive (avg-scale) dist.      : {NAIVE_DISTANCE_M:.2f} m")
print(f"Difference                   : {(TRIPWIRE_DISTANCE_M / NAIVE_DISTANCE_M - 1) * 100:+.1f} %")
print()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

class CentroidTracker:
    """
    Lightweight greedy centroid tracker.
    Assigns consistent IDs across frames using nearest-neighbour matching.
    """

    def __init__(self, max_lost: int = 8, max_distance: int = 120):
        self.next_id      = 0
        self.centroids    = {}               # id → (cx, cy)
        self.lost_count   = defaultdict(int)
        self.max_lost     = max_lost
        self.max_distance = max_distance

    def update(self, rects: list[tuple[int, int, int, int]]) -> dict:
        if not rects:
            for tid in list(self.centroids):
                self.lost_count[tid] += 1
                if self.lost_count[tid] > self.max_lost:
                    del self.centroids[tid]
                    del self.lost_count[tid]
            return dict(self.centroids)

        new_cents = [(int(x + w / 2), int(y + h / 2)) for (x, y, w, h) in rects]

        if not self.centroids:
            for nc in new_cents:
                self.centroids[self.next_id] = nc
                self.lost_count[self.next_id] = 0
                self.next_id += 1
            return dict(self.centroids)

        ex_ids  = list(self.centroids)
        ex_pts  = [self.centroids[i] for i in ex_ids]
        used_ex = set()
        used_nw = set()

        dist_mat = np.linalg.norm(
            np.array(ex_pts)[:, None, :] - np.array(new_cents)[None, :, :],
            axis=-1,
        )

        pairs = sorted(
            [(dist_mat[i, j], i, j)
             for i in range(len(ex_pts))
             for j in range(len(new_cents))],
            key=lambda t: t[0],
        )

        for d, ei, ni in pairs:
            if ei in used_ex or ni in used_nw:
                continue
            if d > self.max_distance:
                break
            tid = ex_ids[ei]
            self.centroids[tid]  = new_cents[ni]
            self.lost_count[tid] = 0
            used_ex.add(ei)
            used_nw.add(ni)

        for ei, tid in enumerate(ex_ids):
            if ei not in used_ex:
                self.lost_count[tid] += 1
                if self.lost_count[tid] > self.max_lost:
                    del self.centroids[tid]
                    del self.lost_count[tid]

        for ni, nc in enumerate(new_cents):
            if ni not in used_nw:
                self.centroids[self.next_id] = nc
                self.lost_count[self.next_id] = 0
                self.next_id += 1

        return dict(self.centroids)


# ---------------------------------------------------------------------------
# TRIPWIRE CROSSING LOGIC
# ---------------------------------------------------------------------------

class TripwireCrossing:
    """
    Detects when a tracked centroid crosses each horizontal tripwire line
    and computes the speed from the crossing-time difference.
    """

    def __init__(self, line_1_y: int, line_2_y: int,
                 corrected_dist_m: float, naive_dist_m: float, fps: float):
        self.line_1_y         = line_1_y
        self.line_2_y         = line_2_y
        self.corrected_dist_m = corrected_dist_m
        self.naive_dist_m     = naive_dist_m
        self.fps              = fps

        # Per-track state
        self.prev_y:       dict[int, int]   = {}   # tid → previous cy
        self.crossed_1:    dict[int, int]   = {}   # tid → frame when crossed line 1
        self.speed_corr:   dict[int, float] = {}   # tid → perspective-corrected speed
        self.speed_naive:  dict[int, float] = {}   # tid → naive speed

    def update(self, active: dict[int, tuple[int, int]], frame_idx: int):
        """
        Call once per frame with the active centroid dict.
        Populates self.speed_corr / self.speed_naive for vehicles that
        have crossed both lines.
        """
        for tid, (cx, cy) in active.items():
            prev = self.prev_y.get(tid)
            self.prev_y[tid] = cy

            if prev is None:
                continue

            # Detect downward crossing of Line 1
            if tid not in self.crossed_1:
                if prev <= self.line_1_y < cy or cy <= self.line_1_y < prev:
                    self.crossed_1[tid] = frame_idx

            # Detect downward crossing of Line 2 (only after crossing Line 1)
            elif tid not in self.speed_corr:
                if prev <= self.line_2_y < cy or cy <= self.line_2_y < prev:
                    elapsed_frames = frame_idx - self.crossed_1[tid]
                    if elapsed_frames > 0:
                        elapsed_s = elapsed_frames / self.fps
                        self.speed_corr[tid]  = (self.corrected_dist_m / elapsed_s) * 3.6
                        self.speed_naive[tid] = (self.naive_dist_m     / elapsed_s) * 3.6

    def get_speed(self, tid: int) -> float | None:
        """Return the perspective-corrected speed for display, or None."""
        return self.speed_corr.get(tid)


# ---------------------------------------------------------------------------
# DRAWING
# ---------------------------------------------------------------------------
PALETTE = [
    (255,  85,  85), ( 85, 255,  85), ( 85,  85, 255), (255, 200,   0),
    (  0, 220, 220), (220,   0, 220), (255, 160,  10), (160, 255,  10),
]


def id_color(tid: int) -> tuple[int, int, int]:
    return PALETTE[tid % len(PALETTE)]


def draw_detection(frame, rect, tid, speed_kmh):
    x, y, w, h = rect
    color = id_color(tid)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    label = f"#{tid}"
    label += f"  {int(speed_kmh)} km/h" if speed_kmh is not None else "  --"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    bg_y1 = max(y - th - 8, 0)
    cv2.rectangle(frame, (x, bg_y1), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, max(y - 4, th)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_tripwires(frame, l1_left, l1_right, l2_left, l2_right):
    """Draw the two green tripwire lines on the frame."""
    green = (0, 255, 0)
    cv2.line(frame, tuple(l1_left.astype(int)), tuple(l1_right.astype(int)), green, 2)
    cv2.line(frame, tuple(l2_left.astype(int)), tuple(l2_right.astype(int)), green, 2)

    # Labels
    cv2.putText(frame, "Line 1", (int(l1_right[0]) + 10, int(l1_right[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1, cv2.LINE_AA)
    cv2.putText(frame, "Line 2", (int(l2_right[0]) + 10, int(l2_right[1]) + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 1, cv2.LINE_AA)


def draw_hud(frame, vehicle_count: int, frame_idx: int, total: int,
             corrected_d: float, naive_d: float):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (420, 110), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Vehicles in frame: {vehicle_count}",
                (20, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2, cv2.LINE_AA)

    pct = frame_idx / total * 100 if total else 0
    cv2.putText(frame, f"Frame {frame_idx}/{total}  ({pct:.0f}%)",
                (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Corrected dist: {corrected_d:.1f}m  |  Naive: {naive_d:.1f}m",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 100), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def main():
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

    # -- Background subtractor ------------------------------------------------
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=150, varThreshold=40, detectShadows=True
    )

    k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT,    (15, 15))

    # -- Tracker & tripwire ---------------------------------------------------
    tracker  = CentroidTracker(max_lost=8, max_distance=100)
    tripwire = TripwireCrossing(
        LINE_1_Y, LINE_2_Y,
        TRIPWIRE_DISTANCE_M, NAIVE_DISTANCE_M,
        fps,
    )

    # Map tid → last known rect (for drawing)
    tid_to_rect: dict[int, tuple] = {}

    # -- Video writer ---------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # --- 1. Foreground mask ----------------------------------------------
        fg = bg_sub.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close)
        fg = cv2.dilate(fg, k_dilate, iterations=1)

        # --- 2. Contours → bounding boxes ------------------------------------
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(c) for c in contours
                 if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

        # --- 3. Track ---------------------------------------------------------
        active = tracker.update(rects)

        # Rebuild tid → rect
        new_tid_to_rect: dict[int, tuple] = {}
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
                new_tid_to_rect[tid] = tid_to_rect[tid]
        tid_to_rect = new_tid_to_rect

        # --- 4. Tripwire crossing detection -----------------------------------
        tripwire.update(active, frame_idx)

        # --- 5. Annotate & write frame ----------------------------------------
        annotated = frame.copy()

        draw_tripwires(annotated, LINE_1_LEFT, LINE_1_RIGHT,
                       LINE_2_LEFT, LINE_2_RIGHT)

        for tid, (cx, cy) in active.items():
            rect = tid_to_rect.get(tid)
            if rect is None:
                continue
            draw_detection(annotated, rect, tid, tripwire.get_speed(tid))

        draw_hud(annotated, len(active), frame_idx, total,
                 TRIPWIRE_DISTANCE_M, NAIVE_DISTANCE_M)
        writer.write(annotated)

        if frame_idx % 50 == 0 or frame_idx == total:
            pct = frame_idx / total * 100 if total else 0
            print(f"  Frame {frame_idx}/{total}  ({pct:.0f}%)", end="\r")

    cap.release()
    writer.release()

    # --- 6. Write CSV --------------------------------------------------------
    with open(CSV_PATH, "w", newline="") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([
            "vehicle_id",
            "speed_corrected_kmh",
            "speed_naive_kmh",
            "difference_pct",
        ])
        for tid in sorted(tripwire.speed_corr):
            s_corr  = tripwire.speed_corr[tid]
            s_naive = tripwire.speed_naive.get(tid, 0.0)
            diff_pct = ((s_corr / s_naive) - 1) * 100 if s_naive else 0.0
            writer_csv.writerow([
                tid,
                f"{s_corr:.1f}",
                f"{s_naive:.1f}",
                f"{diff_pct:+.1f}",
            ])

    n_vehicles = len(tripwire.speed_corr)
    print(f"\nDone!")
    print(f"  Video : {OUTPUT_PATH}")
    print(f"  CSV   : {CSV_PATH}  ({n_vehicles} vehicles measured)")

    if n_vehicles > 0:
        avg_corr  = np.mean(list(tripwire.speed_corr.values()))
        avg_naive = np.mean(list(tripwire.speed_naive.values()))
        print(f"\n  Avg speed (perspective-corrected): {avg_corr:.1f} km/h")
        print(f"  Avg speed (naive / no correction): {avg_naive:.1f} km/h")
        print(f"  Difference:  {((avg_corr/avg_naive)-1)*100:+.1f} %")


if __name__ == "__main__":
    main()
