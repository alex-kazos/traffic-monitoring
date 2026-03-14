"""
Traffic speed monitoring algorithm: detection, tracking, tripwire speed, BEV fallback, CSV export.
"""
import os
import csv
import logging
import numpy as np
import cv2  # type: ignore[import-untyped]
from collections import defaultdict, deque
import pyodbc
import re

from config import (
    FPS,
    MIN_AREA,
    TRUCK_MIN_AREA,
    MAX_LOST,
    MAX_DIST,
    BEV_WINDOW,
    BEV_MIN_FRAMES,
    BEV_SPD_MIN,
    BEV_SPD_MAX,
    SPLIT_X,
    LINE_1_Y,
    LINE_2_Y,
    PERSPECT_L,
    PERSPECT_R,
    connection_string
)


def bev_calculation(pix_x: int, pix_y: int, perspective_array) -> float:
    """
    Bird's Eye View Y-coordinate in real-world meters.
    """
    input_point = np.array([[[float(pix_x), float(pix_y)]]], dtype=np.float32)
    bev = float(cv2.perspectiveTransform(input_point, perspective_array)[0, 0, 1])
    return bev


# Real-world distance between tripwire lines (meters)
DIST_L = abs(bev_calculation(320, LINE_1_Y, PERSPECT_L) - bev_calculation(320, LINE_2_Y, PERSPECT_L))
DIST_R = abs(bev_calculation(960, LINE_1_Y, PERSPECT_R) - bev_calculation(960, LINE_2_Y, PERSPECT_R))


def initialize_tracker() -> dict:
    """New tracker state: next_id, centroids, lost counts."""
    return {"next_id": 0, "centroids": {}, "lost": defaultdict(int)}


def update_tracker(tracker: dict, bounding_box: list) -> dict:
    """
    Match detections to tracks by distance; mark lost, remove if MAX_LOST exceeded.
    Returns active centroids dict (track_id -> (x, y)).
    """
    centroids = tracker["centroids"]
    lost = tracker["lost"]
    new_points = [(int(x + w / 2), int(y + h)) for x, y, w, h in bounding_box]

    if not bounding_box:
        for track_id in list(centroids):
            lost[track_id] += 1
            if lost[track_id] > MAX_LOST:
                del centroids[track_id]
                del lost[track_id]
        return dict(centroids)

    if not centroids:
        for point in new_points:
            centroids[tracker["next_id"]] = point
            lost[tracker["next_id"]] = 0
            tracker["next_id"] += 1
        return dict(centroids)

    existing_ids = list(centroids)
    existing_points = [centroids[i] for i in existing_ids]
    dist = np.linalg.norm(
        np.array(existing_points)[:, None] - np.array(new_points)[None, :], axis=-1
    )
    pairs = sorted(
        [(dist[i, j], i, j) for i in range(len(existing_points)) for j in range(len(new_points))],
        key=lambda x: x[0],
    )
    existing_indices = set()
    new_indices = set()

    for distance, existing_index, new_index in pairs:
        if existing_index in existing_indices or new_index in new_indices:
            continue
        if distance > MAX_DIST:
            break
        track_id = existing_ids[existing_index]
        centroids[track_id] = new_points[new_index]
        lost[track_id] = 0
        existing_indices.add(existing_index)
        new_indices.add(new_index)

    for existing_index, track_id in enumerate(existing_ids):
        if existing_index not in existing_indices:
            lost[track_id] += 1
            if lost[track_id] > MAX_LOST:
                del centroids[track_id]
                del lost[track_id]

    for new_index, point in enumerate(new_points):
        if new_index not in new_indices:
            centroids[tracker["next_id"]] = point
            lost[tracker["next_id"]] = 0
            tracker["next_id"] += 1

    return dict(centroids)


def initialize_tripwire(first_line: int, second_line: int, dist_m: float) -> dict:
    """Tripwire state: prev_y, crossed_first, speed, line y-coords, dist_m."""
    return {
        "prev_y": {},
        "crossed_first": {},
        "speed": {},
        "line_first_y": first_line,
        "line_second_y": second_line,
        "dist_m": dist_m,
    }


def crossed_line_check(prev_y: float, cur_y: float, line_y: int) -> bool:
    """True if the line was crossed between previous and current frame."""
    return (prev_y < line_y <= cur_y) or (cur_y < line_y <= prev_y)


def update_tripwire(tripwire: dict, tracked_vehicles: dict, current_frame: int) -> None:
    """Update tripwire state and compute speed when vehicle crosses both lines."""
    for track_id, (_, current_position_y) in tracked_vehicles.items():
        prev = tripwire["prev_y"].get(track_id)
        tripwire["prev_y"][track_id] = current_position_y
        if prev is None:
            continue
        if track_id not in tripwire["crossed_first"]:
            if crossed_line_check(prev, current_position_y, tripwire["line_first_y"]):
                tripwire["crossed_first"][track_id] = current_frame
        elif track_id not in tripwire["speed"]:
            if crossed_line_check(prev, current_position_y, tripwire["line_second_y"]):
                elapsed = (current_frame - tripwire["crossed_first"][track_id]) / FPS
                if elapsed > 0:
                    tripwire["speed"][track_id] = (tripwire["dist_m"] / elapsed) * 3.6


def run_pipeline(video_path: str, csv_path: str) -> None:
    """
    Run the traffic speed pipeline on one video: detect, track, compute speeds, write CSV.
    """
    logging.info("Processing video: %s -> %s", video_path, csv_path)
    capture_video = cv2.VideoCapture(video_path)
    if not capture_video.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    frame_width = int(capture_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture_video.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info("Video %s: %dx%d, %d frames", video_path, frame_width, frame_height, total_frames)

    # Extract segment ID
    part_match = re.search(r"_part_(\d+)", os.path.basename(video_path))
    segment_id = f"{int(part_match.group(1)):03d}" if part_match else "000"
    segment_id = int(segment_id)

    background = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40, detectShadows=True)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))

    tracker_left = initialize_tracker()
    tracker_right = initialize_tracker()
    tripwire_left = initialize_tripwire(LINE_2_Y, LINE_1_Y, DIST_L)
    tripwire_right = initialize_tripwire(LINE_1_Y, LINE_2_Y, DIST_R)

    vehicle_type = {}
    bev_hist = defaultdict(lambda: deque(maxlen=BEV_WINDOW))
    bev_speed = {}
    entry_frame = {}

    for frame_id in range(1, total_frames + 1):
        result, frame = capture_video.read()
        if not result:
            break

        foreground = background.apply(frame)
        _, foreground = cv2.threshold(foreground, 200, 255, cv2.THRESH_BINARY)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_small)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel_medium)
        foreground = cv2.dilate(foreground, kernel_large)

        centroids, _ = cv2.findContours(foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_bounding_rect = [cv2.boundingRect(c) for c in centroids if cv2.contourArea(c) >= MIN_AREA]
        rects_left = [r for r in all_bounding_rect if (r[0] + r[2] // 2) < SPLIT_X]
        rects_right = [r for r in all_bounding_rect if (r[0] + r[2] // 2) >= SPLIT_X]

        active_left = update_tracker(tracker_left, rects_left)
        active_right = update_tracker(tracker_right, rects_right)

        for tracker_id in active_left:
            key = ("L", tracker_id)
            if key not in entry_frame:
                entry_frame[key] = frame_id
        for tracker_id in active_right:
            key = ("R", tracker_id)
            if key not in entry_frame:
                entry_frame[key] = frame_id

        update_tripwire(tripwire_left, active_left, frame_id)
        update_tripwire(tripwire_right, active_right, frame_id)

        for side, active, matrix_side, rects_side in [
            ("L", active_left, PERSPECT_L, rects_left),
            ("R", active_right, PERSPECT_R, rects_right),
        ]:
            for tracker_id, (current_position_x, current_position_y) in active.items():
                key = (side, tracker_id)
                bev_perspective = cv2.perspectiveTransform(
                    np.array([[[float(current_position_x), float(current_position_y)]]], dtype=np.float32),
                    matrix_side,
                )[0, 0]
                bev_hist[key].append((float(bev_perspective[0]), float(bev_perspective[1])))
                h_pts = list(bev_hist[key])
                if len(h_pts) >= BEV_MIN_FRAMES:
                    diffs = [
                        ((h_pts[i + 1][0] - h_pts[i][0]) ** 2 + (h_pts[i + 1][1] - h_pts[i][1]) ** 2) ** 0.5
                        for i in range(len(h_pts) - 1)
                    ]
                    tracking_speed = float(np.median(diffs)) * FPS * 3.6
                    if BEV_SPD_MIN < tracking_speed < BEV_SPD_MAX:
                        bev_speed[key] = tracking_speed
                if rects_side:
                    rect = min(
                        rects_side,
                        key=lambda r: abs(current_position_x - (r[0] + r[2] // 2))
                        + abs(current_position_y - (r[1] + r[3])),
                    )
                    vehicle_type[key] = "truck" if rect[2] * rect[3] >= TRUCK_MIN_AREA else "car"

    capture_video.release()

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    ## Comment: added null to speed_source, if speed is nullable, speed_source should be nullable too.
    cursor.execute("""
    IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'vehicle_speeds')
    BEGIN
        CREATE TABLE vehicle_speeds (
            id INT IDENTITY(1,1) PRIMARY KEY,
            vehicle_id VARCHAR(100) NOT NULL,
            carriageway VARCHAR(200) NOT NULL,
            vehicle_type VARCHAR(200) NOT NULL,
            speed_kmh FLOAT NULL,
            speed_source VARCHAR(200) NULL,
            entry_frame INT NOT NULL,
            total_frames INT NOT NULL,
            segment_id INT NOT NULL
        );

            CREATE TABLE vehicle_speeds_alerts (
            id INT IDENTITY(1,1) PRIMARY KEY,
            vehicle_id VARCHAR(100) NOT NULL,
            carriageway VARCHAR(200) NOT NULL,
            vehicle_type VARCHAR(200) NOT NULL,
            speed_kmh FLOAT NULL,
            speed_source VARCHAR(200) NULL,
            entry_frame INT NOT NULL,
            total_frames INT NOT NULL,
            segment_id INT NOT NULL
        );
    END
    """)
    conn.commit()

    # column names: "vehicle_id", "carriageway", "vehicle_type", "speed_kmh", "speed_source"
    for side, tripwire, side_label in [
        ("L", tripwire_left, "left"),
        ("R", tripwire_right, "right"),
    ]:
        seen_track_ids = (
            set(tripwire["speed"])
            | {tracker_id for (s, tracker_id) in vehicle_type if s == side}
            | {tracker_id for (s, tracker_id) in bev_speed if s == side}
        )
        for tracker_id in sorted(seen_track_ids):
            key = (side, tracker_id)
            speed_tripwire = tripwire["speed"].get(tracker_id)
            speed_bev = bev_speed.get(key)
            if speed_tripwire is not None:
                tracking_speed, tracking_source = speed_tripwire, "tripwire"
            elif speed_bev is not None:
                tracking_speed, tracking_source = speed_bev, "bev_avg"
            else:
                tracking_speed, tracking_source = None, "none"
            entry_frame_val = entry_frame.get(key, 0)

            vehicle_id = f"{side}{tracker_id}"

            # Insert data to the database
            cursor.execute(
                """
                INSERT INTO vehicle_speeds (vehicle_id, carriageway, vehicle_type, speed_kmh, speed_source, entry_frame, total_frames, segment_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vehicle_id,
                    side_label,
                    vehicle_type.get(key, "unknown"),
                    float(tracking_speed) if tracking_speed is not None else None,
                    tracking_source,
                    entry_frame_val,
                    total_frames,
                    segment_id,
                ),
            )
            # Insert data to the alerts table
            if tracking_speed is not None:
                if tracking_speed > 130:
                    cursor.execute(
                        """
                        INSERT INTO vehicle_speeds_alerts (vehicle_id, carriageway, vehicle_type, speed_kmh, speed_source, entry_frame, total_frames, segment_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            vehicle_id,
                            side_label,
                            vehicle_type.get(key, "unknown"),
                            float(tracking_speed),
                            tracking_source,
                            entry_frame_val,
                            total_frames,
                            segment_id,
                        ),
                    )
    conn.commit()
    logging.info("Inserted vehicle speeds to database")
    cursor.close()
    conn.close() ## yes this will run as many times as the number of videos in the blob storage. It can be optimized to run only once.
    logging.info("Closed database connection")