"""
Configuration: environment variables, paths, and algorithm constants.
"""
import os
import numpy as np
import cv2  # type: ignore[import-untyped]

# ----- Azure Blob Storage -----
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_STORAGE_CONTAINER_NAME = os.environ.get("AZURE_STORAGE_CONTAINER_NAME", "").strip()
BLOB_VIDEO_FOLDER = "road_traffic"

# ----- Local fallback -----
VIDEO_PATH = os.path.join(
    "../Downloads", "Segments",
    "Road traffic video for object recognition_part_1.mp4"
)
CSV_PATH = "vehicle_speeds.csv"

# ----- Output (e.g. /app/output in Docker for mounted volume) -----
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "").strip() or "."

# ----- Algorithm constants -----
FPS = 25.0
MIN_AREA = 1_500
TRUCK_MIN_AREA = 9_000
MAX_LOST = 12
MAX_DIST = 150

BEV_WINDOW = 8
BEV_MIN_FRAMES = 7
BEV_SPD_MIN = 5.0
BEV_SPD_MAX = 150.0  # UK motorway cap (km/h)

# Split detections into left vs right carriageway (bottom-center x)
SPLIT_X = 640

# Tripwire line y-coordinates
LINE_1_Y = 400  # upper
LINE_2_Y = 560  # lower

# Perspective calibration: source quadrilaterals (camera view)
SOURCE_L = np.array([
    [490, 295], [640, 295], [640, 700], [20, 700]
], dtype=np.float32)
SOURCE_R = np.array([
    [640, 295], [790, 295], [1260, 700], [640, 700]
], dtype=np.float32)

# Target rectangle (bird's-eye view)
TARGET_H = np.array([
    [0, 0], [14, 0], [14, 102], [0, 102]
], dtype=np.float32)

# Perspective transform matrices
PERSPECT_L = cv2.getPerspectiveTransform(SOURCE_L, TARGET_H)
PERSPECT_R = cv2.getPerspectiveTransform(SOURCE_R, TARGET_H)
