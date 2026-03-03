import os
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

# --- CONFIGURATION ---
VIDEO_NAME = "Road traffic video for object recognition_part_1.mp4"
OUTPUT_PATH = "traffic_speed_output.mp4"

SOURCE = np.array([
    [ 490, 295],
    [ 790, 295],
    [1260, 700],
    [  20, 700],
])

TARGET = np.array([
    [0, 0],
    [28, 0],
    [28, 102],
    [0, 102],
])

# --- Perspective Transformation Setup ---
SOURCE = SOURCE.astype(np.float32)
TARGET = TARGET.astype(np.float32)
PERSPECTIVE_MATRIX = cv2.getPerspectiveTransform(SOURCE, TARGET)


def transform_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
    transformed_points = cv2.perspectiveTransform(reshaped_points, PERSPECTIVE_MATRIX)
    return transformed_points.reshape(-1, 2)


def main():
    print("Loading YOLO model...")
    model = YOLO("yolov8x.pt")

    video_path = os.path.join("Downloads", "Segments", VIDEO_NAME)
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    print(f"Processing video. Output will be saved to {OUTPUT_PATH}...")

    with sv.VideoSink(OUTPUT_PATH, video_info=video_info) as sink:

        for frame in frame_generator:

            # 1. Object Detection
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Keep only vehicles
            detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]

            # 2. Tracking
            detections = byte_track.update_with_detections(detections=detections)

            # 3. Perspective Transform
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_points = transform_points(points).astype(int)

            labels = []

            # 4. Speed Calculation
            for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):

                coordinates[tracker_id].append(y)

                if len(coordinates[tracker_id]) > video_info.fps / 2:

                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)

                    time = len(coordinates[tracker_id]) / video_info.fps

                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")
                else:
                    labels.append(f"#{tracker_id}")

            # 5. Annotate Frame
            annotated_frame = frame.copy()
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            sink.write_frame(frame=annotated_frame)

    print("Video processing complete!")


if __name__ == "__main__":
    main()