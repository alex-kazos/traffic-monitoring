import os

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

# --- CONFIGURATION ---
VIDEO_NAME = "Road traffic video for object recognition_part_1.mp4"
OUTPUT_PATH = "traffic_speed_output.mp4"

# Note: These coordinates are from the Roboflow tutorial.
# They must be recalibrated for your specific video's perspective!
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET = np.array([
    [0, 0],
    [24, 0],
    [24, 249],
    [0, 249],
])


class ViewTransformer:
    """Handles the perspective transformation from camera pixels to real-world flat coordinates."""

    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # Calculate the perspective transformation matrix
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


def main():
    print("Loading YOLO model...")
    # This will automatically download the YOLOv8x model on the first run
    model = YOLO("yolov8x.pt")

    # Initialize the view transformer for our specific road perspective
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    video_path = os.path.join("Downloads","Segments", VIDEO_NAME)
    # Get video information (resolution, fps, etc.)
    video_info = sv.VideoInfo.from_video_path(video_path)

    # Initialize the video frame generator
    frame_generator = sv.get_video_frames_generator(video_path)

    # Initialize BYTETrack for tracking vehicles across multiple frames
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    # Annotators to draw boxes and speed labels on the output video
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)

    # Dictionary to store the history of y-coordinates for each tracked vehicle
    # We store 1 second worth of data (maxlen = video fps) to smooth out micro-movements
    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    print(f"Processing video. Output will be saved to {OUTPUT_PATH}...")

    # Open a VideoSink to save the results
    with sv.VideoSink(OUTPUT_PATH, video_info=video_info) as sink:

        for frame in frame_generator:
            # 1. Run Object Detection
            result = model(frame, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Filter detections to only include vehicles (YOLO classes: 2=car, 3=motorcycle, 5=bus, 7=truck)
            detections = detections[np.isin(detections.class_id, [2, 3, 5, 7])]

            # 2. Update the tracker
            detections = byte_track.update_with_detections(detections=detections)

            # 3. Perspective Transformation
            # Get the bottom-center coordinates of the bounding boxes (where the car touches the road)
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            # Transform those pixel points into our real-world coordinate map
            transformed_points = view_transformer.transform_points(points=points).astype(int)

            labels = []

            # 4. Calculate Speed
            for tracker_id, [_, y] in zip(detections.tracker_id, transformed_points):
                # Add the new y-coordinate to the vehicle's history
                coordinates[tracker_id].append(y)

                # Wait until we have at least half a second of tracking data to calculate speed
                if len(coordinates[tracker_id]) > video_info.fps / 2:

                    # Calculate distance traveled based on our transformed coordinates
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)

                    # Calculate time elapsed based on the number of frames stored
                    time = len(coordinates[tracker_id]) / video_info.fps

                    # Calculate speed (distance / time) and convert to km/h (* 3.6)
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")
                else:
                    labels.append(f"#{tracker_id}")

            # 5. Annotate the frame
            annotated_frame = frame.copy()
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            # Write the annotated frame to the output video
            sink.write_frame(frame=annotated_frame)

    print("Video processing complete!")


if __name__ == "__main__":
    main()