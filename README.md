# Traffic Monitoring

This project provides robust tools for detecting, tracking, and calculating the speeds of vehicles from fixed traffic camera footage. It implements both traditional computer vision techniques and modern deep-learning-based Multi-Object Tracking (MOT) state-of-the-art approaches.

## Features

- **Traditional CV Methods (`monitor_traffic.py`)**: Employs Background Subtraction (MOG2) coupled with a custom centroid-based Euclidean distance tracker.
- **Deep Learning Methods (`monitor_traffic_supervision.py`)**: Utilizes the power of YOLOv8 for highly accurate object detection and classification, paired with ByteTrack for resilient tracking (ignoring momentary occlusions). Orchestrated using Roboflow's Supervision toolkit.
- **Speed Measurement**: Integrates both primary Tripwire-based speed measurements across defined meter gaps and continuous Bird's Eye View (BEV) median-rolling window derivations.
- **Perspective Transformation**: Mathematically corrects skewed 2D camera perspectives into a flattened top-down Bird's Eye View (Homography Matrix) allowing perfect frame-to-meter translation.
- **Data Exports**: Renders annotated bounding-box video outputs (`.mp4`) and detailed tracking reports (`.csv`) capturing granular speed telemetry for analytics.
- **Containerization**: Includes standalone Dockerfiles for modular processing pipelines (`Split` & `Tracking`).

## Installation & Setup

1. **Clone the repository.**
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   _(Major dependencies include `opencv-python`, `numpy`, and `ffmpeg`)_

To build the project's Docker containers for the `Split` and `Tracking` environments, execute the batch script:

```cmd
.\start_docker.bat
```

### Dockerhub

The docker image is at: https://hub.docker.com/r/giorgoskalesiakis/traffic-monitoring/tags

## Dashboard

The Plotly dashboard in `Dashboard/app.py` now loads its data directly from SQL Server. 
Before starting it, please make sure that you've your environment variables set up correctly to connect to the database.:
If the dashboard cannot connect, it will open with an on-screen status message instead.

## Usage

Simply execute the target monitoring script you wish to evaluate. It will load the defined video paths, compute telemetry on every frame, and push reports dynamically.

```bash
python monitor_traffic.py
# or
python monitor_traffic_supervision.py
```

### Generated Artifacts

Every successful run creates:

- **Video Output**: e.g., `traffic_speed_output.mp4` containing bounding boxes, ID tags, and real-time HUD metrics.
- **Data CSV**: e.g., `vehicle_speeds.csv` storing the final aggregated IDs, median km/h speeds, and timestamps.