#!/bin/bash

set -e

echo "Starting preprocessing container..."

mkdir /data
INPUT_VIDEO="/data/road_traffic.mp4"

OUTPUT_DIR="/output"
mkdir -p "$OUTPUT_DIR"

echo "Downloading video from Blob Storage..."
curl -L "$INPUT_VIDEO_URL" -o "$INPUT_VIDEO"
echo "Video downloaded successfully."

echo "Running video preprocessing (split, upload per clip, Event Hub per clip)..."
export VIDEO_ID=road_traffic
python3 app.py "$INPUT_VIDEO" "$OUTPUT_DIR"
echo "Video preprocessing completed successfully."

echo "Ending preprocessing container..."