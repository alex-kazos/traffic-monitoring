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

echo "Running video preprocessing..."
python3 app.py "$INPUT_VIDEO" "$OUTPUT_DIR"
echo "Video preprocessing completed successfully."

TOKEN=$(echo "${OUTPUT_CONTAINER_URL}" | awk -F'?' '{print $2}')
OUTPUT_CONTAINER_URL=$(echo "${OUTPUT_CONTAINER_URL}" | awk -F'?' '{print $1}')
VIDEO_ID=road_traffic

for file in "$OUTPUT_DIR"/*; do
    filename=$(basename "$file")
    echo "Uploading $filename to Blob Storage..."

    curl --fail -X PUT \
         -H "x-ms-blob-type: BlockBlob" \
         -H "Content-Type: video/mp4" \
         -T "$file" \
         "$OUTPUT_CONTAINER_URL/$VIDEO_ID/$filename?${TOKEN}"
done

echo "Ending preprocessing container..."