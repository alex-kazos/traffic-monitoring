#!/bin/bash
# This script runs when the Docker container starts.
# It: 1) Downloads a video from Azure Blob, 2) Splits it with app.py, 3) Uploads the segments and sends an event for each.

set -e
# "set -e" means: if any command fails, stop the script (don't keep going).

echo "Starting preprocessing container..."

# --- Step 1: Prepare folders and download the video from Azure Blob Storage ---
mkdir /data
INPUT_VIDEO="/data/road_traffic.mp4"
OUTPUT_DIR="/output"
mkdir -p "$OUTPUT_DIR"

echo "Downloading video from Blob Storage..."
# INPUT_VIDEO_URL is set when the container runs (e.g. by Azure). We download that URL into a local file.
curl -L "$INPUT_VIDEO_URL" -o "$INPUT_VIDEO"
echo "Video downloaded successfully."

# --- Step 2: Split the video into smaller segments using our Python app ---
echo "Running video preprocessing..."
python3 app.py "$INPUT_VIDEO" "$OUTPUT_DIR"
echo "Video preprocessing completed successfully."

# --- Step 3: Upload each segment to Blob Storage and send an Event Hubs message for each ---
# We need the Blob URL without the "?token=..." part for the path, and the token separately for auth.
TOKEN=$(echo "${OUTPUT_CONTAINER_URL}" | awk -F'?' '{print $2}')
OUTPUT_CONTAINER_URL=$(echo "${OUTPUT_CONTAINER_URL}" | awk -F'?' '{print $1}')
VIDEO_ID=road_traffic

# Loop over every file in the output folder (each video segment + manifest.json).
for file in "$OUTPUT_DIR"/*; do
    filename=$(basename "$file")
    echo "Uploading $filename to Blob Storage..."

    # Upload this file to Azure Blob Storage (PUT request with the file as body).
    curl --fail -X PUT \
         -H "x-ms-blob-type: BlockBlob" \
         -H "Content-Type: video/mp4" \
         -T "$file" \
         "$OUTPUT_CONTAINER_URL/$VIDEO_ID/$filename?${TOKEN}"

    # Only for .mp4 files: print "uploaded filename" and send that message to Event Hubs (skip for manifest.json etc.).
    if [[ "$filename" == *.mp4 ]]; then
        echo "uploaded sending event for s${filename}"
        python3 /app/send_event.py "${filename}"
        # If send_event.py fails (e.g. Event Hubs unreachable), the script stops because of set -e.
    fi
done

echo "Ending preprocessing container..."