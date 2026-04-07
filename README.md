# Traffic Monitoring - Video Preprocessing

This repository currently contains a lightweight preprocessing pipeline for traffic videos:

- split a source video into fixed-duration clips,
- generate a `manifest.json` for produced clips,
- optionally send one Azure Event Hubs message per uploaded clip.

## What is in this repo

- `app.py` - CLI to split a video into segments with FFmpeg and create `manifest.json`.
- `send_event.py` - sends a single message to Azure Event Hubs.
- `entrypoint.sh` - container startup script that downloads input video, runs splitting, uploads outputs, and emits events.
- `Dockerfile` - container image for the preprocessing workflow.
- `config_local.py` - local constants and environment-backed settings used by related local/debug workflows.
- `debug_queries.sql` - SQL snippets for debugging/inspection.

## Requirements

From `requirements.txt`:

- `ffmpeg-python==0.2.0`
- `future==1.0.0`
- `azure-eventhub==5.12.0`

You also need the FFmpeg binary installed on your machine when running locally.

## Local setup

```powershell
Set-Location "C:\dev\aueb\digital-infrastructure\traffic-monitoring-project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Usage

### 1) Split a video into clips

```powershell
python app.py "Downloads\Road traffic video for object recognition.mp4" "Downloads\Segments" --segment_seconds 120
```

Output:

- segmented files like `..._part_000.mp4`,
- `manifest.json` in the output directory.

### 2) Send one Event Hubs message manually

Set environment variables first:

```powershell
$env:EVENTHUB_CONNECTION_STRING="<your-event-hub-connection-string>"
$env:EVENTHUB_NAME="<your-event-hub-name>"
python send_event.py "uploaded road_traffic_part_000.mp4"
```

## Docker workflow

The container flow in `entrypoint.sh` is:

1. Download video from `$INPUT_VIDEO_URL`.
2. Split video via `python3 app.py`.
3. Upload each output file to `$OUTPUT_CONTAINER_URL`.
4. For each `.mp4`, call `python3 /app/send_event.py "<filename>"`.

Build and run example:

```powershell
docker build -t traffic-preprocess .
docker run --rm \
  -e INPUT_VIDEO_URL="<blob-sas-url-to-source-video>" \
  -e OUTPUT_CONTAINER_URL="<blob-container-sas-url>" \
  -e EVENTHUB_CONNECTION_STRING="<eventhub-connection-string>" \
  -e EVENTHUB_NAME="<eventhub-name>" \
  traffic-preprocess
```

## Data folders (current workspace)

- `Downloads/` - source video and generated split parts.
- `Data/` - local debug SQLite and other data artifacts.
- `Tracking/` - produced tracking video/CSV artifacts.

## Important security note

`config_local.py` currently contains hardcoded credentials/secrets. Treat them as compromised and rotate them before using this project in any shared or production environment.

Recommended follow-up:

- move secrets to environment variables,
- keep only a `config_local.example.py` or `.env.example` with placeholders,
- add real secret files to `.gitignore`.
