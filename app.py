import os
import ffmpeg
import argparse
import json
import subprocess
import time

EVENTHUB_NAME = "video-clips"


def send_clip_event(clip):
    """Send a single Event Hub event when a new clip is created."""
    conn_str = os.environ.get("EVENTHUB_CONNECTION")
    if not conn_str:
        return
    try:
        from azure.eventhub import EventHubProducerClient, EventData

        producer = EventHubProducerClient.from_connection_string(
            conn_str=conn_str,
            eventhub_name=EVENTHUB_NAME,
        )

        event = EventData(json.dumps({"clip_name": clip}))
        producer.send_batch([event])
        producer.close()

        print(f"Event sent for clip: {clip}")

    except Exception as e:
        print(f"Event Hub send failed: {e}")


def create_manifest(video_id, output_dir, segment_time):
    clips = sorted(
        f for f in os.listdir(output_dir)
        if f.endswith('.mp4')
    )

    manifest = {
        "video_id": video_id,
        "segment_duration": segment_time,
        "clip_count": len(clips),
        "clips": [
            {
                "clip_id": os.path.splitext(clip)[0],
                "filename": clip,
            }
            for clip in clips
        ]
    }

    manifest_path = os.path.join(output_dir, "manifest.json")

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=4)

    print(f"Manifest created at: {manifest_path}")


def split_video(input_path, output_dir, segment_time=120):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' does not exist.")

    os.makedirs(output_dir, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_pattern = os.path.join(output_dir, f"{file_name}_part_%03d.mp4")

    print(f"Splitting into chunks of {segment_time} seconds...")

    try:

        # Run ffmpeg as a subprocess so we can monitor new files
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-f", "segment",
            "-segment_time", str(segment_time),
            "-reset_timestamps", "1",
            "-c:v", "copy",
            "-c:a", "copy",
            output_pattern
        ]

        process = subprocess.Popen(cmd)

        seen = set()

        # Monitor the output directory for new clips
        while process.poll() is None:
            for f in os.listdir(output_dir):
                if f.endswith(".mp4") and f not in seen:
                    seen.add(f)
                    print(f"New clip created: {f}")
                    send_clip_event(f)

            time.sleep(1)

        # Catch any final clips after ffmpeg exits
        for f in os.listdir(output_dir):
            if f.endswith(".mp4") and f not in seen:
                print(f"Final clip detected: {f}")
                send_clip_event(f)

        print("Splitting completed successfully")
        create_manifest(file_name, output_dir, segment_time)

    except Exception as e:
        print("FFmpeg Error:")
        print(e)
        raise


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split a video into smaller segments.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("output_dir", help="Directory to save the output segments.")
    parser.add_argument("--segment_seconds", type=int, default=120,
                        help="Duration of each segment in seconds (default: 120).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    split_video(args.input_video, args.output_dir, args.segment_seconds)