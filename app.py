import os
import ffmpeg
import argparse
import json

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
        (
            ffmpeg
            .input(input_path)
            .output(
                output_pattern,
                f='segment',
                segment_time=segment_time,
                reset_timestamps=1,
                vcodec='copy',
                acodec='copy'
            )
            .run()
        )

        print("Splitting completed successfully")
        create_manifest(file_name, output_dir, segment_time)

    except ffmpeg.Error as e:
        print("FFmpeg Error:")
        print(e.stderr.decode())
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