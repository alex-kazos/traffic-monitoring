import sys
import os
import ffmpeg

def split_video(input_path, segment_time=120):
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    file_name = os.path.splitext(os.path.basename(input_path))[0]
    output_pattern = f"{file_name}_part_%03d.mp4"

    print(f"Splitting into chunks of {segment_time} seconds..")

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
        print("Splliting completed successfully")
        
    except ffmpeg.Error as e:
        print("FFmpeg Error:")
        print(e.stderr.decode())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python app.py <input_video>")
        sys.exit(1)

    video_file = sys.argv[1]
    split_video(video_file)