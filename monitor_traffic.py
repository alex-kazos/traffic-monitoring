"""
Traffic monitoring entry point: read inputs (Azure Blob or local), run pipeline per video, write CSVs.
"""
import os
import logging
import time

from config import (
    AZURE_STORAGE_CONTAINER_NAME,
    BLOB_VIDEO_FOLDER,
    VIDEO_PATH,
    CSV_PATH,
    OUTPUT_DIR,
)
from blob_input import _get_blob_client, get_video_inputs
from pipeline import run_pipeline


def main() -> None:
    use_azure_input = _get_blob_client() is not None
    if use_azure_input:
        logging.info(
            "Input: Azure Blob (container=%s, folder=%s)",
            AZURE_STORAGE_CONTAINER_NAME,
            BLOB_VIDEO_FOLDER,
        )
    else:
        logging.info("Input: local file %s", VIDEO_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info("Output directory (CSV only): %s", os.path.abspath(OUTPUT_DIR))

    count = 0
    ## for each video of the blob storage run the pipeline!
    for video_path, stem in get_video_inputs():
        count += 1
        logging.info("--- Video %s ---", stem)
        csv_name = f"vehicle_speeds_{stem}.csv" if use_azure_input else CSV_PATH
        csv_path = os.path.join(OUTPUT_DIR, os.path.basename(csv_name))
        try:
            run_pipeline(video_path, csv_path)
        finally:
            if use_azure_input:
                try:
                    os.remove(video_path)
                    logging.debug("Removed temp file %s", video_path)
                except OSError:
                    pass

    logging.info("Finished processing %d video(s)", count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress Azure SDK request/response dumps (they log at DEBUG)
    logging.getLogger("azure").setLevel(logging.WARNING)
    t0 = time.time()
    main()
    logging.info("Total time: %.1f s", time.time() - t0)
