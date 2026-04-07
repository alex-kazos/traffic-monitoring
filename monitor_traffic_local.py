"""
Traffic monitoring entry point.

- Local debug mode: simulate partition assignment using local segment files.
- Worker mode: receive Event Hub events and process blobs.
"""
import glob
import logging
import os

from config_local import (
    OUTPUT_DIR,
    EVENT_HUB_CONNECTION_STR,
    EVENT_HUB_NAME,
    CONSUMER_GROUP,
    CHECKPOINT_STORAGE_CONN_STR,
    CHECKPOINT_CONTAINER_NAME,
    LOCAL_DEBUG_MODE,
    DEBUG_PARTITIONS,
    DEBUG_TARGET_PARTITION,
    DEBUG_DISTRIBUTION_ONLY,
    DEBUG_LOCAL_SEGMENTS_GLOB,
)
from blob_input import download_blob_to_temp
from pipeline_local import run_pipeline


def resolve_blob_path(raw_blob_name: str) -> str:
    if "/" in raw_blob_name:
        return raw_blob_name
    folder_name = raw_blob_name.split("_part_")[0]
    return f"{folder_name}/{raw_blob_name}"


def process_event_blob(raw_blob_name: str, partition_id: int | None) -> None:
    full_blob_path = resolve_blob_path(raw_blob_name)
    stem = os.path.splitext(os.path.basename(raw_blob_name))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"vehicle_speeds_{stem}.csv")

    local_video_path = download_blob_to_temp(full_blob_path)
    try:
        run_pipeline(
            local_video_path,
            csv_path,
            source_name=raw_blob_name,
            debug_partition_id=partition_id,
        )
    finally:
        if os.path.exists(local_video_path):
            try:
                os.remove(local_video_path)
            except OSError:
                pass


def process_local_video(local_video_path: str, partition_id: int) -> None:
    stem = os.path.splitext(os.path.basename(local_video_path))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"vehicle_speeds_{stem}.csv")
    run_pipeline(
        local_video_path,
        csv_path,
        source_name=os.path.basename(local_video_path),
        debug_partition_id=partition_id,
    )


def run_local_debug() -> None:
    if DEBUG_PARTITIONS <= 0:
        raise ValueError("DEBUG_PARTITIONS must be > 0")

    segments = sorted(glob.glob(DEBUG_LOCAL_SEGMENTS_GLOB))
    if not segments:
        raise FileNotFoundError(f"No segment files found by glob: {DEBUG_LOCAL_SEGMENTS_GLOB}")

    assignments = {pid: [] for pid in range(DEBUG_PARTITIONS)}
    for idx, path in enumerate(segments):
        partition_id = idx % DEBUG_PARTITIONS
        assignments[partition_id].append(path)

    logging.info("Local partition simulation: files=%s partitions=%s", len(segments), DEBUG_PARTITIONS)
    for partition_id in range(DEBUG_PARTITIONS):
        logging.info("Partition %s has %s file(s)", partition_id, len(assignments[partition_id]))
        for path in assignments[partition_id]:
            logging.info("  %s -> partition %s", os.path.basename(path), partition_id)

    if DEBUG_DISTRIBUTION_ONLY:
        logging.info("DEBUG_DISTRIBUTION_ONLY is enabled; skipping pipeline execution")
        return

    target_partitions = range(DEBUG_PARTITIONS)
    if 0 <= DEBUG_TARGET_PARTITION < DEBUG_PARTITIONS:
        target_partitions = [DEBUG_TARGET_PARTITION]

    processed = 0
    for partition_id in target_partitions:
        for segment_path in assignments[partition_id]:
            process_local_video(segment_path, partition_id)
            processed += 1
    logging.info("Local debug finished: processed %s file(s)", processed)


def on_event(partition_context, event):
    raw_blob_name = event.body_as_str(encoding="UTF-8").strip()
    if not raw_blob_name:
        return

    try:
        partition_id = int(partition_context.partition_id)
    except (TypeError, ValueError):
        partition_id = None

    try:
        process_event_blob(raw_blob_name, partition_id)
        partition_context.update_checkpoint(event)
    except Exception as exc:
        logging.error("Error while processing event for blob=%s: %s", raw_blob_name, exc)


def run_worker() -> None:
    from azure.eventhub import EventHubConsumerClient
    from azure.eventhub.extensions.checkpointstoreblob import BlobCheckpointStore

    if not CHECKPOINT_STORAGE_CONN_STR:
        raise ValueError("CHECKPOINT_STORAGE_CONN_STR is required for checkpointing")
    if not EVENT_HUB_CONNECTION_STR:
        raise ValueError("EVENT_HUB_CONNECTION_STR is required")

    checkpoint_store = BlobCheckpointStore.from_connection_string(
        CHECKPOINT_STORAGE_CONN_STR,
        CHECKPOINT_CONTAINER_NAME,
    )
    client = EventHubConsumerClient.from_connection_string(
        EVENT_HUB_CONNECTION_STR,
        consumer_group=CONSUMER_GROUP,
        eventhub_name=EVENT_HUB_NAME,
        checkpoint_store=checkpoint_store,
    )
    logging.info("Worker started. Waiting for Event Hub events")
    with client:
        client.receive(on_event=on_event, starting_position="-1")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info("Output directory: %s", os.path.abspath(OUTPUT_DIR))

    if LOCAL_DEBUG_MODE:
        run_local_debug()
    else:
        run_worker()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("uamqp").setLevel(logging.WARNING)
    main()
