# """
# Traffic monitoring entry point: read inputs (Azure Blob or local), run pipeline per video, write CSVs.
# """
# import os
# import logging
# import time

# from config import (
#     AZURE_STORAGE_CONTAINER_NAME,
#     BLOB_VIDEO_FOLDER,
#     VIDEO_PATH,
#     CSV_PATH,
#     OUTPUT_DIR,
# )
# from blob_input import _get_blob_client, get_video_inputs
# from pipeline import run_pipeline


# def main() -> None:
#     use_azure_input = _get_blob_client() is not None
#     if use_azure_input:
#         logging.info(
#             "Input: Azure Blob (container=%s, folder=%s)",
#             AZURE_STORAGE_CONTAINER_NAME,
#             BLOB_VIDEO_FOLDER,
#         )
#     else:
#         logging.info("Input: local file %s", VIDEO_PATH)

#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     logging.info("Output directory (CSV only): %s", os.path.abspath(OUTPUT_DIR))

#     count = 0
#     ## for each video of the blob storage run the pipeline!
#     for video_path, stem in get_video_inputs():
#         count += 1
#         logging.info("--- Video %s ---", stem)
#         csv_name = f"vehicle_speeds_{stem}.csv" if use_azure_input else CSV_PATH
#         csv_path = os.path.join(OUTPUT_DIR, os.path.basename(csv_name))
#         try:
#             run_pipeline(video_path, csv_path)
#         finally:
#             if use_azure_input:
#                 try:
#                     os.remove(video_path)
#                     logging.debug("Removed temp file %s", video_path)
#                 except OSError:
#                     pass

#     logging.info("Finished processing %d video(s)", count)


# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s [%(levelname)s] %(message)s",
#         datefmt="%Y-%m-%d %H:%M:%S",
#     )
#     # Suppress Azure SDK request/response dumps (they log at DEBUG)
#     logging.getLogger("azure").setLevel(logging.WARNING)
#     t0 = time.time()
#     main()
#     logging.info("Total time: %.1f s", time.time() - t0)


"""
Traffic monitoring entry point: Event-driven architecture.
Reads events from Azure Event Hub, downloads videos from Blob Storage, 
runs pipeline per video, and writes CSVs.
"""
import os
import logging
import time

from azure.eventhub import EventHubConsumerClient
from azure.eventhub.extensions.checkpointstoreblob import BlobCheckpointStore

from config import (
    OUTPUT_DIR,
    EVENT_HUB_CONNECTION_STR,
    EVENT_HUB_NAME,
    CONSUMER_GROUP,
    CHECKPOINT_STORAGE_CONN_STR,
    CHECKPOINT_CONTAINER_NAME,
)
from blob_input import download_blob_to_temp  # Κρατάμε μόνο αυτή! Η list_blobs δεν χρειάζεται πια.
from pipeline import run_pipeline


def on_event(partition_context, event):
    """
    Αυτή η συνάρτηση καλείται αυτόματα κάθε φορά που το Event Hub στέλνει ένα νέο μήνυμα.
    """
    try:
        # # 1. Παίρνουμε το όνομα του αρχείου (blob name)
        # blob_name = event.body_as_str(encoding='UTF-8').strip()
        # if not blob_name:
        #     return
        
         # 1. Παίρνουμε το όνομα του αρχείου (blob name)
        raw_blob_name = event.body_as_str(encoding='UTF-8').strip()
        if not raw_blob_name:
            return
        
        # ΠΡΟΣΘΗΚΗ: Βρίσκουμε τον φάκελο και φτιάχνουμε το πλήρες path (π.χ. road_traffic/road_traffic_part_001.mp4)
        folder_name = raw_blob_name.split('_part_')[0]
        full_blob_path = f"{folder_name}/{raw_blob_name}"

        logging.info("--- Νέο Event στο Partition %s για το βίντεο: %s ---", partition_context.partition_id, raw_blob_name)

        # 2. Φτιάχνουμε τα μονοπάτια για το CSV (Όπως το είχες!)
        stem = os.path.splitext(os.path.basename(raw_blob_name))[0]
        csv_name = f"vehicle_speeds_{stem}.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)

        # 3. Κατεβάζουμε το βίντεο τοπικά
        local_video_path = download_blob_to_temp(full_blob_path)

        # 4. Τρέχουμε την επεξεργασία (Pipeline)
        try:
            logging.info("Εκτέλεση pipeline... Εξαγωγή στο: %s", csv_path)
            run_pipeline(local_video_path, csv_path)
        finally:
            # 5. Καθαρισμός του προσωρινού .mp4 αρχείου
            if os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                    logging.debug("Removed temp file %s", local_video_path)
                except OSError:
                    pass

        # 6. Ενημερώνουμε το Checkpoint για να μην ξαναδιαβάσουμε το ίδιο event
        partition_context.update_checkpoint(event)
        logging.info("Η επεξεργασία του %s ολοκληρώθηκε επιτυχώς.", stem)

    except Exception as e:
        logging.error("Σφάλμα κατά την επεξεργασία του event: %s", e)


def main() -> None:
    # Δημιουργία του φακέλου εξόδου για τα CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info("Output directory (CSV only): %s", os.path.abspath(OUTPUT_DIR))

    # Αρχικοποίηση του Checkpoint Store
    if not CHECKPOINT_STORAGE_CONN_STR:
        raise ValueError("Δεν βρέθηκε το CHECKPOINT_STORAGE_CONN_STR. Είναι απαραίτητο για το Checkpointing.")
        
    checkpoint_store = BlobCheckpointStore.from_connection_string(
        CHECKPOINT_STORAGE_CONN_STR, 
        CHECKPOINT_CONTAINER_NAME
    )

    # Αρχικοποίηση του Event Hub Consumer
    if not EVENT_HUB_CONNECTION_STR:
        raise ValueError("Δεν βρέθηκε το EVENT_HUB_CONNECTION_STR.")

    client = EventHubConsumerClient.from_connection_string(
        EVENT_HUB_CONNECTION_STR,
        consumer_group=CONSUMER_GROUP,
        eventhub_name=EVENT_HUB_NAME,
        checkpoint_store=checkpoint_store,
    )

    logging.info("Ο Worker ξεκίνησε. Σύνδεση στο Event Hub και αναμονή για events...")
    
    # Ξεκινάει το "άπειρο" loop που ακούει για events
    with client:
        # Το starting_position="-1" διαβάζει τα νέα events ή συνεχίζει από το τελευταίο checkpoint
        client.receive(on_event=on_event, starting_position="-1")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Suppress Azure SDK request/response dumps
    logging.getLogger("azure").setLevel(logging.WARNING)
    logging.getLogger("uamqp").setLevel(logging.WARNING)  # Συχνά το event hub πετάει logs από εδώ
    
    main()
    