"""
Azure Blob Storage input: list and download .mp4 videos from a container folder.
"""
import os
import logging
import tempfile

from config import (
    AZURE_STORAGE_CONNECTION_STRING,
    AZURE_STORAGE_CONTAINER_NAME,
    BLOB_VIDEO_FOLDER,
)

## We call this each time we need to access the Azure Blob Storage
def _get_blob_client():
    """Return container client if Azure is configured, else None."""
    if not AZURE_STORAGE_CONNECTION_STRING or not AZURE_STORAGE_CONTAINER_NAME:
        return None
    try:
        from azure.storage.blob import BlobServiceClient  # type: ignore[import-untyped]
        ## get the container client
        container = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        ).get_container_client(AZURE_STORAGE_CONTAINER_NAME)
        return container
    except Exception:
        return None

## Just a list of the .mp4 files in the Azure Blob Storage
def list_video_blobs():
    """List .mp4 blob names under BLOB_VIDEO_FOLDER. Returns list of paths, or [] if not Azure."""
    container = _get_blob_client()
    if container is None:
        return []
    prefix = f"{BLOB_VIDEO_FOLDER}/"
    ## just a cleaning step to get only the .mp4 files
    names = [b.name for b in container.list_blobs(name_starts_with=prefix) if b.name.lower().endswith(".mp4")]
    for path in names:
        logging.info(path)
    return names


## Download the file to a temporary file
def download_blob_to_temp(blob_name: str) -> str:
    """
    Download a blob to a temporary file and return its path.
    Caller is responsible for deleting the file when done.
    """
    container = _get_blob_client()
    if container is None:
        raise RuntimeError("Azure Blob Storage is not configured")
    # The file we want to access
    blob_client = container.get_blob_client(blob_name)
    logging.info("Downloading blob: %s ...", blob_name)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    with open(path, "wb") as f:
        blob_client.download_blob().readinto(f)
    logging.info("Downloaded blob: %s -> %s", blob_name, path)
    # Return the path of the downloaded file
    return path


def get_video_inputs():
    """Yield (video_path, stem) for each .mp4 in Azure. Raises if Azure is not configured."""
    container = _get_blob_client()
    if container is None:
        raise RuntimeError(
            "Azure Blob Storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER_NAME."
        )
    # list the files in the container
    blob_names = list_video_blobs()
    if not blob_names:
        raise FileNotFoundError(f"No .mp4 in container '{AZURE_STORAGE_CONTAINER_NAME}' under '{BLOB_VIDEO_FOLDER}/'")
        # for each file
    for blob_name in blob_names:
        # download the file
        path = download_blob_to_temp(blob_name)
        # get the filename without the extension
        stem = os.path.splitext(os.path.basename(blob_name))[0]
        yield path, stem  ## smart way to iterate over the videos one at the time!
        ## the yield keyword is used to return a generator object that can be iterated over and not stored in memory!
