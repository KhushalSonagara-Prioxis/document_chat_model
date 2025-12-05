import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, AzureError
from config import AZURE_BLOB_CONN_STR, AZURE_CONTAINER_NAME

logger = logging.getLogger(__name__)

try:
    if not AZURE_BLOB_CONN_STR:
        raise ValueError("AZURE_BLOB_CONN_STR is missing.")
        
    blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
    container = blob_client.get_container_client(AZURE_CONTAINER_NAME)
except Exception as e:
    logger.critical(f"Failed to initialize Azure Blob Client: {e}")
    raise

def upload_blob(filename: str, data: bytes):
    try:
        if not data:
            raise ValueError(f"Cannot upload empty data: {filename}")
        blob = container.get_blob_client(filename)
        blob.upload_blob(data, overwrite=True)
        return True
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}")
        raise

def download_blob(filename: str) -> bytes:
    try:
        blob = container.get_blob_client(filename)
        if not blob.exists():
            raise FileNotFoundError(f"Blob not found: {filename}")
        return blob.download_blob().readall()
    except FileNotFoundError:
        # Expected during first run or missing index
        logger.info(f"Blob not found (expected if fresh): {filename}")
        raise 
    except Exception as e:
        logger.error(f"Download failed for {filename}: {e}")
        raise

def delete_blob(filename: str):
    try:
        blob = container.get_blob_client(filename)
        if blob.exists():
            blob.delete_blob()
            return True
        return False
    except Exception as e:
        logger.error(f"Delete failed for {filename}: {e}")
        return False # Return False instead of crashing on delete

def list_blobs(prefix: str = None):
    try:
        if prefix:
            return [b.name for b in container.list_blobs(name_starts_with=prefix)]
        return [b.name for b in container.list_blobs()]
    except Exception as e:
        logger.error(f"List blobs failed: {e}")
        return []