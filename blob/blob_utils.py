import logging
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError, AzureError
from config import AZURE_BLOB_CONN_STR, AZURE_CONTAINER_NAME

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Azure Blob Config
# ---------------------------------------------------
try:
    if not AZURE_BLOB_CONN_STR:
        raise ValueError("AZURE_BLOB_CONN_STR is missing in environment variables.")
        
    blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
    container = blob_client.get_container_client(AZURE_CONTAINER_NAME)
except Exception as e:
    logger.critical(f"Failed to initialize Azure Blob Client: {e}")
    raise

# ---------------------------------------------------
# Upload Blob
# ---------------------------------------------------
def upload_blob(filename: str, data: bytes):
    """
    Upload file bytes to Azure Blob.
    """
    try:
        if not data:
            raise ValueError(f"Cannot upload empty data for file: {filename}")
            
        blob = container.get_blob_client(filename)
        blob.upload_blob(data, overwrite=True)
        return True
        
    except AzureError as e:
        logger.error(f"Azure Upload Failed for {filename}: {e}")
        raise RuntimeError(f"Azure Storage Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error uploading {filename}: {e}")
        raise

# ---------------------------------------------------
# Download Blob
# ---------------------------------------------------
def download_blob(filename: str) -> bytes:
    """Download blob file bytes."""
    try:
        blob = container.get_blob_client(filename)
        if not blob.exists():
            # triggers the FileNotFoundError block below
            raise FileNotFoundError(f"Blob not found in Azure: {filename}")
        return blob.download_blob().readall()

    except FileNotFoundError:
        logger.info(f"Blob requested but not found: {filename}")
        raise 
        
    except ResourceNotFoundError:
        logger.warning(f"Blob {filename} does not exist (ResourceNotFound).")
        raise FileNotFoundError(f"Blob not found: {filename}")
    except AzureError as e:
        logger.error(f"Azure Download Failed for {filename}: {e}")
        raise RuntimeError(f"Azure Download Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error downloading {filename}: {e}")
        raise

# ---------------------------------------------------
# Delete Blob
# ---------------------------------------------------
def delete_blob(filename: str):
    """Delete a blob file."""
    try:
        blob = container.get_blob_client(filename)
        if blob.exists():
            blob.delete_blob()
            return True
        else:
            logger.info(f"Blob {filename} already deleted or does not exist.")
            return False
            
    except AzureError as e:
        logger.error(f"Azure Delete Failed for {filename}: {e}")
        raise RuntimeError(f"Failed to delete blob: {e}")

# ---------------------------------------------------
# List Blobs
# ---------------------------------------------------
def list_blobs(prefix: str = None):
    """
    List all blobs inside the container.
    """
    try:
        if prefix:
            return [b.name for b in container.list_blobs(name_starts_with=prefix)]
        return [b.name for b in container.list_blobs()]
        
    except ResourceNotFoundError:
        logger.warning(f"Container '{AZURE_CONTAINER_NAME}' not found.")
        return []
    except AzureError as e:
        logger.error(f"Failed to list blobs: {e}")
        return []