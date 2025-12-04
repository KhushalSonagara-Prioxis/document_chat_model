from azure.storage.blob import BlobServiceClient
from config import AZURE_BLOB_CONN_STR, AZURE_CONTAINER_NAME

# ---------------------------------------------------
# Azure Blob Config
# ---------------------------------------------------
blob_client = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
container = blob_client.get_container_client(AZURE_CONTAINER_NAME)


# ---------------------------------------------------
# Upload Blob
# ---------------------------------------------------
def upload_blob(filename: str, data: bytes):
    """
    Upload file bytes to Azure Blob.
    filename should include the folder path (e.g., 'pdfs/file.pdf')
    """
    blob = container.get_blob_client(filename)
    blob.upload_blob(data, overwrite=True)
    return True


# ---------------------------------------------------
# Download Blob
# ---------------------------------------------------
def download_blob(filename: str) -> bytes:
    """Download blob file bytes."""
    blob = container.get_blob_client(filename)
    if not blob.exists():
        raise FileNotFoundError(f"Blob not found: {filename}")
    return blob.download_blob().readall()


# ---------------------------------------------------
# Delete Blob
# ---------------------------------------------------
def delete_blob(filename: str):
    """Delete a blob file."""
    blob = container.get_blob_client(filename)
    if blob.exists():
        blob.delete_blob()
    return True


# ---------------------------------------------------
# List Blobs
# ---------------------------------------------------
def list_blobs(prefix: str = None):
    """
    List all blobs inside the container.
    If prefix is provided (e.g., 'pdfs/'), only list files in that 'folder'.
    """
    if prefix:
        return [b.name for b in container.list_blobs(name_starts_with=prefix)]
    return [b.name for b in container.list_blobs()]