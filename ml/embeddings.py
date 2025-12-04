# ml/embeddings.py
from langchain_openai import AzureOpenAIEmbeddings
from config import AZURE_EMBEDDING_API_VERSION,AZURE_EMBEDDINGS_ENDPOINT,AZURE_EMBEDDINGS_API_KEY,AZURE_EMBEDDINGS_DEPLOYMENT

def get_embeddings():
    """Return Azure embedding model (text-embedding-3-small)."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_EMBEDDINGS_ENDPOINT,
        azure_deployment=AZURE_EMBEDDINGS_DEPLOYMENT,
        api_version=AZURE_EMBEDDING_API_VERSION,
        api_key=AZURE_EMBEDDINGS_API_KEY
    )
