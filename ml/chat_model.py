from langchain_openai import AzureChatOpenAI
from functools import lru_cache
from config import (
    AZURE_CHAT_ENDPOINT,
    AZURE_CHAT_API_KEY,
    AZURE_CHAT_API_VERSION,
    AZURE_CHAT_MODEL
)

@lru_cache(maxsize=1)
def get_chat_model():
    """Return Azure ChatGPT model instance (Cached)."""
    return AzureChatOpenAI(
        api_key=AZURE_CHAT_API_KEY,
        azure_endpoint=AZURE_CHAT_ENDPOINT,
        api_version=AZURE_CHAT_API_VERSION,
        model=AZURE_CHAT_MODEL,
        temperature=0
    )