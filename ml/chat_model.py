# ml/chat_model.py
from langchain_openai import AzureChatOpenAI
from config import (
    AZURE_CHAT_ENDPOINT,
    AZURE_CHAT_API_KEY,
    AZURE_CHAT_API_VERSION,
    AZURE_CHAT_MODEL
)

def get_chat_model():
    """Return Azure ChatGPT model instance."""
    print("ver : ", AZURE_CHAT_API_VERSION)
    print("model : ", AZURE_CHAT_MODEL)

    return AzureChatOpenAI(
        api_key=AZURE_CHAT_API_KEY,
        azure_endpoint=AZURE_CHAT_ENDPOINT,
        api_version=AZURE_CHAT_API_VERSION,
        model=AZURE_CHAT_MODEL,
        temperature=0
    )
