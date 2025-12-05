# utils/splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_text(text: str):
    """
    Split text into smaller overlapping chunks for FAISS + BM25.
    - chunk_size: 1600 tokens
    - chunk_overlap: 200 tokens
    This improves retrieval relevance and reduces LLM confusion.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1600,
        chunk_overlap=200
    )
    return splitter.split_text(text)
