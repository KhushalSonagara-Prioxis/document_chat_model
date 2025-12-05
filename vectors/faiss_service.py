
# vectors/faiss_service.py
import logging
import numpy as np
from langchain_community.vectorstores import FAISS
from .bm25_service import search_bm25

# -----------------------------
# Logging setup
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -----------------------------
# FAISS similarity search
# -----------------------------
def search_faiss(store: FAISS, query: str, k: int = 5):
    """
    Perform FAISS similarity search.
    Returns list of tuples: (docstore_id, score)
    """
    if not store or not hasattr(store, "index") or store.index is None:
        return []

    results = store.similarity_search_with_score(query, k=k)
    output = []
    for doc, score in results:
        doc_id = doc.id  # Internal FAISS docstore ID
        output.append((doc_id, score))
    return output
