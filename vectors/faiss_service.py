
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

# -----------------------------
# Safe chunk fetch
# -----------------------------
def fetch_chunk(store, doc_id):
    try:
        chunk = store.docstore.lookup(doc_id)
        if chunk is None:
            logger.warning("Chunk %s not found in docstore.", doc_id)
        return chunk
    except AttributeError as e:
        logger.warning("Docstore does not support lookup: %s", e)
        return None

# -----------------------------
# Hybrid search helpers
# -----------------------------
def normalize_scores(scores):
    if scores is None or len(scores) == 0:
        return np.zeros(0)
    scores = np.array(scores, dtype=float)
    if np.max(scores) == 0:
        return np.zeros_like(scores)
    return scores / np.max(scores)

# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(store, query, k=5):
    logger.info("Running hybrid search for query: '%s'", query)

    # FAISS retrieval
    faiss_results = search_faiss(store, query, k=k)
    faiss_ids = [x[0] for x in faiss_results]
    faiss_scores = [x[1] for x in faiss_results]
    logger.info("FAISS retrieved %d results.", len(faiss_ids))

    # BM25 retrieval
    bm25_ids, bm25_scores = search_bm25(store, query, k=k)
    if bm25_ids:
        logger.info("BM25 retrieved %d results.", len(bm25_ids))
    else:
        logger.info("No BM25 results found.")

    # Combine scores
# --- Combine scores (Improved Version) ---
    if bm25_scores is not None and len(bm25_scores) > 0:

        # Convert FAISS distances -> similarity
        faiss_sim = 1 / (1 + np.array(faiss_scores))

        # Normalise both sets
        faiss_norm = normalize_scores(faiss_sim)
        bm25_norm = normalize_scores(bm25_scores)

        alpha, beta = 0.6, 0.4  # fusion weights
        combined_scores = {}

        # Add FAISS scores
        for idx, score in zip(faiss_ids, faiss_norm):
            combined_scores[idx] = combined_scores.get(idx, 0) + alpha * score

        # Add BM25 scores
        for idx, score in zip(bm25_ids, bm25_norm):
            combined_scores[idx] = combined_scores.get(idx, 0) + beta * score

        # Sort by combined score
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        result_ids = [x[0] for x in sorted_ids[:k]]
        logger.info("Combined top %d results.", len(result_ids))

    else:
        # Only FAISS results available
        result_ids = faiss_ids


    # Fetch chunks
    chunks = []
    for doc_id in result_ids:
        chunk = fetch_chunk(store, doc_id)
        if chunk is not None:
            chunks.append(chunk)
        else:
            logger.warning("Chunk %s was skipped because it could not be fetched.", doc_id)

    logger.info("Hybrid search returning %d chunks.", len(chunks))
    return chunks
