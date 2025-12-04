import os
import tempfile
import pickle
import shutil
import numpy as np
import logging
from typing import List, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from ml.embeddings import get_embeddings
from blob.blob_utils import download_blob
# Import folder constants
from config import FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, INDEX_FOLDER
from .faiss_service import search_faiss
from .bm25_service import search_bm25

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

_cached_store: Optional[FAISS] = None

# -----------------------------
# Load FAISS + BM25 (cached)
# -----------------------------
def load_faiss() -> Optional[FAISS]:
    """
    Load FAISS + BM25 from blob storage (INDEX_FOLDER) with persistent cache.
    """
    global _cached_store
    if _cached_store:
        logger.info("Using cached FAISS store.")
        return _cached_store

    try:
        logger.info("Downloading FAISS files from blob folder: %s...", INDEX_FOLDER)
        
        # Prepend INDEX_FOLDER to filenames
        index_bytes = download_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}")
        store_bytes = download_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}")

        try:
            bm25_bytes = download_blob(f"{INDEX_FOLDER}{BM25_FILE}")
            logger.info("BM25 blob downloaded.")
        except Exception:
            bm25_bytes = None
            logger.warning("BM25 blob not found in %s; proceeding without BM25.", INDEX_FOLDER)

        # -----------------------------
        # Save blobs to temporary local dir
        # -----------------------------
        workdir = tempfile.mkdtemp(prefix="faiss_load_")
        idx_path = os.path.join(workdir, FAISS_INDEX_FILE)
        pkl_path = os.path.join(workdir, FAISS_STORE_FILE)
        bm25_path = os.path.join(workdir, BM25_FILE) if bm25_bytes else None

        with open(idx_path, "wb") as f:
            f.write(index_bytes)
        with open(pkl_path, "wb") as f:
            f.write(store_bytes)
        if bm25_bytes:
            with open(bm25_path, "wb") as f:
                f.write(bm25_bytes)

        local_dir = os.path.join(workdir, "faiss_store")
        os.makedirs(local_dir, exist_ok=True)
        shutil.copy(idx_path, os.path.join(local_dir, FAISS_INDEX_FILE))
        shutil.copy(pkl_path, os.path.join(local_dir, FAISS_STORE_FILE))
        if bm25_bytes:
            shutil.copy(bm25_path, os.path.join(local_dir, BM25_FILE))

        embeddings = get_embeddings()
        store = FAISS.load_local(
            local_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )

        # -----------------------------
        # Load BM25 if exists
        # -----------------------------
        if bm25_bytes:
            with open(os.path.join(local_dir, BM25_FILE), "rb") as f:
                bm25_data = pickle.load(f)
                store.bm25_doc_ids = bm25_data.get("doc_ids", [])
                store.bm25 = bm25_data.get("bm25")
                store.bm25_corpus = bm25_data.get("corpus", [])
            logger.info("BM25 loaded successfully.")
        else:
            store.bm25 = None
            store.bm25_corpus = []

        _cached_store = store
        logger.info("FAISS + BM25 loaded successfully.")
        return store

    except Exception as e:
        logger.error("FAISS load error: %s", e, exc_info=True)
        return None

# -----------------------------
# Fetch chunk safely
# -----------------------------
def fetch_chunk(store: FAISS, doc_id: str) -> Optional[Union[str, Document]]:
    if hasattr(store.docstore, "lookup"):
        chunk = store.docstore.lookup(doc_id)
    elif hasattr(store.docstore, "_dict"):
        chunk = store.docstore._dict.get(doc_id, None)
    else:
        logger.warning("Unsupported docstore type: %s", type(store.docstore))
        return None

    if chunk is None:
        logger.warning("Chunk %s not found in docstore.", doc_id)
        return None
    return chunk

# -----------------------------
# Normalize scores
# -----------------------------
def normalize_scores(scores: Union[List[float], np.ndarray, None]) -> np.ndarray:
    if scores is None:
        return np.zeros(0)
    arr = np.array(scores, dtype=float)
    if arr.size == 0:
        return np.zeros(0)
    if np.max(arr) == 0:
        return np.zeros_like(arr)
    return arr / np.max(arr)

# -----------------------------
# Hybrid search
# -----------------------------
def hybrid_search(store: FAISS, query: str, k: int = 5) -> List[str]:
    logger.info("Running hybrid search for query: '%s'", query)

    # --- FAISS search ---
    faiss_results = search_faiss(store, query, k=k) if store else []
    faiss_ids = [x[0] for x in faiss_results]
    faiss_scores = [x[1] for x in faiss_results]

    # --- BM25 search ---
    bm25_ids, bm25_scores = [], []
    if getattr(store, "bm25", None) is not None:
        bm25_ids, bm25_scores = search_bm25(store, query, k=k)

    # --- Adaptive weighting ---
    alpha, beta = 0.6, 0.4
    tokens = query.strip().split()
    if len(tokens) < 3 or any(t.isdigit() for t in tokens):
        alpha, beta = 0.3, 0.7

    combined_scores = {}

    # FAISS scoring
    if faiss_scores:
        faiss_sim = 1 / (1 + np.array(faiss_scores, dtype=float))
        faiss_norm = normalize_scores(faiss_sim)
        for idx, score in zip(faiss_ids, faiss_norm):
            combined_scores[idx] = combined_scores.get(idx, 0) + alpha * score

    # BM25 scoring
    if bm25_scores:
        bm25_norm = normalize_scores(bm25_scores)
        for idx, score in zip(bm25_ids, bm25_norm):
            combined_scores[idx] = combined_scores.get(idx, 0) + beta * score

    # Sort and take top-k
    if combined_scores:
        sorted_ids = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        result_ids = [x[0] for x in sorted_ids[:k]]
    else:
        result_ids = faiss_ids or bm25_ids or []

    # --- Fetch chunks with metadata ---
    chunks = []
    for doc_id in result_ids:
        chunk = fetch_chunk(store, doc_id)
        if not chunk:
            logger.warning("Chunk %s missing in docstore; skipping", doc_id)
            continue
        meta = getattr(chunk, "metadata", {}) if isinstance(chunk, Document) else {}
        meta_str = f"--- Source: {meta.get('source','?')} | Chunk: {meta.get('chunk_id','?')} ---\n"
        text = chunk.page_content if isinstance(chunk, Document) else str(chunk)
        chunks.append(meta_str + text)

    logger.info("Hybrid search returning %d chunks.", len(chunks))
    return chunks

# -----------------------------
# Clear cache (used after deletion)
# -----------------------------
def clear_cache():
    global _cached_store
    _cached_store = None
    logger.info("FAISS cache cleared.")