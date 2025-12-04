# # vectors/bm25_service.py
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def search_bm25(store, query, k=5):
    """
    Returns (list_of_doc_ids, list_of_scores)
    doc_ids are FAISS docstore IDs aligned to the BM25 corpus ordering.
    """
    if not hasattr(store, "bm25") or store.bm25 is None:
        return [], []

    tokenized_query = query.lower().split()
    scores = store.bm25.get_scores(tokenized_query)

    if scores is None or len(scores) == 0:
        return [], []

    # ranked positions in BM25 corpus order
    ranked = np.argsort(scores)[::-1][:k]

    # Validate mapping exists and length matches corpus
    doc_ids_map = getattr(store, "bm25_doc_ids", None)
    corpus_len = len(scores)

    if doc_ids_map is None or len(doc_ids_map) < corpus_len:
        # Fallback: if store has bm25_corpus, try to create a range mapping
        corpus = getattr(store, "bm25_corpus", None)
        if corpus is None:
            logger.error("BM25 doc_id mapping missing and bm25_corpus unavailable.")
            return [], []
        # create default mapping as string indices if no doc ids available
        logger.warning("bm25_doc_ids missing or shorter than corpus; returning BM25 indices as doc ids.")
        doc_ids_map = list(range(len(corpus)))

    # Map BM25 indices back to FAISS docstore IDs (safe access)
    try:
        doc_ids = [doc_ids_map[i] for i in ranked]
    except IndexError as e:
        logger.error("BM25->doc_id mapping index error: %s (ranked=%s, map_len=%d)", e, ranked, len(doc_ids_map))
        # fallback: return BM25 corpus slices (indices) with their scores
        fallback_ids = [int(i) for i in ranked]
        return fallback_ids, scores[ranked].tolist()

    return doc_ids, scores[ranked].tolist()
