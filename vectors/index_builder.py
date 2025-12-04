import os
import hashlib
import pickle
import logging
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ml.embeddings import get_embeddings
from utils.pdf_reader import read_pdf_text
from utils.splitter import split_text
from blob.blob_utils import list_blobs, download_blob, upload_blob
# Import folder constants
from config import FAISS_INDEX_FILE, FAISS_STORE_FILE, PDF_CHUNK_MAP_FILE, BM25_FILE, PDF_FOLDER, INDEX_FOLDER
from vectors.vector_service import load_faiss, _cached_store

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# -------------------------------
# Deterministic doc ID
# -------------------------------
def _hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()

# -------------------------------
# BM25 tokenizer
# -------------------------------
def bm25_tokenize(text: str):
    import re
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

# -------------------------------
# Chunk PDF + Metadata
# -------------------------------
def _chunk_and_metadata_from_pdf_bytes(pdf_filename_only: str, pdf_bytes: bytes) -> Tuple[List[str], List[dict]]:
    """
    pdf_filename_only: Just 'file.pdf', stripped of 'pdfs/' folder prefix.
    """
    tmp_path = f"/tmp/{pdf_filename_only}"
    with open(tmp_path, "wb") as f:
        f.write(pdf_bytes)

    raw_text = read_pdf_text(tmp_path)
    chunks = split_text(raw_text)

    metadatas = []
    for i, c in enumerate(chunks):
        metadatas.append({
            "source": pdf_filename_only,  # Store clean filename in metadata
            "chunk_id": i,
            "chunk_hash": _hash_text(c),
            "text": c
        })
    return chunks, metadatas

# -------------------------------
# BM25 Corpus Builder
# -------------------------------
def build_bm25_corpus(texts):
    tokenized = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts

# -------------------------------
# Load PDF → chunk map
# -------------------------------
def load_pdf_chunk_map():
    try:
        # Load from INDEX_FOLDER
        data = download_blob(f"{INDEX_FOLDER}{PDF_CHUNK_MAP_FILE}")
        return pickle.loads(data)
    except Exception:
        return {}

# -------------------------------
# Save PDF → chunk map
# -------------------------------
def save_pdf_chunk_map(pdf_chunk_map):
    data = pickle.dumps(pdf_chunk_map)
    # Save to INDEX_FOLDER
    upload_blob(f"{INDEX_FOLDER}{PDF_CHUNK_MAP_FILE}", data)

# -------------------------------
# Clear FAISS
# -------------------------------
def _clear_faiss_store():
    workdir = "/tmp/faiss_store"
    if os.path.exists(workdir):
        import shutil
        shutil.rmtree(workdir)
    
    # Delete from INDEX_FOLDER
    for blob_file in [FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, PDF_CHUNK_MAP_FILE]:
        try:
            from blob.blob_utils import delete_blob
            delete_blob(f"{INDEX_FOLDER}{blob_file}")
        except Exception:
            pass
    global _cached_store
    _cached_store = None
    logger.info("FAISS store cleared in %s.", INDEX_FOLDER)

# -------------------------------
# Full rebuild / incremental FAISS
# -------------------------------
def rebuild_faiss(new_pdfs: List[str] = None, rebuild_all: bool = False):
    """
    Incrementally add new PDFs to FAISS and BM25.
    new_pdfs: List of filenames (e.g., ['file1.pdf']), NOT paths.
    """
    logger.info("Starting FAISS rebuild/incremental update...")

    # 1. List only files in the PDF_FOLDER
    all_blobs = list_blobs(prefix=PDF_FOLDER)
    pdf_files_full_paths = [b for b in all_blobs if b.lower().endswith(".pdf")]

    # 2. Filter logic: Match just the filename part
    # pdf_files_full_paths contains ['pdfs/a.pdf', 'pdfs/b.pdf']
    if new_pdfs:
        # Create a set of expected full paths: {'pdfs/new.pdf'}
        target_paths = set(f"{PDF_FOLDER}{p}" for p in new_pdfs)
        pdf_files_to_process = [p for p in pdf_files_full_paths if p in target_paths]
    else:
        pdf_files_to_process = pdf_files_full_paths

    if not pdf_files_to_process:
        logger.info("No PDFs found to process.")
        if not pdf_files_full_paths:
             # Only clear if NO PDFs exist at all in storage
            _clear_faiss_store()
            return "FAISS cleared; no PDFs found."
        return "No new PDFs to add."

    embeddings = get_embeddings()
    local_dir = "/tmp/faiss_store"
    os.makedirs(local_dir, exist_ok=True)

    # Load existing FAISS store from INDEX_FOLDER
    store = None
    existing_ids = set()
    pdf_chunk_map = load_pdf_chunk_map() if not rebuild_all else {}

    try:
        # Check if index file exists in local temp (downloaded via load_faiss or previous step)
        # For simplicity, we assume we might need to fetch it if not in memory, 
        # but normally load_faiss handles the download.
        # Here we try to load what is currently in blob if we don't have it.
        if os.path.exists(os.path.join(local_dir, FAISS_STORE_FILE)):
            store = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
            existing_ids = set(store.docstore._dict.keys())
            logger.info("Existing FAISS store loaded for incremental update.")
        else:
            # Try loading via service to fetch from blob
            from vectors.vector_service import load_faiss
            store = load_faiss()
            if store:
                existing_ids = set(store.docstore._dict.keys())
    except Exception:
        logger.info("No existing FAISS store found; starting fresh.")
        pdf_chunk_map = {}

    documents_to_add = []

    for pdf_full_path in pdf_files_to_process:
        logger.info("Processing PDF: %s", pdf_full_path)
        stream = download_blob(pdf_full_path)
        
        # Strip "pdfs/" from name to use as clean ID/Source
        clean_filename = os.path.basename(pdf_full_path)
        
        texts, metadatas = _chunk_and_metadata_from_pdf_bytes(clean_filename, stream)

        new_ids = []
        for i, (t, m) in enumerate(zip(texts, metadatas)):
            # Create deterministic ID based on clean filename
            uid = f"{clean_filename}_{i}"
            if uid in existing_ids or not t.strip():
                continue
            existing_ids.add(uid)
            new_ids.append(uid)
            doc = Document(page_content=t, metadata=m, id=uid)
            documents_to_add.append(doc)

        if new_ids:
            pdf_chunk_map[clean_filename] = pdf_chunk_map.get(clean_filename, []) + new_ids

    if not documents_to_add and store is None:
        _clear_faiss_store()
        # It's possible we just processed files that were already indexed
        return "No new chunks added (files already indexed)."

    # Initialize or add to FAISS
    if store is None and documents_to_add:
        store = FAISS.from_documents(documents_to_add[:1], embedding=embeddings)
        if len(documents_to_add) > 1:
            store.add_documents(documents_to_add[1:])
        logger.info("FAISS initialized with %d chunks.", len(documents_to_add))
    elif documents_to_add:
        store.add_documents(documents_to_add)
        logger.info("Incrementally added %d new chunks to FAISS.", len(documents_to_add))
    else:
        logger.info("No new chunks to add; FAISS store unchanged.")

    # Save FAISS locally then upload to INDEX_FOLDER
    store.save_local(local_dir)
    with open(os.path.join(local_dir, FAISS_INDEX_FILE), "rb") as f_idx:
        upload_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}", f_idx.read())
    with open(os.path.join(local_dir, FAISS_STORE_FILE), "rb") as f_pkl:
        upload_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}", f_pkl.read())

    # Rebuild BM25 from entire FAISS store
    all_docs = list(store.docstore._dict.values())
    if all_docs:
        all_texts = [d.page_content for d in all_docs]
        all_doc_ids = [d.id for d in all_docs]
        bm25, corpus = build_bm25_corpus(all_texts)
        bm25_pickle_path = os.path.join(local_dir, BM25_FILE)
        with open(bm25_pickle_path, "wb") as f:
            pickle.dump({
                "bm25": bm25,
                "corpus": corpus,
                "doc_ids": all_doc_ids
            }, f)
        with open(bm25_pickle_path, "rb") as f:
            upload_blob(f"{INDEX_FOLDER}{BM25_FILE}", f.read())
        logger.info("BM25 index rebuilt successfully with %d chunks.", len(all_texts))

    # Save updated PDF → chunk map
    save_pdf_chunk_map(pdf_chunk_map)
    logger.info("Updated PDF → chunk map saved.")
    
    return "FAISS + BM25 rebuilt successfully"

# -------------------------------
# Delete PDF chunks from FAISS
# -------------------------------
def delete_pdf_from_faiss(pdf_filename_only: str):
    """
    pdf_filename_only: 'file.pdf' (no folder path)
    """
    from .vector_service import clear_cache

    store = load_faiss()
    if not store:
        return

    # IDs are stored as "filename.pdf_1", so we search for that prefix
    ids_to_delete = [doc_id for doc_id in store.docstore._dict.keys()
                     if doc_id.startswith(pdf_filename_only + "_")]

    if ids_to_delete:
        store.delete(ids_to_delete)

        local_dir = "/tmp/faiss_store"
        os.makedirs(local_dir, exist_ok=True)
        store.save_local(local_dir)

        # Upload updated FAISS to INDEX_FOLDER
        with open(os.path.join(local_dir, FAISS_INDEX_FILE), "rb") as f_idx:
            upload_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}", f_idx.read())
        with open(os.path.join(local_dir, FAISS_STORE_FILE), "rb") as f_pkl:
            upload_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}", f_pkl.read())

        clear_cache()
        logger.info("Deleted %d chunks for PDF '%s'", len(ids_to_delete), pdf_filename_only)

    # Remove from PDF → chunk map
    pdf_chunk_map = load_pdf_chunk_map()
    if pdf_filename_only in pdf_chunk_map:
        del pdf_chunk_map[pdf_filename_only]
        save_pdf_chunk_map(pdf_chunk_map)