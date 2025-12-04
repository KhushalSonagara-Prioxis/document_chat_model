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
# Constants
# -------------------------------
BATCH_SIZE = 3  # Process 3 files at a time

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
    if not raw_text:
        return [], []

    chunks = split_text(raw_text)

    metadatas = []
    for i, c in enumerate(chunks):
        metadatas.append({
            "source": pdf_filename_only,  # Store clean filename in metadata
            "chunk_id": i,
            "chunk_hash": _hash_text(c),
            "text": c
        })
    # clean up temp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
        
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
        # Return empty map if file missing or corrupt
        return {}

# -------------------------------
# Save PDF → chunk map
# -------------------------------
def save_pdf_chunk_map(pdf_chunk_map):
    try:
        data = pickle.dumps(pdf_chunk_map)
        upload_blob(f"{INDEX_FOLDER}{PDF_CHUNK_MAP_FILE}", data)
    except Exception as e:
        logger.error(f"Failed to save PDF chunk map: {e}")

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
            
    # Reset Cache
    from vectors.vector_service import clear_cache
    clear_cache()
    logger.info("FAISS store cleared in %s.", INDEX_FOLDER)

# -------------------------------
# Full rebuild / incremental FAISS
# -------------------------------
def rebuild_faiss(new_pdfs: List[str] = None, rebuild_all: bool = False):
    """
    Incrementally add new PDFs to FAISS and BM25 in batches.
    """
    logger.info("Starting FAISS rebuild/incremental update...")

    try:
        # 1. List only files in the PDF_FOLDER
        all_blobs = list_blobs(prefix=PDF_FOLDER)
        pdf_files_full_paths = [b for b in all_blobs if b.lower().endswith(".pdf")]

        # Handle Empty Storage Case
        if not pdf_files_full_paths:
             logger.info("No PDFs found in storage. Clearing index.")
             _clear_faiss_store()
             return "EMPTY_STORAGE"

        # 2. Filter logic
        pdf_files_to_process = []
        if new_pdfs:
            target_paths = set(f"{PDF_FOLDER}{p}" for p in new_pdfs)
            pdf_files_to_process = [p for p in pdf_files_full_paths if p in target_paths]
        else:
            pdf_files_to_process = pdf_files_full_paths

        if not pdf_files_to_process:
            return "No matching PDFs found to process."

        embeddings = get_embeddings()
        local_dir = "/tmp/faiss_store"
        os.makedirs(local_dir, exist_ok=True)

        # Load existing FAISS store (if any)
        store = None
        existing_ids = set()
        pdf_chunk_map = load_pdf_chunk_map() if not rebuild_all else {}

        if not rebuild_all:
            try:
                from vectors.vector_service import load_faiss
                store = load_faiss()
                if store:
                    existing_ids = set(store.docstore._dict.keys())
            except Exception:
                logger.info("No existing FAISS store found; starting fresh.")
                pdf_chunk_map = {}
        else:
            pdf_chunk_map = {}

        total_files = len(pdf_files_to_process)
        errors = []
        chunks_added_total = 0

        # --- BATCH PROCESSING LOOP ---
        # Slicing [i : i+3] safely handles cases where remaining files < 3
        for i in range(0, total_files, BATCH_SIZE):
            batch_files = pdf_files_to_process[i : i + BATCH_SIZE]
            batch_documents = []
            
            logger.info(f"Processing Batch {i//BATCH_SIZE + 1}: {len(batch_files)} files...")

            for pdf_full_path in batch_files:
                try:
                    stream = download_blob(pdf_full_path)
                    clean_filename = os.path.basename(pdf_full_path)
                    
                    texts, metadatas = _chunk_and_metadata_from_pdf_bytes(clean_filename, stream)

                    new_ids = []
                    for k, (t, m) in enumerate(zip(texts, metadatas)):
                        uid = f"{clean_filename}_{k}"
                        # Skip duplicates
                        if uid in existing_ids or not t.strip():
                            continue
                        existing_ids.add(uid)
                        new_ids.append(uid)
                        doc = Document(page_content=t, metadata=m, id=uid)
                        batch_documents.append(doc)

                    if new_ids:
                        pdf_chunk_map[clean_filename] = pdf_chunk_map.get(clean_filename, []) + new_ids
                
                except Exception as e:
                    logger.error(f"Failed to process PDF {pdf_full_path}: {e}")
                    errors.append(f"{pdf_full_path}: {str(e)}")
                    continue 

            # --- Add Batch to FAISS ---
            if batch_documents:
                if store is None:
                    # Initialize store with first batch
                    store = FAISS.from_documents(batch_documents, embedding=embeddings)
                    logger.info(f"FAISS initialized with {len(batch_documents)} chunks (Batch {i//BATCH_SIZE + 1}).")
                else:
                    # Append to existing store
                    store.add_documents(batch_documents)
                    logger.info(f"Added {len(batch_documents)} chunks to FAISS (Batch {i//BATCH_SIZE + 1}).")
                
                chunks_added_total += len(batch_documents)
            else:
                logger.info(f"Batch {i//BATCH_SIZE + 1} produced no new chunks (duplicates or empty).")

        # --- END OF BATCH LOOP ---

        if chunks_added_total == 0 and store is None:
            _clear_faiss_store()
            return "EMPTY_STORAGE" # Fallback if everything was corrupt

        # Save FAISS locally then upload
        store.save_local(local_dir)
        try:
            with open(os.path.join(local_dir, FAISS_INDEX_FILE), "rb") as f_idx:
                upload_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}", f_idx.read())
            with open(os.path.join(local_dir, FAISS_STORE_FILE), "rb") as f_pkl:
                upload_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}", f_pkl.read())
        except Exception as e:
            raise RuntimeError(f"Failed to upload index files to blob: {e}")

        # Rebuild BM25 (Must be done on the FULL corpus)
        try:
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
                logger.info(f"BM25 Index rebuilt with {len(all_docs)} total chunks.")
        except Exception as e:
            logger.error(f"BM25 rebuild failed: {e}")

        save_pdf_chunk_map(pdf_chunk_map)
        
        result_msg = f"Processed {total_files} files. Added {chunks_added_total} chunks."
        if errors:
            result_msg += f" (Note: {len(errors)} files failed to process)."
        return result_msg

    except Exception as e:
        logger.critical(f"Critical error in rebuild_faiss: {e}", exc_info=True)
        raise RuntimeError(f"Index rebuild failed: {e}")
    
# -------------------------------
# Delete PDF chunks from FAISS
# -------------------------------
def delete_pdf_from_faiss(pdf_filename_only: str):
    try:
        from .vector_service import clear_cache

        store = load_faiss()
        if not store:
            return

        ids_to_delete = [doc_id for doc_id in store.docstore._dict.keys()
                         if doc_id.startswith(pdf_filename_only + "_")]

        if ids_to_delete:
            store.delete(ids_to_delete)

            local_dir = "/tmp/faiss_store"
            os.makedirs(local_dir, exist_ok=True)
            store.save_local(local_dir)

            # Upload updated FAISS
            with open(os.path.join(local_dir, FAISS_INDEX_FILE), "rb") as f_idx:
                upload_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}", f_idx.read())
            with open(os.path.join(local_dir, FAISS_STORE_FILE), "rb") as f_pkl:
                upload_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}", f_pkl.read())

            clear_cache()
            logger.info("Deleted %d chunks for PDF '%s'", len(ids_to_delete), pdf_filename_only)

        # Remove from map
        pdf_chunk_map = load_pdf_chunk_map()
        if pdf_filename_only in pdf_chunk_map:
            del pdf_chunk_map[pdf_filename_only]
            save_pdf_chunk_map(pdf_chunk_map)

    except Exception as e:
        logger.error(f"Error deleting PDF from FAISS: {e}")
        # Don't raise, just log, so we don't block the file deletion in main.py