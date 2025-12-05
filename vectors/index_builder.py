import os
import hashlib
import pickle
import logging
import fitz # PyMuPDF
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from ml.embeddings import get_embeddings
from utils.splitter import split_text
from utils.pdf_reader import clean_text 
from blob.blob_utils import list_blobs, download_blob, upload_blob
from config import FAISS_INDEX_FILE, FAISS_STORE_FILE, PDF_CHUNK_MAP_FILE, BM25_FILE, PDF_FOLDER, INDEX_FOLDER
from vectors.vector_service import load_faiss, _cached_store 

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

BATCH_SIZE = 3 

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
# Chunk PDF (In-Memory)
# -------------------------------
def _chunk_and_metadata_from_pdf_bytes(pdf_filename_only: str, pdf_bytes: bytes) -> Tuple[List[str], List[dict]]:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        full_text = []

        for page in doc:
            blocks = page.get_text("blocks")
            page_text = []
            for b in blocks:
                block_text = b[4].strip()
                if block_text:
                    page_text.append(block_text)
            
            page_str = " ".join(page_text)
            cleaned = clean_text(page_str)
            if cleaned:
                full_text.append(cleaned)
        
        raw_text = "\n\n".join(full_text)
        
        if not raw_text:
            return [], []

        chunks = split_text(raw_text)

        metadatas = []
        for i, c in enumerate(chunks):
            metadatas.append({
                "source": pdf_filename_only,
                "chunk_id": i,
                "chunk_hash": _hash_text(c),
                "text": c
            })
            
        return chunks, metadatas

    except Exception as e:
        logger.error(f"Error parsing PDF bytes for {pdf_filename_only}: {e}")
        return [], []

# -------------------------------
# BM25 Corpus Builder
# -------------------------------
def build_bm25_corpus(texts):
    tokenized = [bm25_tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, texts

# -------------------------------
# Load/Save Map
# -------------------------------
def load_pdf_chunk_map():
    try:
        data = download_blob(f"{INDEX_FOLDER}{PDF_CHUNK_MAP_FILE}")
        return pickle.loads(data)
    except Exception:
        return {}

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
    
    for blob_file in [FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, PDF_CHUNK_MAP_FILE]:
        try:
            from blob.blob_utils import delete_blob
            delete_blob(f"{INDEX_FOLDER}{blob_file}")
        except Exception:
            pass
            
    from vectors.vector_service import clear_cache
    clear_cache()
    logger.info("FAISS store cleared in %s.", INDEX_FOLDER)

# -------------------------------
# Rebuild FAISS
# -------------------------------
def rebuild_faiss(new_pdfs: List[str] = None, rebuild_all: bool = False):
    logger.info("Starting FAISS rebuild/incremental update...")

    try:
        all_blobs = list_blobs(prefix=PDF_FOLDER)
        pdf_files_full_paths = [b for b in all_blobs if b.lower().endswith(".pdf")]

        if not pdf_files_full_paths:
             logger.info("No PDFs found in storage. Clearing index.")
             _clear_faiss_store()
             return "EMPTY_STORAGE"

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

        store = None
        existing_ids = set()
        pdf_chunk_map = load_pdf_chunk_map() if not rebuild_all else {}

        if not rebuild_all:
            try:
                from vectors.vector_service import _cached_store
                if _cached_store:
                    store = _cached_store
                    logger.info("Used cached FAISS index for update.")
                else:
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

            if batch_documents:
                if store is None:
                    store = FAISS.from_documents(batch_documents, embedding=embeddings)
                else:
                    store.add_documents(batch_documents)
                chunks_added_total += len(batch_documents)

        if chunks_added_total == 0 and store is None:
            _clear_faiss_store()
            return "EMPTY_STORAGE"

        store.save_local(local_dir)
        try:
            with open(os.path.join(local_dir, FAISS_INDEX_FILE), "rb") as f_idx:
                upload_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}", f_idx.read())
            with open(os.path.join(local_dir, FAISS_STORE_FILE), "rb") as f_pkl:
                upload_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}", f_pkl.read())
        except Exception as e:
            raise RuntimeError(f"Failed to upload index files: {e}")

        # Rebuild BM25
        try:
            all_docs = list(store.docstore._dict.values())
            if all_docs:
                all_texts = [d.page_content for d in all_docs]
                all_doc_ids = [d.id for d in all_docs]
                bm25, corpus = build_bm25_corpus(all_texts)
                
                bm25_pickle_path = os.path.join(local_dir, BM25_FILE)
                with open(bm25_pickle_path, "wb") as f:
                    pickle.dump({"bm25": bm25, "corpus": corpus, "doc_ids": all_doc_ids}, f)
                with open(bm25_pickle_path, "rb") as f:
                    upload_blob(f"{INDEX_FOLDER}{BM25_FILE}", f.read())
        except Exception as e:
            logger.error(f"BM25 rebuild failed: {e}")

        save_pdf_chunk_map(pdf_chunk_map)
        
        result_msg = f"Processed {total_files} files. Added {chunks_added_total} chunks."
        if errors:
            result_msg += f" (Errors: {len(errors)})."
        return result_msg

    except Exception as e:
        logger.critical(f"Critical error in rebuild_faiss: {e}", exc_info=True)
        raise RuntimeError(f"Index rebuild failed: {e}")

# -------------------------------
# Delete PDF from FAISS (Fixed)
# -------------------------------
def delete_pdf_from_faiss(pdf_filename_only: str):
    """
    Deletes the chunks of a specific PDF from FAISS AND Updates BM25.
    """
    try:
        from vectors.vector_service import clear_cache, load_faiss, _cached_store
        
        store = _cached_store if _cached_store else load_faiss()
        if not store:
            return

        ids_to_delete = [doc_id for doc_id in store.docstore._dict.keys()
                         if doc_id.startswith(pdf_filename_only + "_")]

        if ids_to_delete:
            # 1. Delete from FAISS (Vectors + Docstore)
            store.delete(ids_to_delete)

            local_dir = "/tmp/faiss_store"
            os.makedirs(local_dir, exist_ok=True)
            store.save_local(local_dir)

            # 2. Upload updated FAISS
            with open(os.path.join(local_dir, FAISS_INDEX_FILE), "rb") as f_idx:
                upload_blob(f"{INDEX_FOLDER}{FAISS_INDEX_FILE}", f_idx.read())
            with open(os.path.join(local_dir, FAISS_STORE_FILE), "rb") as f_pkl:
                upload_blob(f"{INDEX_FOLDER}{FAISS_STORE_FILE}", f_pkl.read())

            # 3. CRITICAL FIX: Rebuild & Upload BM25 (Keywords)
            # This ensures BM25 no longer references the deleted chunks
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
                    logger.info("BM25 index updated (deleted items removed).")
            except Exception as bm25_e:
                logger.error(f"Failed to update BM25 after delete: {bm25_e}")

            clear_cache()
            logger.info("Deleted %d chunks for PDF '%s'", len(ids_to_delete), pdf_filename_only)

        # 4. Remove from mapping
        pdf_chunk_map = load_pdf_chunk_map()
        if pdf_filename_only in pdf_chunk_map:
            del pdf_chunk_map[pdf_filename_only]
            save_pdf_chunk_map(pdf_chunk_map)

    except Exception as e:
        logger.error(f"Error deleting PDF from FAISS: {e}")