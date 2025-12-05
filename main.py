import os
import glob
import shutil
import tempfile
import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool # <--- IMPORTANT IMPORT
from pydantic import BaseModel

from blob.blob_utils import upload_blob, list_blobs, delete_blob
from vectors.index_builder import rebuild_faiss, delete_pdf_from_faiss, _clear_faiss_store
from vectors.vector_service import load_faiss, hybrid_search, clear_cache 
from ml.chat_model import get_chat_model
from config import MODEL_INPUT_PRICE, MODEL_OUTPUT_PRICE, PDF_FOLDER, INDEX_FOLDER, FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, PDF_CHUNK_MAP_FILE

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

conversation_history = {}  
faiss_lock = asyncio.Lock()

# -------------------------------
# Auto-Cleanup Logic
# -------------------------------
def _clean_os_temp_junk():
    temp_dir = tempfile.gettempdir()
    patterns = ["faiss_load_*", "BlobRegistryFiles*", "tmp*.pdf"]
    deleted_count = 0
    
    for pattern in patterns:
        search_path = os.path.join(temp_dir, pattern)
        for path in glob.glob(search_path):
            try:
                if "faiss_load_" in path or "BlobRegistryFiles" in path or path.endswith(".pdf"):
                    if os.path.isdir(path): shutil.rmtree(path)
                    else: os.remove(path)
                    deleted_count += 1
            except Exception: pass
    return deleted_count

async def run_periodic_cleanup():
    while True:
        try:
            await asyncio.sleep(6 * 3600) # 6 Hours
            await asyncio.to_thread(_clean_os_temp_junk)
        except asyncio.CancelledError: break
        except Exception: await asyncio.sleep(60)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(run_periodic_cleanup())
    yield
    cleanup_task.cancel()
    try: await cleanup_task
    except asyncio.CancelledError: pass

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal Server Error", "details": str(exc)})

def _format_context(chunks: list) -> str:
    formatted = []
    for c in chunks:
        if isinstance(c, str):
            formatted.append(c)
        elif hasattr(c, "page_content"):
            meta = getattr(c, "metadata", {})
            source = meta.get("source", "unknown")
            chunk_id = meta.get("chunk_id", "?")
            formatted.append(f"--- Source: {source} | Chunk: {chunk_id} ---\n{c.page_content}")
    return "\n\n".join(formatted)

# -------------------------------
# Upload PDFs (Multiple) - Optimized
# -------------------------------
@app.post("/uploads")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    uploaded_files = []
    
    try:
        for file in files:
            data = await file.read()
            target_path = f"{PDF_FOLDER}{file.filename}"
            # OPTIMIZATION: Run blocking upload in threadpool
            await run_in_threadpool(upload_blob, target_path, data)
            uploaded_files.append(file.filename)

        async with faiss_lock:
            # OPTIMIZATION: Run blocking rebuild in threadpool
            status_msg = await run_in_threadpool(rebuild_faiss, new_pdfs=uploaded_files)
            clear_cache()

        return {
            "message": "Bulk upload complete",
            "uploaded_files": uploaded_files,
            "details": status_msg
        }
    except Exception as e:
        logger.error("Bulk upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# List PDF files
# -------------------------------
@app.get("/files")
def list_files():
    try:
        full_paths = list_blobs(prefix=PDF_FOLDER)
        filenames = [os.path.basename(path) for path in full_paths if path.lower().endswith(".pdf")]
        return {"count": len(filenames), "files": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Delete PDF (Single)
# -------------------------------
@app.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    try:
        target_path = f"{PDF_FOLDER}{filename}"
        # Run blocking delete in thread
        deleted = await run_in_threadpool(delete_blob, target_path)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="File not found")

        async with faiss_lock:
            await run_in_threadpool(delete_pdf_from_faiss, filename)
            clear_cache()

            remaining_pdfs = await run_in_threadpool(list_blobs, prefix=PDF_FOLDER)
            pdf_count = len([b for b in remaining_pdfs if b.lower().endswith(".pdf")])
            
            if pdf_count == 0:
                _clear_faiss_store()
                clear_cache()

        return {"message": f"Deleted '{filename}'"}
    except HTTPException as he: raise he
    except Exception as e:
        logger.error("Error deleting: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Delete ALL Files (Reset)
# -------------------------------
@app.delete("/delete-all")
async def delete_all_files():
    try:
        deleted_count = 0
        all_pdfs = await run_in_threadpool(list_blobs, prefix=PDF_FOLDER)
        
        for pdf_path in all_pdfs:
            try:
                await run_in_threadpool(delete_blob, pdf_path)
                deleted_count += 1
            except Exception: pass

        async with faiss_lock:
            _clear_faiss_store()
            clear_cache()
            conversation_history.clear()

        junk_deleted = await run_in_threadpool(_clean_os_temp_junk)
        
        return {
            "message": "System reset successful.",
            "details": {
                "deleted_files": deleted_count,
                "cleaned_temp": junk_deleted
            }
        }
    except Exception as e:
        logger.error("Reset error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# Ask Endpoint
# -------------------------------
class AskRequest(BaseModel):
    query: str
    chat_id: str | None = None

@app.post("/ask")
async def ask(body: AskRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    chat_id = body.chat_id or str(uuid.uuid4())

    try:
        # OPTIMIZATION: Load index in threadpool
        store = await run_in_threadpool(load_faiss)
        
        if not store:
            async with faiss_lock:
                try:
                    rebuild_msg = await run_in_threadpool(rebuild_faiss)
                    if "EMPTY_STORAGE" in rebuild_msg:
                        return JSONResponse(status_code=404, content={"answer": "Please upload a PDF first.", "chat_id": chat_id, "error_code": "NO_DOCUMENTS"})
                    store = await run_in_threadpool(load_faiss)
                except Exception:
                    raise HTTPException(status_code=500, detail="Knowledge base unavailable.")

        if not store: raise HTTPException(status_code=500, detail="Failed to load KB.")

        history = conversation_history.get(chat_id, [])
        history_text = "\n\n".join([f"USER: {t['user']}\nASSISTANT: {t['assistant']}" for t in history])

        # Search in threadpool
        k = 10 if "calculate" in body.query.lower() or "combine" in body.query.lower() else 4
        chunks = await run_in_threadpool(hybrid_search, store, body.query, k=k)
        
        if not chunks:
            return {"answer": "No relevant info found.", "chat_id": chat_id}
            
        context = _format_context(chunks)

        prompt = f"""
You are an expert insurance analyst. Use ONLY the provided context and history.
Instructions:
1. **Deep Analysis:** The answer is likely contained in the **Context** below, but it may be fragmented or inside a complex table.
2. **Connect the Dots:** If the user asks about a specific rule (e.g., "Age 70"), look for related keywords like "Senior," "Elderly," or "Limit" even if the exact phrase is missing.
3. **Table Reconstruction:** If you see text that looks like a broken table row (e.g., "Plan A ... $50 ... Plan B ... $100"), reconstruct the table logic to answer.
4. **Logical Inference:** You are allowed to make logical deductions based *strictly* on the provided text. (e.g., If "All plans cover X", then "Plan B covers X").
5. **Format:** - Use **Markdown Tables** for comparisons.
   - Use bullet points for lists.
6. **Context:** Use **Chat History** to understand follow-up questions.

==========================
STRICT RESPONSE RULES
==========================
Cite specific all files name used in new line. Don't add not chunk number.
e.g. "(source: Files Name)"

==========================
CONVERSATION HISTORY
==========================
{history_text}

==========================
CONTEXT
==========================
{context}

==========================
QUESTION
==========================
{body.query}
"""
        llm = get_chat_model()
        response = await llm.ainvoke(prompt) # Use Async invoke for LLM

        if chat_id not in conversation_history: conversation_history[chat_id] = []
        conversation_history[chat_id].append({"user": body.query, "assistant": response.content})

        # Metadata processing...
        meta = response.response_metadata or {}
        token_usage = meta.get("token_usage", {})
        
        return {
            "answer": response.content,
            "chat_id": chat_id,
            "metadata": {"model_used": meta.get("model_name"), "token_usage": token_usage}
        }
        
    except HTTPException as he: raise he
    except Exception as e:
        logger.error("Ask error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))