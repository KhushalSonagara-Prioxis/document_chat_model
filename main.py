import os
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
from pydantic import BaseModel
import uuid
import asyncio

from blob.blob_utils import upload_blob, list_blobs, delete_blob
from vectors.index_builder import rebuild_faiss, delete_pdf_from_faiss, load_pdf_chunk_map, save_pdf_chunk_map
from vectors.vector_service import load_faiss, hybrid_search, clear_cache 
from ml.chat_model import get_chat_model
# Import folder constants
from config import MODEL_INPUT_PRICE, MODEL_OUTPUT_PRICE, PDF_FOLDER, INDEX_FOLDER, FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, PDF_CHUNK_MAP_FILE

logger = logging.getLogger("rag_api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_history = {}  
faiss_lock = asyncio.Lock()

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
# Upload PDF (single)
# -------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    data = await file.read()
    
    # Upload to PDF folder
    target_path = f"{PDF_FOLDER}{file.filename}"
    upload_blob(target_path, data)

    async with faiss_lock:
        try:
            # Pass only the filename; index_builder will prepend PDF_FOLDER to find it
            rebuild_faiss(new_pdfs=[file.filename])
            clear_cache() 
            logger.info("Cache cleared after upload.")
        except Exception as e:
            logger.error("Error rebuilding FAISS: %s", e)
            return {"error": str(e)}

    return {"message": f"Uploaded & indexed '{file.filename}' incrementally"}


# -------------------------------
# Upload PDFs (multiple)
# -------------------------------
@app.post("/uploads")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    if not files:
        return {"error": "No files uploaded"}

    uploaded_files = []
    for file in files:
        data = await file.read()
        
        # Upload to PDF folder
        target_path = f"{PDF_FOLDER}{file.filename}"
        upload_blob(target_path, data)
        
        uploaded_files.append(file.filename)

    async with faiss_lock:
        try:
            rebuild_faiss(new_pdfs=uploaded_files)
            clear_cache()
            logger.info("Cache cleared after bulk upload.")
        except Exception as e:
            logger.error("Error rebuilding FAISS: %s", e)
            return {"error": str(e)}

    return {
        "message": "All PDFs uploaded & indexed successfully",
        "uploaded_files": uploaded_files
    }


# -------------------------------
# Delete PDF
# -------------------------------
@app.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    try:
        # 1️⃣ Delete blob file from PDF folder
        target_path = f"{PDF_FOLDER}{filename}"
        delete_blob(target_path)

        async with faiss_lock:
            # 2️⃣ Delete from FAISS (IDs are stored based on filename only)
            delete_pdf_from_faiss(filename)

            # Remove from map
            pdf_chunk_map = load_pdf_chunk_map()
            if filename in pdf_chunk_map:
                del pdf_chunk_map[filename]
                save_pdf_chunk_map(pdf_chunk_map)
                logger.info("Removed '%s' from pdf_chunk_map", filename)

            clear_cache()

        # 3️⃣ Check if any PDFs remain in the PDF_FOLDER
        # list_blobs now requires the prefix
        remaining_pdfs = [b for b in list_blobs(prefix=PDF_FOLDER) if b.lower().endswith(".pdf")]
        
        if not remaining_pdfs:
            logger.info("No PDFs remaining; cleaning up FAISS/BM25 files in INDEX_FOLDER.")

            # Local cleanup
            faiss_dir = "/tmp/faiss_store"
            if os.path.exists(faiss_dir):
                import shutil
                shutil.rmtree(faiss_dir)

            # Blob cleanup in INDEX_FOLDER
            for blob_file in [FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, PDF_CHUNK_MAP_FILE]:
                try:
                    delete_blob(f"{INDEX_FOLDER}{blob_file}")
                except Exception:
                    pass
            
            clear_cache()

        return {"message": f"Deleted '{filename}' and updated FAISS/BM25 successfully."}

    except Exception as e:
        logger.error("Error deleting PDF '%s': %s", filename, e, exc_info=True)
        return {"error": str(e)}


# -------------------------------
# List PDF files
# -------------------------------
@app.get("/files")
def list_files():
    # Get all blobs starting with "pdfs/"
    full_paths = list_blobs(prefix=PDF_FOLDER)
    
    # Strip the folder path to return only ["file.pdf", ...]
    return [os.path.basename(path) for path in full_paths]

# -------------------------------
# Ask Endpoint
# -------------------------------
class AskRequest(BaseModel):
    query: str
    chat_id: str | None = None


@app.post("/ask")
async def ask(body: AskRequest):
    query = body.query
    chat_id = body.chat_id or str(uuid.uuid4())

    store = load_faiss()
    
    if not store:
        async with faiss_lock:
            try:
                # Fallback: if load fails, try one rebuild (will check PDF_FOLDER)
                rebuild_faiss()
                store = load_faiss()
            except Exception as e:
                logger.error("FAISS load error: %s", e)
                return {"error": "FAISS index could not be loaded or rebuilt"}

    history = conversation_history.get(chat_id, []) if chat_id else []
    history_text = "\n\n".join(
        [f"USER: {turn['user']}\nASSISTANT: {turn['assistant']}" for turn in history]
    ) if history else "(No previous history)"

    k = 10 if "calculate" in query.lower() or "combine" in query.lower() else 4
    chunks = hybrid_search(store, query, k=k)
    context = _format_context(chunks)

    prompt = f"""
You are an expert insurance analyst specialising in MetLife MultiProtect, MetLife EverydayProtect, and similar products.

Use ONLY the provided context and conversation history to answer.

==========================
STRICT RESPONSE RULES
==========================
Cite specific file name used. Don't add not chunk number.

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
{query}
"""

    llm = get_chat_model()
    response = llm.invoke(prompt)

    if chat_id not in conversation_history:
        conversation_history[chat_id] = []
    conversation_history[chat_id].append({
        "user": query,
        "assistant": response.content
    })

    metadata = response.response_metadata or {}
    token_usage = metadata.get("token_usage", {})
    prompt_tokens = token_usage.get("prompt_tokens", 0)
    completion_tokens = token_usage.get("completion_tokens", 0)
    total_tokens = token_usage.get("total_tokens", 0)
    model_name = metadata.get("model_name", "unknown")

    cost_in = (prompt_tokens / 1_000_000) * MODEL_INPUT_PRICE if MODEL_INPUT_PRICE else None
    cost_out = (completion_tokens / 1_000_000) * MODEL_OUTPUT_PRICE if MODEL_OUTPUT_PRICE else None
    total_cost = (cost_in + cost_out) if cost_in and cost_out else None

    return {
        "answer": response.content,
        "chat_id": chat_id,
        "history_length": len(conversation_history.get(chat_id, [])),
        "metadata": {
            "model_used": model_name,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            "pricing": {
                "input_price_per_million": MODEL_INPUT_PRICE,
                "output_price_per_million": MODEL_OUTPUT_PRICE,
            },
            "cost_estimate_usd": {
                "input_cost": cost_in,
                "output_cost": cost_out,
                "total_cost": total_cost,
                "note": "Costs are calculated only if pricing is set in .env"
            },
        }
    }