import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
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

# -------------------------------
# Global Exception Handler
# -------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global error occurred: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "details": str(exc)},
    )

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
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename missing")

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="File is empty")
        
        target_path = f"{PDF_FOLDER}{file.filename}"
        upload_blob(target_path, data)

        async with faiss_lock:
            # Rebuild index
            status_msg = rebuild_faiss(new_pdfs=[file.filename])
            clear_cache() 
            logger.info("Cache cleared after upload.")

        return {"message": f"Uploaded & indexed '{file.filename}'", "details": status_msg}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error uploading/indexing: %s", e)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# -------------------------------
# Upload PDFs (multiple)
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
            upload_blob(target_path, data)
            uploaded_files.append(file.filename)

        async with faiss_lock:
            status_msg = rebuild_faiss(new_pdfs=uploaded_files)
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
        return [os.path.basename(path) for path in full_paths]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {str(e)}")


# -------------------------------
# Delete PDF
# -------------------------------
@app.delete("/delete/{filename}")
async def delete_pdf(filename: str):
    try:
        # 1. Delete blob
        target_path = f"{PDF_FOLDER}{filename}"
        deleted = delete_blob(target_path)
        
        if not deleted:
            raise HTTPException(status_code=404, detail=f"File {filename} not found in storage.")

        async with faiss_lock:
            # 2. Delete from FAISS
            delete_pdf_from_faiss(filename)

            clear_cache()

            # 3. Check if empty
            remaining_pdfs = [b for b in list_blobs(prefix=PDF_FOLDER) if b.lower().endswith(".pdf")]
            
            if not remaining_pdfs:
                logger.info("No PDFs remaining; cleaning up Index.")
                # Local cleanup
                faiss_dir = "/tmp/faiss_store"
                if os.path.exists(faiss_dir):
                    import shutil
                    shutil.rmtree(faiss_dir)
                # Blob cleanup
                for blob_file in [FAISS_INDEX_FILE, FAISS_STORE_FILE, BM25_FILE, PDF_CHUNK_MAP_FILE]:
                    delete_blob(f"{INDEX_FOLDER}{blob_file}")
                
                clear_cache()

        return {"message": f"Deleted '{filename}' successfully."}

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error deleting PDF: %s", e)
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

    query = body.query
    chat_id = body.chat_id or str(uuid.uuid4())

    try:
        store = load_faiss()
        
        # --- Handle Empty Store / First Run ---
        if not store:
            async with faiss_lock:
                try:
                    # Attempt to rebuild if files exist but index is missing
                    rebuild_msg = rebuild_faiss()
                    
                    if "Index cleared" in rebuild_msg:
                        # Means no PDFs were found in storage
                        return JSONResponse(
                            status_code=404, 
                            content={
                                "answer": "I don't have any documents to reference. Please upload a PDF first.", 
                                "chat_id": chat_id,
                                "error_code": "NO_DOCUMENTS"
                            }
                        )
                    
                    # Retry loading after rebuild
                    store = load_faiss()
                except Exception as e:
                    logger.error("Auto-rebuild failed: %s", e)
                    raise HTTPException(status_code=500, detail="Knowledge base unavailable.")

        if not store:
             # If still None, something is wrong
             raise HTTPException(status_code=500, detail="Failed to load knowledge base.")

        # --- Search & Generate ---
        history = conversation_history.get(chat_id, []) if chat_id else []
        history_text = "\n\n".join(
            [f"USER: {turn['user']}\nASSISTANT: {turn['assistant']}" for turn in history]
        ) if history else "(No previous history)"

        k = 10 if "calculate" in query.lower() or "combine" in query.lower() else 4
        chunks = hybrid_search(store, query, k=k)
        
        if not chunks:
            return {
                "answer": "I couldn't find any information relevant to your query in the uploaded documents.",
                "chat_id": chat_id,
                "metadata": {}
            }
            
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

        # Calculate Metadata/Cost
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
                "token_usage": token_usage,
                "cost_estimate_usd": {
                    "total_cost": total_cost,
                },
            }
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error("Error in /ask endpoint: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))