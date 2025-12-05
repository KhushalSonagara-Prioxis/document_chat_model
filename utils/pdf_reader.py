import fitz  # PyMuPDF
import re
import logging
import os

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove multiple spaces
    - Remove non-printable characters
    """
    if not text:
        return ""
    
    # Normalize whitespace (tabs, newlines -> single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    
    return text.strip()

def read_pdf_text(path: str) -> str:
    """
    Extract text from PDF using block-based extraction.
    Returns empty string if file is unreadable (logs warning).
    """
    if not os.path.exists(path):
        logger.error(f"PDF file not found locally: {path}")
        raise FileNotFoundError(f"Local PDF path missing: {path}")

    try:
        doc = fitz.open(path)
        full_text = []

        for page_num, page in enumerate(doc):
            try:
                blocks = page.get_text("blocks")
                page_text = []
                for b in blocks:
                    block_text = b[4].strip()
                    if block_text:
                        page_text.append(block_text)
                
                page_text_str = " ".join(page_text)
                page_text_str = clean_text(page_text_str)
                if page_text_str:
                    full_text.append(page_text_str)
            except Exception as page_err:
                logger.warning(f"Error reading page {page_num} of {path}: {page_err}")
                continue

        doc.close()
        
        extracted_text = "\n\n".join(full_text)
        
        if not extracted_text:
            logger.warning(f"PDF {path} extracted to empty text (possibly scanned image).")
            
        return extracted_text

    except Exception as e:
        logger.error(f"Failed to open/read PDF {path}: {e}")
        raise ValueError(f"Corrupt or unreadable PDF file: {os.path.basename(path)}")