# utils/pdf_reader.py
import fitz  # PyMuPDF
import re

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Remove multiple spaces
    - Remove repeated headers/footers if obvious
    - Remove non-printable characters
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove non-printable characters
    text = ''.join(c for c in text if c.isprintable())
    return text.strip()

def read_pdf_text(path: str) -> str:
    """
    Extract text from PDF using block-based extraction for better structure.
    Combines all blocks per page into coherent text.
    """
    doc = fitz.open(path)
    full_text = []

    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks")  # better than "text"
        page_text = []
        for b in blocks:
            block_text = b[4].strip()  # block text content
            if block_text:
                page_text.append(block_text)
        page_text_str = " ".join(page_text)
        page_text_str = clean_text(page_text_str)
        if page_text_str:
            full_text.append(page_text_str)

    return "\n\n".join(full_text)
