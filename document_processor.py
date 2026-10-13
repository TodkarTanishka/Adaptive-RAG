"""
document_processor.py
---------------------
Responsible for handling uploaded documents:
1. Extracting raw text from PDF files using PyMuPDF (fitz).
2. Splitting text into smaller, overlapping chunks using standard Python lists.

Beginner concept note:
- We use pure functions, standard loops, and simple dictionaries.
- No heavy frameworks or complex class hierarchies.
"""

from typing import List, Dict, Any


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts all readable text from a PDF file.

    Parameters:
        pdf_bytes (bytes): The raw bytes of the uploaded PDF file.

    Returns:
        str: Combined plain text extracted from all pages.
    """
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            return "PyMuPDF (fitz) is not installed. Please run: pip install pymupdf"

    # Open the PDF directly from memory bytes
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = []

    # Loop through each page one by one
    for page_number in range(len(doc)):
        page = doc[page_number]
        page_text = page.get_text()
        if page_text.strip():
            all_text.append(page_text.strip())

    doc.close()

    # Join pages with clean line breaks
    return "\n\n".join(all_text)


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> List[Dict[str, Any]]:
    """
    Splits long text into smaller overlapping chunks using simple list slicing.

    Why overlap?
    If an important sentence is split right between chunk 1 and chunk 2,
    having 40 overlapping words ensures the context is not lost!

    Parameters:
        text (str): The full extracted text.
        chunk_size (int): Number of words per chunk (e.g., 200 words).
        overlap (int): Number of words shared between consecutive chunks.

    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries:
        [
            {"chunk_id": 0, "text": "...", "word_count": 200},
            {"chunk_id": 1, "text": "...", "word_count": 200},
            ...
        ]
    """
    # Split text into a simple list of words
    words = text.split()

    if not words:
        return []

    chunks = []
    chunk_id = 0
    start = 0
    step = max(1, chunk_size - overlap)

    # Slide a window across the words list
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk_dict = {
            "chunk_id": chunk_id,
            "text": " ".join(chunk_words),
            "word_count": len(chunk_words)
        }
        chunks.append(chunk_dict)

        chunk_id += 1
        start += step

        # If the remaining words are fewer than the overlap, stop
        if end >= len(words):
            break

    return chunks
