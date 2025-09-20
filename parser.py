import io
import re
import fitz            # PyMuPDF (more robust PDF text extraction)
import PyPDF2          # Fallback PDF parser
from docx import Document  # For .docx resumes


def _extract_pdf(raw: bytes) -> str:
    """
    Try robust extraction via PyMuPDF; if it fails, fallback to PyPDF2.
    """
    try:
        # Open from in-memory bytes, not from file path
        doc = fitz.open(stream=raw, filetype="pdf")
        # Group text from all pages
        text = "\n".join(page.get_text("text") for page in doc)
        return text
    
    #if PyMuPDF fails
    except Exception:
        # Fallback: some PDFs parse better with PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    
    
def _extract_docx(raw: bytes) -> str:
    """
    Extract text from a .docx file using python-docx.
    """
    doc = Document(io.BytesIO(raw))
    text = "\n".join(p.text for p in doc.paragraphs)
    return text


def _clean_text(text: str) -> str:
    """
    Normalize whitespace and collapse excessive newlines to make LLM input cleaner.
    """
    text = text.replace("\xa0", " ")            # non-breaking space -> normal space
    text = re.sub(r"[ \t]+", " ", text)         # collapse multiple spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text)      # collapse 3+ newlines into 2
    return text.strip()


def extract_text_from_file(uploaded_file) -> str:
    """
    Unified entrypoint.
    - Reads bytes from the uploaded file.
    - Dispatches to the correct extractor by file type.
    - Returns cleaned text.
    """
    
    MAX_BYTES = 8 * 1024 * 1024  # 8 MB hard cap for uploads
    
    uploaded_file.seek(0) # cache parsing results
    raw = uploaded_file.read()
    
    # Size guardrail
    if len(raw) == 0:
        raise ValueError("The file appears to be empty.")
    if len(raw) > MAX_BYTES:
        raise ValueError("The file is too large (limit ~8MB). Please upload a smaller document.")

    
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        txt = _extract_pdf(raw)
    elif file_type == "text/plain":
        txt = raw.decode("utf-8", errors="ignore")
    elif file_type in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
        txt = _extract_docx(raw)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    return _clean_text(txt)