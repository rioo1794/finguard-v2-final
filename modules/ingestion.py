import re
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text(pdf_file):
    """Extract text from uploaded PDF"""
    doc = pymupdf.open(stream=pdf_file.read(), filetype="pdf")
    pages = []
    for page in doc:
        pages.append(page.get_text())
    full_text = "\n".join(pages)
    doc.close()
    return full_text, len(pages)

def scrub_pii(text):
    """Remove PII before processing"""
    # Aadhaar numbers (12 digits)
    text = re.sub(r'\b[2-9]{1}[0-9]{3}\s?[0-9]{4}\s?[0-9]{4}\b',
                  '[AADHAAR-REDACTED]', text)
    # PAN card (AAAAA0000A format)
    text = re.sub(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',
                  '[PAN-REDACTED]', text)
    # Phone numbers
    text = re.sub(r'\b[6-9]\d{9}\b', '[PHONE-REDACTED]', text)
    # Email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL-REDACTED]', text)
    # Bank account numbers (9-18 digits)
    text = re.sub(r'\b[0-9]{9,18}\b', '[ACCOUNT-REDACTED]', text)
    return text

def chunk_text(text):
    """Split text into chunks for embedding"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)