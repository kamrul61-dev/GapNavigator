import os
import fitz  # PyMuPDF
from typing import BinaryIO, Union
import logging

logger = logging.getLogger("gapnavigator.parsers.pdf_parser")

class PDFParsingError(Exception):
    """Custom exception raised when PDF parsing fails."""
    pass

def extract_text_from_pdf(pdf_source: Union[str, bytes, BinaryIO]) -> str:
    """
    Extracts plain text from a PDF file source.
    
    Args:
        pdf_source: File path (str), raw bytes (bytes), or file-like object (BinaryIO)
        
    Returns:
        str: Extracted plain text content
        
    Raises:
        PDFParsingError: If the PDF is invalid, empty, or cannot be parsed.
    """
    doc = None
    try:
        if isinstance(pdf_source, str):
            if not os.path.exists(pdf_source):
                raise PDFParsingError(f"PDF file not found at path: {pdf_source}")
            doc = fitz.open(pdf_source)
        elif isinstance(pdf_source, bytes):
            doc = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            # File-like object (e.g. Streamlit UploadedFile)
            # Streamlit UploadedFile has a read() method that returns bytes.
            # PyMuPDF open(stream=...) expects bytes.
            try:
                pdf_bytes = pdf_source.read()
                # If we read it, reset pointer if possible (Streamlit files are usually fine, but good practice)
                if hasattr(pdf_source, "seek"):
                    pdf_source.seek(0)
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            except Exception as e:
                raise PDFParsingError(f"Failed to read from file-like object: {e}")
        
        if not doc:
            raise PDFParsingError("Unable to open PDF document.")
            
        if len(doc) == 0:
            raise PDFParsingError("The provided PDF file is empty (contains 0 pages).")
            
        text_parts = []
        for i, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
                
        extracted_text = "\n".join(text_parts).strip()
        
        if not extracted_text:
            raise PDFParsingError("No readable text could be extracted from the PDF. It might be scanned or image-only.")
            
        logger.info(f"Successfully extracted {len(extracted_text)} characters from PDF.")
        return extracted_text
        
    except fitz.FileDataError as e:
        logger.error(f"Invalid PDF structure: {e}")
        raise PDFParsingError(f"Invalid PDF file structure: {e}")
    except Exception as e:
        if not isinstance(e, PDFParsingError):
            logger.error(f"Unexpected error during PDF parsing: {e}")
            raise PDFParsingError(f"An unexpected error occurred during PDF parsing: {e}")
        raise e
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass
