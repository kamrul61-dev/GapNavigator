import pytest
import fitz
from app.parsers.pdf_parser import extract_text_from_pdf, PDFParsingError

def test_extract_text_from_valid_pdf_bytes():
    """Test extracting text from a valid PDF byte stream generated in-memory."""
    # Create an in-memory PDF
    doc = fitz.open()
    page = doc.new_page()
    test_text = "John Doe\nPython Developer\nEmail: john.doe@example.com"
    page.insert_text((50, 50), test_text)
    pdf_bytes = doc.write()
    doc.close()
    
    # Run extractor
    extracted_text = extract_text_from_pdf(pdf_bytes)
    
    # Assertions
    assert "John Doe" in extracted_text
    assert "Python Developer" in extracted_text
    assert "john.doe@example.com" in extracted_text

def test_extract_text_file_not_found():
    """Test that a non-existent file path raises PDFParsingError."""
    with pytest.raises(PDFParsingError) as exc_info:
        extract_text_from_pdf("non_existent_resume_file_path_12345.pdf")
    assert "PDF file not found" in str(exc_info.value)

def test_extract_text_invalid_bytes():
    """Test that parsing random invalid bytes raises PDFParsingError."""
    invalid_bytes = b"This is not a PDF file format content."
    with pytest.raises(PDFParsingError):
        extract_text_from_pdf(invalid_bytes)

def test_extract_text_empty_pdf():
    """Test that a PDF with no readable text raises PDFParsingError."""
    doc = fitz.open()
    doc.new_page()  # Create 1 blank page
    pdf_bytes = doc.write()
    doc.close()
    
    with pytest.raises(PDFParsingError) as exc_info:
        extract_text_from_pdf(pdf_bytes)
    assert "No readable text" in str(exc_info.value)
