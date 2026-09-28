import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessorError(Exception):
    """Base exception class for PDF processing errors."""
    pass

class CorruptedPDFException(PDFProcessorError):
    """Raised when a PDF file is corrupted and cannot be opened."""
    pass

class EmptyPDFException(PDFProcessorError):
    """Raised when a PDF file contains no pages or has zero size."""
    pass

class ScannedPDFException(PDFProcessorError):
    """Raised when a PDF has text extraction capacity of zero, indicating scanned pages without OCR."""
    pass

def clean_text_safely(text: str) -> str:
    """
    Cleans extracted text safely by stripping whitespace and normalizing line breaks.
    Does NOT remove punctuation, math expressions, citations, or URLs.
    
    Args:
        text (str): The raw text extracted from a PDF page.
        
    Returns:
        str: The safely preprocessed text.
    """
    if not text:
        return ""
    # Normalize double line endings and strip whitespace
    cleaned = text.strip()
    return cleaned

def extract_text_from_pdf(file_path: Path | str) -> List[Dict[str, Any]]:
    """
    Opens a PDF and extracts text page-by-page, capturing metadata for source citation.
    
    Args:
        file_path (Path | str): Path to the target PDF file.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains:
            - "text": The cleaned extracted text of the page.
            - "source": The filename of the PDF source.
            - "page": The 1-indexed page number.
            
    Raises:
        CorruptedPDFException: If the file is not a valid PDF or is corrupted.
        EmptyPDFException: If the PDF contains no pages.
        ScannedPDFException: If all pages are completely blank (no text can be extracted).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"The PDF file at {path} does not exist.")
        
    if path.stat().st_size == 0:
        raise EmptyPDFException(f"The PDF file at '{path.name}' is empty (0 bytes).")

    try:
        # fitz.open will raise FileDataError for corrupted files
        doc = fitz.open(path)
    except Exception as e:
        logger.error(f"Failed to open PDF file {path.name}: {str(e)}")
        raise CorruptedPDFException(f"The file '{path.name}' appears to be corrupted or is not a valid PDF.") from e

    num_pages = len(doc)
    if num_pages == 0:
        doc.close()
        raise EmptyPDFException(f"The PDF file '{path.name}' contains no pages.")

    pages_data: List[Dict[str, Any]] = []
    blank_pages_count = 0

    for page_idx in range(num_pages):
        page = doc[page_idx]
        raw_text = page.get_text()
        cleaned_text = clean_text_safely(raw_text)
        
        # 1-indexed page number for student-friendly referencing
        page_num = page_idx + 1
        
        if not cleaned_text:
            blank_pages_count += 1
            logger.warning(f"No text extracted from page {page_num} of {path.name}.")
            # We still keep the page metadata, but store empty text.
            # However, if the whole document has zero text, it is scanned.
        
        pages_data.append({
            "text": cleaned_text,
            "source": path.name,
            "page": page_num
        })
        
    doc.close()

    # If all pages returned empty text, it indicates a scanned PDF requiring OCR.
    if blank_pages_count == num_pages:
        raise ScannedPDFException(
            f"The file '{path.name}' contains no extractable text. "
            "It might be a scanned document (image-only PDF). "
            "OCR (Optical Character Recognition) is not implemented in this version."
        )

    logger.info(f"Successfully extracted text from '{path.name}' ({num_pages} pages, {blank_pages_count} blank).")
    return pages_data
