import pytest
from unittest.mock import MagicMock, patch
from src.pdf_processor import (
    extract_text_from_pdf,
    CorruptedPDFException,
    EmptyPDFException,
    ScannedPDFException
)

@patch("fitz.open")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_extract_text_from_pdf_success(mock_stat, mock_exists, mock_open):
    """Test that text is extracted successfully and page numbers are preserved."""
    # Setup mock file details
    mock_exists.return_value = True
    mock_stat.return_value.st_size = 1024
    
    # Setup mock PDF structure
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    
    mock_page_1 = MagicMock()
    mock_page_1.get_text.return_value = " This is standard text on page one. "
    
    mock_page_2 = MagicMock()
    mock_page_2.get_text.return_value = "This is some code: print('hello world') on page two."
    
    mock_doc.__getitem__.side_effect = [mock_page_1, mock_page_2]
    mock_open.return_value = mock_doc
    
    # Execute extraction
    result = extract_text_from_pdf("dummy.pdf")
    
    # Assertions
    assert len(result) == 2
    assert result[0]["text"] == "This is standard text on page one."
    assert result[0]["page"] == 1
    assert result[0]["source"] == "dummy.pdf"
    
    assert "print('hello world')" in result[1]["text"]
    assert result[1]["page"] == 2
    assert result[1]["source"] == "dummy.pdf"
    
    # Verify file was opened and closed
    mock_open.assert_called_once()
    mock_doc.close.assert_called_once()

@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_extract_text_from_pdf_zero_bytes(mock_stat, mock_exists):
    """Test that a 0-byte file raises an EmptyPDFException."""
    mock_exists.return_value = True
    mock_stat.return_value.st_size = 0
    
    with pytest.raises(EmptyPDFException) as exc_info:
        extract_text_from_pdf("empty.pdf")
    assert "empty.pdf' is empty" in str(exc_info.value)

@patch("fitz.open")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_extract_text_from_pdf_corrupted(mock_stat, mock_exists, mock_open):
    """Test that fitz opening errors raise a CorruptedPDFException."""
    mock_exists.return_value = True
    mock_stat.return_value.st_size = 500
    
    # Mock fitz.open raising an exception representing file corruption
    mock_open.side_effect = Exception("Format error: not a PDF")
    
    with pytest.raises(CorruptedPDFException) as exc_info:
        extract_text_from_pdf("corrupt.pdf")
    assert "corrupted or is not a valid PDF" in str(exc_info.value)

@patch("fitz.open")
@patch("pathlib.Path.exists")
@patch("pathlib.Path.stat")
def test_extract_text_from_pdf_scanned(mock_stat, mock_exists, mock_open):
    """Test that documents with all blank pages raise a ScannedPDFException."""
    mock_exists.return_value = True
    mock_stat.return_value.st_size = 2048
    
    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    
    # Mock empty pages (representing a scanned image-only PDF)
    mock_page_1 = MagicMock()
    mock_page_1.get_text.return_value = "   "  # whitespace
    
    mock_page_2 = MagicMock()
    mock_page_2.get_text.return_value = ""     # empty
    
    mock_doc.__getitem__.side_effect = [mock_page_1, mock_page_2]
    mock_open.return_value = mock_doc
    
    with pytest.raises(ScannedPDFException) as exc_info:
        extract_text_from_pdf("scanned.pdf")
    assert "contains no extractable text" in str(exc_info.value)
