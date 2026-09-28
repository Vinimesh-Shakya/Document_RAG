import pytest
from src.text_splitter import split_pages_into_chunks

def test_split_pages_into_chunks_basic():
    """Verify that text splitting preserves source and page metadata across chunk boundaries."""
    pages_data = [
        {
            "text": "This is page one. It has a paragraph of text. " * 30,  # about ~1350 characters
            "source": "example.pdf",
            "page": 1
        },
        {
            "text": "Short page two text.",
            "source": "example.pdf",
            "page": 2
        }
    ]

    # Split using small sizes to guarantee multiple chunks
    chunk_size = 400
    chunk_overlap = 100
    
    chunks = split_pages_into_chunks(
        pages_data=pages_data,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Assertions
    assert len(chunks) > 1
    
    # Check metadata integrity
    for chunk in chunks:
        assert "text" in chunk
        assert len(chunk["text"]) <= chunk_size
        assert chunk["source"] == "example.pdf"
        assert chunk["page"] in [1, 2]
        
    # Check page boundaries mapping
    page_1_chunks = [c for c in chunks if c["page"] == 1]
    page_2_chunks = [c for c in chunks if c["page"] == 2]
    
    assert len(page_1_chunks) > 1
    assert len(page_2_chunks) == 1
    assert page_2_chunks[0]["text"] == "Short page two text."
