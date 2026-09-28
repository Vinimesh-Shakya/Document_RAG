import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document
from src.retriever import retrieve_relevant_chunks

def test_retrieve_relevant_chunks_format():
    """Verify that retrieval results are mapped to dictionary outputs with score and page metadata."""
    # 1. Create Mock FAISS db
    mock_db = MagicMock()
    
    # 2. Setup mock documents returned by FAISS similarity search
    doc_1 = Document(
        page_content="Deep learning models use multi-layer artificial neural networks.",
        metadata={"source": "deep_learning.pdf", "page": 4}
    )
    doc_2 = Document(
        page_content="Machine learning is a subset of artificial intelligence.",
        metadata={"source": "ml_basics.pdf", "page": 1}
    )
    
    # Mock FAISS returns: List[Tuple[Document, float]]
    # Lower float represents closer L2 distance
    mock_db.similarity_search_with_score.return_value = [
        (doc_1, 0.1234),
        (doc_2, 0.5678)
    ]
    
    # 3. Call retriever
    query = "What is deep learning?"
    results = retrieve_relevant_chunks(db=mock_db, query=query, top_k=2)
    
    # Assertions
    assert len(results) == 2
    
    # Verify Chunk 1
    assert results[0]["text"] == "Deep learning models use multi-layer artificial neural networks."
    assert results[0]["source"] == "deep_learning.pdf"
    assert results[0]["page"] == 4
    assert results[0]["score"] == 0.1234
    
    # Verify Chunk 2
    assert results[1]["text"] == "Machine learning is a subset of artificial intelligence."
    assert results[1]["source"] == "ml_basics.pdf"
    assert results[1]["page"] == 1
    assert results[1]["score"] == 0.5678
    
    # Verify mock database search was called with correct parameters
    mock_db.similarity_search_with_score.assert_called_once_with(query, k=2)

def test_retrieve_relevant_chunks_empty_query():
    """Verify that search with empty queries logs a warning and returns immediately."""
    mock_db = MagicMock()
    
    # Query is empty
    results = retrieve_relevant_chunks(db=mock_db, query="   ", top_k=5)
    
    assert results == []
    mock_db.similarity_search_with_score.assert_not_called()
