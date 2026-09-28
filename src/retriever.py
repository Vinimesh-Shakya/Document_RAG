import logging
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from config import settings

logger = logging.getLogger(__name__)

def retrieve_relevant_chunks(
    db: FAISS,
    query: str,
    top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Searches the FAISS vector database to retrieve the top-K most semantically 
    relevant text chunks matching the user's question.
    
    Args:
        db (FAISS): The loaded vector database.
        query (str): The search query (user's question).
        top_k (int, optional): The number of chunks to return. Defaults to settings value.
        
    Returns:
        List[Dict[str, Any]]: List of retrieved chunk dicts, each with keys:
            - "text": Text contents of the chunk.
            - "source": Source document filename.
            - "page": Original page number (1-indexed).
            - "score": FAISS L2 distance score (lower is more similar).
    """
    if top_k is None:
        top_k = settings.DEFAULT_TOP_K

    if not query.strip():
        logger.warning("Empty search query received. Returning empty list.")
        return []

    logger.info(f"Retrieving top {top_k} chunks for query: '{query}'")

    try:
        # FAISS.similarity_search_with_score returns a list of tuples: (Document, float_score)
        # The float_score is the L2 (Euclidean) distance.
        # A smaller L2 distance indicates that the vectors are closer together in space
        # (i.e. more semantically similar).
        results_with_scores = db.similarity_search_with_score(query, k=top_k)
        
        retrieved_chunks = []
        for doc, score in results_with_scores:
            retrieved_chunks.append({
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", 0),
                "score": float(score)
            })
            
        logger.info(f"Successfully retrieved {len(retrieved_chunks)} relevant chunks from FAISS.")
        return retrieved_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving chunks from FAISS: {str(e)}")
        raise e
