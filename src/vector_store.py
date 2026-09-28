import logging
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import settings
from src.embeddings import get_embeddings_model

logger = logging.getLogger(__name__)

def create_vector_store(chunks_data: List[Dict[str, Any]], session_id: str) -> FAISS:
    """
    Converts list of chunk dictionaries into LangChain Document objects,
    generates their embeddings, builds a FAISS index, and saves it to disk.
    
    Args:
        chunks_data (List[Dict[str, Any]]): List of text chunks with source & page metadata.
        session_id (str): Unique identifier for the user session to prevent overlap.
        
    Returns:
        FAISS: The constructed LangChain FAISS vector store.
    """
    logger.info(f"Creating vector store for session '{session_id}' with {len(chunks_data)} chunks...")
    
    # 1. Convert raw chunk dicts into LangChain Document objects
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk["source"],
                "page": chunk["page"]
            }
        )
        for chunk in chunks_data
    ]

    # 2. Get the cached embeddings model (all-MiniLM-L6-v2)
    embeddings = get_embeddings_model()

    # 3. Create FAISS index from documents and embeddings
    # This calls the Hugging Face model locally to embed all the text chunks.
    db = FAISS.from_documents(documents, embeddings)

    # 4. Save the index to disk for persistence
    session_dir = settings.INDEX_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    db.save_local(str(session_dir))
    logger.info(f"Vector store saved successfully at: {session_dir}")
    
    return db

def load_vector_store(session_id: str) -> Optional[FAISS]:
    """
    Loads an existing FAISS vector store from disk.
    
    Args:
        session_id (str): Unique identifier for the user session.
        
    Returns:
        Optional[FAISS]: The loaded vector store, or None if the index does not exist.
    """
    session_dir = settings.INDEX_DIR / session_id
    if not (session_dir / "index.faiss").exists():
        logger.info(f"No FAISS index found for session '{session_id}'.")
        return None

    logger.info(f"Loading FAISS index for session '{session_id}' from {session_dir}...")
    embeddings = get_embeddings_model()
    
    try:
        # allow_dangerous_deserialization=True is required because FAISS saves
        # metadata using Python's pickle module. Since we create the index ourselves
        # locally on the user's computer, it is safe to load.
        db = FAISS.load_local(
            folder_path=str(session_dir),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully.")
        return db
    except Exception as e:
        logger.error(f"Error loading vector store for session '{session_id}': {str(e)}")
        return None

def delete_vector_store(session_id: str) -> None:
    """
    Deletes the FAISS index files for a user session from disk.
    
    Args:
        session_id (str): Unique identifier for the user session.
    """
    session_dir = settings.INDEX_DIR / session_id
    if session_dir.exists():
        logger.info(f"Deleting vector store directory for session '{session_id}' at {session_dir}...")
        try:
            shutil.rmtree(session_dir)
            logger.info("Vector store directory deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete vector store directory: {str(e)}")
    else:
        logger.info(f"No vector store directory found to delete at {session_dir}.")
