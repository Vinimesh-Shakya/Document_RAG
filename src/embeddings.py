import logging
from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Initializes and caches the Hugging Face embedding model.
    By using lru_cache, the model is only loaded into memory once and reused 
    across all requests, saving startup time and RAM.
    
    Returns:
        HuggingFaceEmbeddings: The initialized LangChain wrapper for local CPU embeddings.
    """
    model_name = settings.EMBEDDING_MODEL_NAME
    
    logger.info(f"Loading local embedding model: {model_name} on CPU...")
    
    # Configure the model to run explicitly on CPU
    model_kwargs = {"device": "cpu"}
    
    # We set encode_kwargs. Normalization helps if we want cosine similarity to be 
    # equivalent to simple dot product, which is standard practice in semantic search.
    encode_kwargs = {"normalize_embeddings": True}
    
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        logger.info(f"Embedding model '{model_name}' successfully loaded.")
        return embeddings
    except Exception as e:
        logger.error(f"Failed to load embedding model '{model_name}': {str(e)}")
        raise e
