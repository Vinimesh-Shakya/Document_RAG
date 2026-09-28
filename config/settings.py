import os
from pathlib import Path
from dotenv import load_dotenv

# Get the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env file
load_dotenv(BASE_DIR / ".env")

# Project directories
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
INDEX_DIR = BASE_DIR / "data" / "indexes"

# Create directories if they do not exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Retrieval-Augmented Generation (RAG) Config Defaults
# Chunk size is the maximum number of characters per text block.
# 800 characters represents roughly 100-150 words - a good size for semantic integrity.
DEFAULT_CHUNK_SIZE = 800

# Overlap is the number of characters shared between adjacent chunks.
# 150 characters represents roughly 20-30 words, preserving context across chunk splits.
DEFAULT_CHUNK_OVERLAP = 150

# Top-K is the number of most similar document chunks to retrieve for LLM grounding.
DEFAULT_TOP_K = 4

# Local CPU Embedding model name (Hugging Face)
# all-MiniLM-L6-v2 is an extremely lightweight and high-performing sentence transformer model.
# Dimension: 384 dimensions. Runs very fast on standard CPUs.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Client Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# Check and warn if the API key is missing
def validate_config() -> bool:
    """
    Validates that the required configurations are set properly.
    Returns True if valid, False otherwise.
    """
    if not GROQ_API_KEY:
        print("[WARNING] GROQ_API_KEY is not set in the environment or .env file. "
              "LLM responses will not work. Please add it to your .env file.")
        return False
    return True
