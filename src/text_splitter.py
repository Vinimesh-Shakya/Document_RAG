import logging
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import settings

logger = logging.getLogger(__name__)

def split_pages_into_chunks(
    pages_data: List[Dict[str, Any]],
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Dict[str, Any]]:
    """
    Splits page-level text into overlapping chunks using a recursive text splitter.
    Preserves document metadata (source name and page number) on each chunk.
    
    Args:
        pages_data (List[Dict[str, Any]]): List of pages from pdf_processor.
        chunk_size (int, optional): Max size of a chunk in characters. Defaults to settings value.
        chunk_overlap (int, optional): Character overlap between adjacent chunks. Defaults to settings value.
        
    Returns:
        List[Dict[str, Any]]: A list of chunk dictionaries, where each dict has:
            - "text": The chunk text.
            - "source": Source PDF name.
            - "page": The page number where this chunk originated.
    """
    if chunk_size is None:
        chunk_size = settings.DEFAULT_CHUNK_SIZE
    if chunk_overlap is None:
        chunk_overlap = settings.DEFAULT_CHUNK_OVERLAP

    logger.info(f"Initializing RecursiveCharacterTextSplitter with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # Recursive character text splitter splits on paragraphs ("\n\n"), then sentences ("\n"),
    # then words (" "), and characters (""), attempting to keep semantic structures together.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks_data: List[Dict[str, Any]] = []

    for page in pages_data:
        text = page["text"]
        source = page["source"]
        page_num = page["page"]
        
        # If the page is empty, we do not need to split it
        if not text:
            continue

        # Split text into chunks
        page_chunks = splitter.split_text(text)
        
        for chunk in page_chunks:
            # We preserve the exact metadata (source file and page number)
            # so the LLM can reference it precisely in citations
            chunks_data.append({
                "text": chunk,
                "source": source,
                "page": page_num
            })

    logger.info(f"Split {len(pages_data)} pages into {len(chunks_data)} text chunks.")
    return chunks_data
