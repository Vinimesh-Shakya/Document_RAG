from typing import List, Dict, Any

# System Prompt enforces strict context-grounded response rules on the generative LLM.
# These guidelines restrict the model from drawing on pre-trained external knowledge,
# reducing the possibility of hallucination.
SYSTEM_PROMPT = """You are a document question-answering assistant.

Answer the user's question using ONLY the supplied document context.

Rules:
1. Do not use unsupported external knowledge.
2. Do not invent facts, references, page numbers, or quotations.
3. If the answer is not available in the provided context, clearly say:
   "I could not find this information in the uploaded documents."
4. If the retrieved context is incomplete or uncertain, state that limitation.
5. Cite the relevant source file and page number.
6. Keep answers clear, accurate, and concise.
"""

def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """
    Formatically builds the context block to be injected into the prompt.
    Includes file source and page numbers for the LLM to read and cite.
    
    Args:
        retrieved_chunks (List[Dict[str, Any]]): Top-K matching chunks from the retriever.
        
    Returns:
        str: Formatted context block.
    """
    if not retrieved_chunks:
        return "No document context is available."

    formatted_parts = []
    
    for idx, chunk in enumerate(retrieved_chunks):
        source = chunk.get("source", "Unknown Source")
        page = chunk.get("page", "Unknown Page")
        content = chunk.get("text", "").strip()
        
        formatted_parts.append(
            f"--- Context Segment {idx + 1} ---\n"
            f"Source File: {source}\n"
            f"Page Number: {page}\n"
            f"Content:\n{content}\n"
        )
        
    return "\n".join(formatted_parts)
