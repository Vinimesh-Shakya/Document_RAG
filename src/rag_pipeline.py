import logging
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from src.retriever import retrieve_relevant_chunks
from src.prompts import SYSTEM_PROMPT, format_context_for_prompt
from src.llm_client import generate_answer

logger = logging.getLogger(__name__)

def run_rag_pipeline(
    db: FAISS,
    query: str,
    top_k: Optional[int] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes the full RAG workflow:
    1. Retrieval: Finds semantically similar document chunks for the query.
    2. Context Formatting: Packages these chunks for the LLM system prompt.
    3. Memory Formatting: Injects recent conversation history (limited to avoid token bloat).
    4. Generation: Queries the Groq LLM under strict factuality instructions.
    5. Citation Compilation: Programmatically structures citations from metadata.
    
    Args:
        db (FAISS): The active vector store index.
        query (str): The user's question.
        top_k (int, optional): The number of chunks to retrieve. Defaults to configuration.
        chat_history (List[Dict[str, str]], optional): A list of previous conversation turns.
        
    Returns:
        Dict[str, Any]: A dictionary containing:
            - "answer": The LLM response.
            - "retrieved_chunks": List of chunk information (for UI debugging).
            - "citations": Unique source citations extracted programmatically.
    """
    # 1. Retrieve the top-K chunks from the database
    retrieved_chunks = retrieve_relevant_chunks(db, query, top_k=top_k)
    
    # 2. Format the retrieved chunks for the prompt
    context_str = format_context_for_prompt(retrieved_chunks)
    
    # Assemble the system prompt with the context injected
    full_system_prompt = (
        f"{SYSTEM_PROMPT}\n"
        f"--- Ground Truth Context Documents ---\n"
        f"{context_str}\n"
        f"--- End of Context Documents ---"
    )

    # 3. Format the recent conversation history
    # We restrict this to the last 3 turns (6 messages) to prevent:
    # - "Context Window Bloat": sending too many tokens, leading to high latency.
    # - "Memory Loss / Confusion": older history might drift from the current question topic.
    history_prefix = ""
    if chat_history:
        # Take last 6 entries (e.g. 3 user-assistant cycles)
        recent_turns = chat_history[-6:]
        lines = []
        for msg in recent_turns:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_prefix = "Recent Chat History (for context):\n" + "\n".join(lines) + "\n\n"

    # Compile user prompt with context history if available
    user_prompt = f"{history_prefix}Current Question: {query}"

    # 4. Invoke the Groq LLM to generate the answer
    # It receives instructions to answer *only* based on the context in full_system_prompt
    answer = generate_answer(
        system_prompt=full_system_prompt,
        user_prompt=user_prompt,
        model_name=model_name
    )

    # 5. Programmatically compile the citations directly from the chunk metadata
    # This prevents the LLM from hallucinating page numbers in the text.
    citations = []
    for chunk in retrieved_chunks:
        source = chunk["source"]
        page = chunk["page"]
        citation = f"{source} (Page {page})"
        if citation not in citations:
            citations.append(citation)

    return {
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
        "citations": citations
    }
