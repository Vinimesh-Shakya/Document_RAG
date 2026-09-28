import streamlit as st
import uuid
import shutil
import time
from pathlib import Path
from typing import List

# Setup Streamlit page configuration first before other imports
st.set_page_config(
    page_title="Research Document RAG Assistant",
    page_icon="📚",
    layout="wide"
)

from config import settings
from src.pdf_processor import (
    extract_text_from_pdf,
    CorruptedPDFException,
    EmptyPDFException,
    ScannedPDFException
)
from src.text_splitter import split_pages_into_chunks
from src.vector_store import create_vector_store, load_vector_store, delete_vector_store
from src.rag_pipeline import run_rag_pipeline

@st.cache_data(ttl=3600)
def get_available_groq_models() -> List[str]:
    """
    Fetches the list of active models directly from the Groq API using the client.
    Filters out whisper (audio) and guard (guardrail) models to keep text models only.
    """
    default_models = [
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
    if not settings.GROQ_API_KEY:
        return default_models

    try:
        from src.llm_client import get_groq_client
        client = get_groq_client()
        models = client.models.list()
        
        text_models = []
        for model in models.data:
            m_id = model.id
            if any(term in m_id.lower() for term in ["whisper", "guard", "audio", "vision"]):
                continue
            text_models.append(m_id)
            
        if text_models:
            return sorted(text_models)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Failed to fetch Groq models dynamically: {str(e)}")
        
    return default_models

def cleanup_expired_sessions(directory: Path, max_age_hours: int = 24):
    """
    Deletes subdirectories that have not been modified in the last `max_age_hours` hours.
    """
    if not directory.exists():
        return
    now = time.time()
    max_age_seconds = max_age_hours * 3600
    for session_dir in directory.iterdir():
        if session_dir.is_dir():
            # Check the modification time of the session folder
            try:
                mtime = session_dir.stat().st_mtime
                age = now - mtime
                if age > max_age_seconds:
                    shutil.rmtree(session_dir)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to delete expired session {session_dir.name}: {str(e)}")

# Ensure essential folders exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Run cleanup on startup to remove abandoned sessions older than 24 hours
cleanup_expired_sessions(settings.UPLOAD_DIR, max_age_hours=24)
cleanup_expired_sessions(settings.INDEX_DIR, max_age_hours=24)

# Define session state variables to track session and uploads across reruns
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []
if "filenames" not in st.session_state:
    st.session_state.filenames = []

session_id = st.session_state.session_id

# ----------------- SIDEBAR DESIGN -----------------
with st.sidebar:
    st.title("RAG Configurations")
    st.write("Control RAG settings and upload documents here.")
    st.markdown("---")
    
    # 1. Document Upload
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload PDF research papers, manuals, or lecture notes."
    )
    
    # Render file name list
    if uploaded_files:
        st.write("**Selected Files:**")
        for f in uploaded_files:
            st.caption(f"{f.name} ({f.size / 1024:.1f} KB)")
            
    # Process files action
    process_button = st.button("Process Documents", type="primary", use_container_width=True)
    
    st.markdown("---")
    
    # 2. Embedding Model Display
    st.subheader("Embedding Model")
    st.info(f"Model: `{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}`\n\nRuns locally on CPU (384 dimensions).")
    
    # 3. Sliding controls for Chunking & Retrieval Parameters
    st.subheader("Parameters")
    
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=200,
        max_value=2000,
        value=settings.DEFAULT_CHUNK_SIZE,
        step=100,
        help="The maximum size of each text block. Larger chunks preserve broader context but might include irrelevant text."
    )
    
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=settings.DEFAULT_CHUNK_OVERLAP,
        step=50,
        help="The amount of text shared between adjacent chunks. Prevents cutting ideas in half."
    )
    
    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=1,
        max_value=10,
        value=settings.DEFAULT_TOP_K,
        step=1,
        help="The number of most relevant chunks retrieved from the FAISS database to answer a question."
    )
    
    # 4. LLM Selection
    # Groq supported models list (fetched dynamically)
    groq_models = get_available_groq_models()
    
    # Attempt to align settings default if available
    default_model_idx = 0
    if settings.LLM_MODEL_NAME in groq_models:
        default_model_idx = groq_models.index(settings.LLM_MODEL_NAME)
    elif len(groq_models) > 0:
        default_model_idx = 0
        
    selected_llm_model = st.selectbox(
        "Generative LLM (Groq)",
        options=groq_models,
        index=default_model_idx,
        help="The Large Language Model hosted on Groq that will synthesize your answer based on context."
    )
    
    st.markdown("---")

    reset_button = st.button("🗑️ Clear Documents & Reset", type="secondary", use_container_width=True)


# ----------------- SESSION RESET FUNCTION -----------------
if reset_button:
    # 1. Clear FAISS files from disk
    delete_vector_store(session_id)
    
    # 2. Clear uploaded files under the session dir
    session_upload_dir = settings.UPLOAD_DIR / session_id
    if session_upload_dir.exists():
        try:
            shutil.rmtree(session_upload_dir)
        except Exception as e:
            st.sidebar.error(f"Error clearing files: {str(e)}")
            
    # 3. Clear st.session_state
    st.session_state.chat_history = []
    st.session_state.processed = False
    st.session_state.retrieved_chunks = []
    st.session_state.filenames = []
    
    st.toast("Application reset completed successfully!", icon="🗑️")
    st.rerun()


# ----------------- MAIN VIEW DESIGN -----------------
st.title("Document RAG Assistant")
st.write("Ask natural-language questions about your uploaded PDF documents. Answers are grounded strictly in your PDFs.")
st.markdown("---")

# ----------------- DOCUMENT PROCESSING LOGIC -----------------
if process_button:
    if not uploaded_files:
        st.error("Please upload at least one PDF file first.")
    else:
        # Create a sandboxed folder for this session's uploaded PDFs
        session_upload_dir = settings.UPLOAD_DIR / session_id
        session_upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files to disk and process them
        all_pages = []
        filenames = []
        
        with st.status("Reading and processing uploaded PDFs...", expanded=True) as status:
            try:
                for u_file in uploaded_files:
                    status.write(f"Saving temporary file: `{u_file.name}`...")
                    temp_path = session_upload_dir / u_file.name
                    with open(temp_path, "wb") as f:
                        f.write(u_file.read())
                    
                    status.write(f"Extracting text from: `{u_file.name}`...")
                    pages_data = extract_text_from_pdf(temp_path)
                    all_pages.extend(pages_data)
                    filenames.append(u_file.name)
                
                status.write(f"Splitting text into chunks (size={chunk_size}, overlap={chunk_overlap})...")
                chunks = split_pages_into_chunks(all_pages, chunk_size, chunk_overlap)
                
                status.write("Generating local embeddings and building FAISS vector index...")
                create_vector_store(chunks, session_id)
                
                # Update session states
                st.session_state.processed = True
                st.session_state.filenames = filenames
                st.session_state.retrieved_chunks = []
                
                status.update(label="Processing Complete! Your database is ready.", state="complete")
                st.success(f"Successfully processed {len(filenames)} documents ({len(chunks)} chunks). You can now ask questions!")
                
            except (CorruptedPDFException, EmptyPDFException, ScannedPDFException) as e:
                status.update(label="Processing Failed.", state="error")
                st.error(str(e))
                # Delete stored files to maintain clean state
                if session_upload_dir.exists():
                    shutil.rmtree(session_upload_dir)
            except Exception as e:
                status.update(label="Unexpected Error Occurred.", state="error")
                st.error(f"An unexpected error occurred during processing: {str(e)}")
                if session_upload_dir.exists():
                    shutil.rmtree(session_upload_dir)


# ----------------- CHAT CONSOLE VIEW -----------------
# Define system warning if Groq key is missing
if not settings.GROQ_API_KEY:
    st.warning("**Groq API Key Missing**: You cannot ask questions yet. Add `GROQ_API_KEY` to your `.env` file in the project folder and restart Streamlit.")

# Prompt initial landing display if no documents processed
if not st.session_state.processed:
    st.info("**Get Started**: Upload one or more research PDFs in the sidebar and click **Process Documents**.")
else:
    st.write(f"**Active Documents:** {', '.join(st.session_state.filenames)}")
    
    # Render conversation history using native chat UI elements
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                st.markdown("**Sources Cited:**")
                for citation in message["citations"]:
                    st.caption(f"- {citation}")

    # Process new user questions
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Check Groq Key Validation
        if not settings.GROQ_API_KEY:
            st.error("Please configure your `GROQ_API_KEY` in the `.env` file first.")
        else:
            # 1. Render user prompt in screen
            with st.chat_message("user"):
                st.markdown(user_query)
            
            # Save user prompt to state
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # 2. Query RAG Assistant Pipeline
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document database and generating answer..."):
                    try:
                        # Load session's FAISS index
                        db = load_vector_store(session_id)
                        if not db:
                            st.error("Error: Could not retrieve the document index. Please try re-processing the files.")
                        else:
                            # Run inference
                            result = run_rag_pipeline(
                                db=db,
                                query=user_query,
                                top_k=top_k,
                                chat_history=st.session_state.chat_history[:-1], # Exclude current query from memory
                                model_name=selected_llm_model
                            )
                            
                            answer = result["answer"]
                            retrieved = result["retrieved_chunks"]
                            citations = result["citations"]
                            
                            # Render response details
                            st.markdown(answer)
                            
                            if citations:
                                st.markdown("**Sources Cited:**")
                                for citation in citations:
                                    st.caption(f"- {citation}")
                                    
                            # Update session states
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "citations": citations
                            })
                            st.session_state.retrieved_chunks = retrieved
                            
                    except Exception as e:
                        error_msg = f"An error occurred while answering your question: {str(e)}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    # Render debug context details if retrieved chunks exist
    if st.session_state.retrieved_chunks:
        st.markdown("---")
        with st.expander("Debug: Show Retrieved Context (FAISS Search Results)"):
            st.write(f"Showing top {len(st.session_state.retrieved_chunks)} relevant document chunks retrieved matching your query:")
            for idx, chunk in enumerate(st.session_state.retrieved_chunks):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.caption(f"**Chunk {idx+1} from {chunk['source']}**")
                with col2:
                    st.caption(f"**Page:** {chunk['page']}")
                with col3:
                    st.caption(f"**FAISS L2 Distance:** {chunk['score']:.4f}")
                st.text_area(
                    label=f"Text content chunk {idx+1}",
                    value=chunk["text"],
                    height=120,
                    key=f"chunk_txt_{idx}",
                    disabled=True
                )


#Run .\.venv\Scripts\streamlit.exe run app.py