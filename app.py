"""
app.py
------
Streamlit User Interface for Beginner-Friendly Adaptive RAG.

Key Features:
- PDF upload and in-memory chunking + vector generation.
- Mock mode toggle so anyone can run it immediately without API keys.
- Visual routing indicators showing why Adaptive RAG chose Direct vs Retrieval vs Fallback.
- Full inspection panels for chunk similarity scores and decision traces.
"""

import streamlit as st
import numpy as np

from document_processor import extract_text_from_pdf, chunk_text
from retriever import load_embedding_model, get_embeddings
from adaptive_rag import run_adaptive_rag
from llm_service import is_mock_mode


# Cache the embedding model so it only loads once into memory
@st.cache_resource(show_spinner="Loading Sentence Transformer model...")
def get_cached_embedding_model():
    return load_embedding_model("all-MiniLM-L6-v2")


def main():
    st.set_page_config(
        page_title="Adaptive RAG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧠 Adaptive RAG")
    st.markdown(
        "A demonstration of **Adaptive Retrieval-Augmented Generation** "
        "built with simple Python concepts, Streamlit, and NumPy."
    )

    # Load local embedding model
    embedding_model = get_cached_embedding_model()

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Settings")

        # Mock Mode Toggle
        default_mock = is_mock_mode()
        mock_mode = st.toggle("Enable Mock Mode (Free / No Key)", value=default_mock)

        api_key = ""
        if not mock_mode:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="Paste AI Studio Key here...",
                help="Get your free key at https://aistudio.google.com/"
            )
            if not api_key:
                st.warning("Please provide an API key or enable Mock Mode above.")

        st.divider()
        st.header("🎛️ Adaptive Thresholds")
        similarity_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.80,
            value=0.35,
            step=0.05,
            help="If document match similarity is below this score, the system falls back safely instead of hallucinating."
        )
        top_k = st.slider("Top-K Chunks to Retrieve", min_value=1, max_value=5, value=3)

        st.divider()
        st.header("📄 Upload Document")
        uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_file is not None:
            # Check if this is a newly uploaded file or previously processed
            if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
                with st.spinner("Extracting text and chunking PDF..."):
                    pdf_bytes = uploaded_file.read()
                    raw_text = extract_text_from_pdf(pdf_bytes)
                    chunks = chunk_text(raw_text, chunk_size=200, overlap=40)

                    st.session_state.current_file_name = uploaded_file.name
                    st.session_state.raw_text = raw_text
                    st.session_state.chunks = chunks

                with st.spinner("Generating embeddings for chunks..."):
                    chunk_texts = [c["text"] for c in chunks]
                    st.session_state.chunk_embeddings = get_embeddings(chunk_texts, embedding_model)

            # Show document stats
            total_chunks = len(st.session_state.get("chunks", []))
            word_count = len(st.session_state.get("raw_text", "").split())
            st.success(f"**Loaded:** {uploaded_file.name}")
            st.metric("Total Words", f"{word_count:,}")
            st.metric("Total Chunks", total_chunks)

    # --- MAIN CONTENT AREA ---
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if "route" in msg:
                # Show route badge
                route_color = {
                    "DIRECT": "🟢 Direct LLM (Skipped Retrieval)",
                    "DOCUMENT_SUMMARY": "📑 Document Summary (Title & Overview Chunks)",
                    "DOCUMENT_RETRIEVAL": "🔵 Document Retrieval (Grounded in PDF)",
                    "LOW_CONFIDENCE_FALLBACK": "🟡 Low-Confidence Fallback (Guard against Hallucination)",
                    "CLARIFY": "🟣 Clarification Requested"
                }.get(msg["route"], msg["route"])
                st.caption(f"**Strategy:** {route_color}")

            st.markdown(msg["content"])

            if msg.get("chunks"):
                with st.expander("🔍 View Retrieved Chunks"):
                    for c in msg["chunks"]:
                        st.markdown(f"**Chunk #{c['chunk_id']}** (Similarity: `{c['score'] * 100:.1f}%`)")
                        st.text(c["text"])
                        st.divider()

            if msg.get("reasoning"):
                with st.expander("🧠 Decision Reasoning Trace"):
                    st.info(msg["reasoning"])

    # Chat input box
    query = st.chat_input("Ask a question about your PDF or anything else...")

    if query:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Run Adaptive RAG
        with st.chat_message("assistant"):
            with st.spinner("Evaluating query and deciding routing strategy..."):
                doc_chunks = st.session_state.get("chunks", [])
                doc_embeddings = st.session_state.get("chunk_embeddings", np.array([]))

                result = run_adaptive_rag(
                    query=query,
                    chunks=doc_chunks,
                    chunk_embeddings=doc_embeddings,
                    embedding_model=embedding_model,
                    mock=mock_mode,
                    api_key=api_key,
                    similarity_threshold=similarity_threshold,
                    top_k=top_k
                )

            # Route visual badge
            badge_map = {
                "DIRECT": "🟢 Direct LLM Answer (Skipped Retrieval)",
                "DOCUMENT_SUMMARY": "📑 Document Summary (Title & Overview Chunks)",
                "DOCUMENT_RETRIEVAL": "🔵 Document Retrieval (Grounded in PDF)",
                "LOW_CONFIDENCE_FALLBACK": "🟡 Low-Confidence Fallback (Avoided Hallucination)",
                "CLARIFY": "🟣 Clarification Requested"
            }
            st.caption(f"**Strategy:** {badge_map.get(result['route_taken'], result['route_taken'])}")

            st.markdown(result["answer"])

            # Inspect retrieved chunks
            if result.get("retrieved_chunks"):
                with st.expander("🔍 View Retrieved Chunks"):
                    for c in result["retrieved_chunks"]:
                        st.markdown(f"**Chunk #{c['chunk_id']}** (Cosine Similarity: `{c['score'] * 100:.1f}%`)")
                        st.text(c["text"])
                        st.divider()

            # Inspect decision reasoning
            if result.get("reasoning"):
                with st.expander("🧠 Decision Reasoning Trace"):
                    st.info(result["reasoning"])

        # Store in session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "route": result["route_taken"],
            "chunks": result.get("retrieved_chunks", []),
            "reasoning": result.get("reasoning", "")
        })


if __name__ == "__main__":
    main()
