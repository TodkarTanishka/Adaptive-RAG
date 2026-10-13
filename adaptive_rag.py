"""
adaptive_rag.py
---------------
The core decision-making brain of the application.

Adaptive RAG Workflow:
1. Classify the user query into:
   - DIRECT: General questions or greetings (no document search needed).
   - DOCUMENT_QUERY: Questions requiring information from the PDF.
   - CLARIFY: Vague or ambiguous queries.
2. Route intelligently:
   - If DIRECT -> Send straight to the LLM.
   - If DOCUMENT_QUERY -> Search the document for top matching chunks.
3. Check Confidence / Hallucination Guard:
   - If cosine similarity is too low (< threshold), adapt strategy and inform the user
     that the document lacks this information instead of making up facts.
   - If confidence is high -> Feed relevant chunks into LLM for an accurate grounded answer.
"""

from typing import Dict, Any, List, Optional
from llm_service import call_gemini
from retriever import retrieve_top_k


def classify_query(query: str, api_key: Optional[str] = None, mock: Optional[bool] = None) -> str:
    """
    Uses the LLM (or mock heuristic) to classify query intent.

    Returns one of:
      - 'DIRECT': For greetings, small talk, or general queries.
      - 'DOCUMENT_QUERY': For questions asking about document facts.
      - 'CLARIFY': For vague or incomplete inputs.
    """
    clean_query = query.strip()
    query_word = clean_query.strip("?!., ").lower()

    # Fast heuristic check for simple single-word greetings or small talk
    if query_word in ["hi", "hello", "hey", "good morning", "good evening", "howdy"]:
        return "DIRECT"

    if len(clean_query.split()) <= 1 and query_word in ["why", "what", "how", "huh", "help"]:
        return "CLARIFY"

    # Fast check for document overview / summary queries
    summary_phrases = [
        "what is the uploaded", "what is this pdf about", "what is the pdf about",
        "what is this document about", "what is the document about",
        "summarize", "summary", "overview", "main topic", "what is this paper about"
    ]
    if any(phrase in clean_query.lower() for phrase in summary_phrases):
        return "DOCUMENT_SUMMARY"

    # LLM Classification prompt
    classification_prompt = (
        "You are an AI router in an Adaptive RAG system.\n"
        "Classify the following user query into exactly ONE category:\n"
        "1. DIRECT - For casual greetings, chit-chat, jokes, or general questions not requiring specific documents.\n"
        "2. DOCUMENT_SUMMARY - For requests asking to summarize, overview, or explain what the entire uploaded PDF/document is about.\n"
        "3. DOCUMENT_QUERY - For specific questions asking about targeted facts, numbers, sections, or details in the document.\n"
        "4. CLARIFY - For queries that are too short or vague to understand.\n\n"
        f"User Query: \"{clean_query}\"\n\n"
        "Reply with ONLY one category name: DIRECT, DOCUMENT_SUMMARY, DOCUMENT_QUERY, or CLARIFY."
    )

    response = call_gemini(classification_prompt, api_key=api_key, mock=mock).strip().upper()

    if "SUMMARY" in response:
        return "DOCUMENT_SUMMARY"
    elif "DIRECT" in response:
        return "DIRECT"
    elif "CLARIFY" in response:
        return "CLARIFY"
    else:
        return "DOCUMENT_QUERY"


def run_adaptive_rag(
    query: str,
    chunks: List[Dict[str, Any]],
    chunk_embeddings,
    embedding_model,
    mock: Optional[bool] = None,
    api_key: Optional[str] = None,
    similarity_threshold: float = 0.35,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Orchestrates the Adaptive RAG pipeline.

    Parameters:
        query (str): The user's input question.
        chunks (List[Dict]): Parsed chunks from the uploaded PDF.
        chunk_embeddings: Precomputed NumPy embeddings for all chunks.
        embedding_model: Loaded SentenceTransformer model.
        mock (bool, optional): Whether to use mock mode.
        api_key (str, optional): Gemini API key.
        similarity_threshold (float): Minimum cosine similarity to trust retrieval (default: 0.35).
        top_k (int): Number of top chunks to retrieve.

    Returns:
        Dict[str, Any] with keys:
          - route_taken: 'DIRECT' | 'DOCUMENT_RETRIEVAL' | 'LOW_CONFIDENCE_FALLBACK' | 'CLARIFY'
          - answer: str
          - retrieved_chunks: list of matched chunks with scores
          - confidence_score: float or None
          - reasoning: str
    """
    clean_query = query.strip()
    if not clean_query:
        return {
            "route_taken": "EMPTY_QUERY",
            "answer": "Please type a question to get started.",
            "retrieved_chunks": [],
            "confidence_score": 0.0,
            "reasoning": "The query was blank."
        }

    # Step 1: Classify intent
    route = classify_query(clean_query, api_key=api_key, mock=mock)

    # Path A: Clarification needed
    if route == "CLARIFY":
        return {
            "route_taken": "CLARIFY",
            "answer": "Your question seems a bit brief or ambiguous. Could you please specify more details?",
            "retrieved_chunks": [],
            "confidence_score": 0.0,
            "reasoning": "The router identified this question as vague or incomplete. Asking the user to clarify avoids unnecessary search."
        }

    # Path B: Document Summary / Overview
    if route == "DOCUMENT_SUMMARY":
        if not chunks:
            return {
                "route_taken": "DOCUMENT_SUMMARY",
                "answer": "No PDF has been uploaded yet. Please upload a PDF in the sidebar to get an overview!",
                "retrieved_chunks": [],
                "confidence_score": None,
                "reasoning": "User asked what the document is about, but no PDF is uploaded."
            }

        # For academic papers and reports, the first 2-3 chunks contain the title, abstract, and intro
        summary_chunks = [dict(c) for c in chunks[:min(3, len(chunks))]]
        # Add dummy 100% score for display
        for c in summary_chunks:
            c["score"] = 1.0

        context_text = "\n\n".join([f"--- Section {i+1} ---\n{c['text']}" for i, c in enumerate(summary_chunks)])
        summary_prompt = (
            "You are an expert AI document analyst.\n"
            "Provide a clear overview and summary of this document based on its introductory sections:\n\n"
            f"{context_text}\n\n"
            "Highlight:\n"
            "- What the document/paper is about\n"
            "- Key goals or proposals\n"
            "- Main contributions or findings\n\n"
            "Summary:"
        )
        answer = call_gemini(summary_prompt, api_key=api_key, mock=mock)
        return {
            "route_taken": "DOCUMENT_SUMMARY",
            "answer": answer,
            "retrieved_chunks": summary_chunks,
            "confidence_score": 1.0,
            "reasoning": "Adaptive RAG identified this as a global summary query. Instead of pointwise similarity search, it routed to the title, abstract, and intro chunks to construct a complete document overview."
        }

    # Path C: Direct answer (No retrieval needed)
    if route == "DIRECT" or not chunks:
        reason = (
            "No document has been uploaded yet, so answering directly."
            if not chunks else
            "Identified as general knowledge or greeting. Skipped document retrieval to save compute and prevent noisy context."
        )
        direct_prompt = (
            f"You are a helpful, concise AI assistant.\n"
            f"Answer this user question directly:\n{clean_query}"
        )
        answer = call_gemini(direct_prompt, api_key=api_key, mock=mock)
        return {
            "route_taken": "DIRECT",
            "answer": answer,
            "retrieved_chunks": [],
            "confidence_score": None,
            "reasoning": reason
        }

    # Path D: Specific Document Query -> Perform retrieval
    top_chunks = retrieve_top_k(
        query=clean_query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        model=embedding_model,
        top_k=top_k
    )

    top_score = top_chunks[0]["score"] if top_chunks else 0.0

    # Path D: Fallback if similarity confidence is low
    if top_score < similarity_threshold:
        return {
            "route_taken": "LOW_CONFIDENCE_FALLBACK",
            "answer": (
                f"⚠️ **Information Not Found in Document**\n\n"
                f"I searched your document, but the highest similarity match was only **{top_score * 100:.1f}%** "
                f"(below the confidence threshold of {similarity_threshold * 100:.0f}%).\n\n"
                f"Rather than guessing or hallucinating facts, I suggest checking if your document covers this topic "
                f"or rephrasing your question."
            ),
            "retrieved_chunks": top_chunks,
            "confidence_score": top_score,
            "reasoning": f"Top match similarity ({top_score:.3f}) was below threshold ({similarity_threshold}). Adapted to safe fallback instead of hallucinating."
        }

    # Path E: High-confidence Document Retrieval -> Synthesize Answer
    context_text = ""
    for idx, chunk in enumerate(top_chunks, 1):
        context_text += f"\n--- Context Snippet {idx} (Similarity: {chunk['score'] * 100:.1f}%) ---\n"
        context_text += chunk["text"] + "\n"

    rag_prompt = (
        "You are an assistant answering questions strictly based on the provided document excerpts.\n"
        "Guidelines:\n"
        "1. Answer using ONLY facts mentioned in the context below.\n"
        "2. If the context does not fully answer the question, state what is known and what is missing.\n"
        "3. Keep the answer clear and concise.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Question: {clean_query}\n\n"
        "Answer:"
    )

    answer = call_gemini(rag_prompt, api_key=api_key, mock=mock)

    return {
        "route_taken": "DOCUMENT_RETRIEVAL",
        "answer": answer,
        "retrieved_chunks": top_chunks,
        "confidence_score": top_score,
        "reasoning": f"Successfully retrieved relevant chunks with high similarity ({top_score * 100:.1f}%). Grounded the answer in document facts."
    }
