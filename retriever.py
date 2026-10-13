"""
retriever.py
------------
Handles semantic search:
1. Generates numerical embeddings (vectors) using Sentence Transformers.
2. Calculates cosine similarity with pure NumPy (no ChromaDB or vector database needed).
3. Retrieves the top-K most relevant chunks for a given query.
"""

from typing import List, Dict, Any
import numpy as np


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Loads a lightweight, local SentenceTransformer model.

    Parameters:
        model_name (str): HuggingFace model name (default: all-MiniLM-L6-v2).

    Returns:
        SentenceTransformer: The loaded model instance, or None if not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception as e:
        # Fallback if sentence_transformers is not yet installed
        return None


def get_embeddings(texts: List[str], model=None) -> np.ndarray:
    """
    Converts a list of text strings into an array of embeddings.
    If no model is loaded (e.g. testing in mock mode without heavy downloads),
    it generates deterministic pseudo-vectors using word hash frequencies.

    Parameters:
        texts (List[str]): List of chunk strings or queries.
        model: Loaded SentenceTransformer model.

    Returns:
        np.ndarray: 2D NumPy array where each row is an embedding vector.
    """
    if not texts:
        return np.array([])

    if model is not None:
        embeddings = model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)

    # Lightweight fallback for Mock Mode / offline testing:
    # Generates deterministic 64-dimensional pseudo-vectors based on word hashes
    pseudo_vectors = []
    dim = 64
    for text in texts:
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for w in words:
            # Map word to a bucket
            idx = abs(hash(w)) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        pseudo_vectors.append(vec)

    return np.array(pseudo_vectors, dtype=np.float32)


def calculate_cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D NumPy vectors using basic math:
        similarity = dot_product(a, b) / (||a|| * ||b||)

    Score ranges from -1.0 to 1.0:
      - 1.0 means identical direction (high semantic match).
      - 0.0 means completely orthogonal (no match).

    Parameters:
        vector_a (np.ndarray): First embedding vector.
        vector_b (np.ndarray): Second embedding vector.

    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    dot_product = float(np.dot(vector_a, vector_b))
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    # Clamp between 0.0 and 1.0 for clean percentage display
    return max(0.0, min(1.0, float(similarity)))


def retrieve_top_k(
    query: str,
    chunks: List[Dict[str, Any]],
    chunk_embeddings: np.ndarray,
    model=None,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Finds the top-K chunks most relevant to the user query.

    Parameters:
        query (str): The user's question.
        chunks (List[Dict]): List of chunk dictionaries.
        chunk_embeddings (np.ndarray): 2D array of chunk embeddings.
        model: Embedding model.
        top_k (int): Number of top matches to return.

    Returns:
        List[Dict[str, Any]]: Top chunk dicts sorted by score descending,
                              with an added 'score' field.
    """
    if not chunks or len(chunk_embeddings) == 0:
        return []

    # 1. Generate embedding for the user query
    query_vector = get_embeddings([query], model)[0]

    # 2. Calculate similarity between query and each chunk
    scored_chunks = []
    for i in range(len(chunks)):
        score = calculate_cosine_similarity(query_vector, chunk_embeddings[i])
        chunk_item = dict(chunks[i])  # Make a shallow copy
        chunk_item["score"] = round(score, 4)
        scored_chunks.append(chunk_item)

    # 3. Sort by score in descending order (highest score first)
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    # 4. Return top-K matches
    return scored_chunks[:top_k]
