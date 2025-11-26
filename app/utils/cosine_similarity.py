import numpy as np
from typing import List, Tuple
import heapq


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_top_similar(
    query_vector: np.ndarray,
    candidate_vectors: List[np.ndarray],
    candidate_ids: List[int],
    top_k: int = 5,
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar vectors to query vector

    Args:
        query_vector: The query embedding vector
        candidate_vectors: List of candidate embedding vectors
        candidate_ids: Corresponding IDs for candidates
        top_k: Number of top results to return

    Returns:
        List of (id, similarity_score) tuples, sorted by similarity descending
    """
    similarities = []

    for vec, idx in zip(candidate_vectors, candidate_ids):
        sim = cosine_similarity(query_vector, vec)
        similarities.append((idx, sim))

    # Get top-k using heap
    top_similar = heapq.nlargest(top_k, similarities, key=lambda x: x[1])

    return top_similar
