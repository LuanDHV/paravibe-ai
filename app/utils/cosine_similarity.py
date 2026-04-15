import numpy as np
from typing import List, Tuple
import heapq


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Tính độ tương tự cosine giữa hai vector"""
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
    Tìm top-k vector tương tự nhất với query vector

    Args:
        query_vector: Vector embedding truy vấn
        candidate_vectors: Danh sách các vector ứng viên
        candidate_ids: ID tương ứng cho từng ứng viên
        top_k: Số kết quả hàng đầu cần trả về

    Returns:
        Danh sách tuple (id, similarity_score), sắp xếp giảm dần theo độ tương tự
    """
    similarities = []

    for vec, idx in zip(candidate_vectors, candidate_ids):
        sim = cosine_similarity(query_vector, vec)
        similarities.append((idx, sim))

    # Lấy top-k sử dụng heap
    top_similar = heapq.nlargest(top_k, similarities, key=lambda x: x[1])

    return top_similar
