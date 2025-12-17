from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from typing import List, Optional
import time

logger = logging.getLogger(__name__)


class LyricsEmbedder:
    """
    Embedder cho text sử dụng mô hình SBERT.
    Có thể dùng cho lyrics hoặc metadata text.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Tải mô hình SBERT"""
        try:
            logger.info(f"Loading SBERT model: {self.model_name}")
            start_time = time.time()

            self.model = SentenceTransformer(self.model_name)

            load_time = time.time() - start_time
            logger.info(".2f")
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """Trích xuất embedding 384 chiều từ text"""
        try:
            start_time = time.time()

            # Lấy embedding
            embedding = self.model.encode(text, convert_to_numpy=True)

            processing_time = time.time() - start_time
            logger.info(".2f")

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Error extracting embedding from text: {e}")
            raise

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Trích xuất embeddings cho nhiều text"""
        try:
            start_time = time.time()

            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)

            processing_time = time.time() - start_time
            logger.info(f"Batch processed {len(texts)} texts in {processing_time:.2f}s")

            return [emb.astype(np.float32) for emb in embeddings]

        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise


# Instance toàn cục để lưu cache
_metadata_embedder: Optional[LyricsEmbedder] = None


def get_metadata_embedder() -> LyricsEmbedder:
    """Lấy hoặc tạo instance MetadataEmbedder đã được cache"""
    global _metadata_embedder
    if _metadata_embedder is None:
        _metadata_embedder = LyricsEmbedder()
    return _metadata_embedder
