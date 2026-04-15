import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModel
import logging
from typing import List, Optional
import time
from ..utils.download_audio import download_audio_from_url, cleanup_temp_audio

logger = logging.getLogger(__name__)


class AudioEmbedder:
    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Tải mô hình MERT và bộ xử lý"""
        try:
            logger.info(f"Loading MERT model: {self.model_name}")
            start_time = time.time()

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.to(self.device)
            self.model.eval()

            load_time = time.time() - start_time
            logger.info(".2f")
        except Exception as e:
            logger.error(f"Failed to load MERT model: {e}")
            raise

    def preprocess_audio(self, audio_path: str, target_sr: int = 24000) -> torch.Tensor:
        """Tải và tiền xử lý file âm thanh"""
        try:
            # Tải âm thanh với ffmpeg backend để hỗ trợ M4A/AAC
            # sr=None để tải sample rate gốc, sau đó resample
            audio, sr = librosa.load(audio_path, sr=None, mono=True)

            # Resample nếu cần
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Chuyển đổi thành tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Xử lý với bộ xử lý MERT
            inputs = self.processor(
                audio_tensor, sampling_rate=target_sr, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            return inputs
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """Trích xuất embedding 768 chiều từ file âm thanh"""
        try:
            start_time = time.time()

            # Download audio if URL
            local_path = audio_path
            if audio_path.startswith(("http://", "https://")):
                local_path = download_audio_from_url(audio_path)
                if local_path is None:
                    raise ValueError(f"Failed to download audio from {audio_path}")

            # Tiền xử lý âm thanh
            inputs = self.preprocess_audio(local_path)

            # Lấy embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Sử dụng mean pooling của trạng thái ẩn cuối cùng
                embedding = (
                    outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                )

            processing_time = time.time() - start_time
            logger.info(f"Audio embedding extracted in {processing_time:.2f}s")

            # Cleanup temporary file if it was downloaded
            if audio_path.startswith(("http://", "https://")):
                cleanup_temp_audio(local_path)

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {e}")
            raise

    def get_embeddings_batch(self, audio_paths: List[str]) -> List[np.ndarray]:
        """Trích xuất embeddings cho nhiều file âm thanh"""
        embeddings = []
        for path in audio_paths:
            try:
                embedding = self.get_embedding(path)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                embeddings.append(None)
        return embeddings


# Instance toàn cục để lưu cache
_audio_embedder: Optional[AudioEmbedder] = None


def get_audio_embedder() -> AudioEmbedder:
    """Lấy hoặc tạo instance AudioEmbedder đã được cache"""
    global _audio_embedder
    if _audio_embedder is None:
        _audio_embedder = AudioEmbedder()
    return _audio_embedder
