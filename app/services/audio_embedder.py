import torch
import torchaudio
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModel
import logging
from typing import List, Optional
import time

logger = logging.getLogger(__name__)


class AudioEmbedder:
    def __init__(self, model_name: str = "m-a-p/MERT-v1-95M"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load MERT model and processor"""
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
        """Load and preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)

            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Process with MERT processor
            inputs = self.processor(
                audio_tensor, sampling_rate=target_sr, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            return inputs
        except Exception as e:
            logger.error(f"Error preprocessing audio {audio_path}: {e}")
            raise

    def get_embedding(self, audio_path: str) -> np.ndarray:
        """Extract 768-dim embedding from audio file"""
        try:
            start_time = time.time()

            # Preprocess audio
            inputs = self.preprocess_audio(audio_path)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden state
                embedding = (
                    outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                )

            processing_time = time.time() - start_time
            logger.info(".2f")

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Error extracting embedding from {audio_path}: {e}")
            raise

    def get_embeddings_batch(self, audio_paths: List[str]) -> List[np.ndarray]:
        """Extract embeddings for multiple audio files"""
        embeddings = []
        for path in audio_paths:
            try:
                embedding = self.get_embedding(path)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                embeddings.append(None)
        return embeddings


# Global instance for caching
_audio_embedder: Optional[AudioEmbedder] = None


def get_audio_embedder() -> AudioEmbedder:
    """Get or create cached AudioEmbedder instance"""
    global _audio_embedder
    if _audio_embedder is None:
        _audio_embedder = AudioEmbedder()
    return _audio_embedder
