import requests
import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def download_audio_from_url(audio_url: str, timeout: int = 30) -> Optional[str]:
    """
    Download audio file from URL and save to temporary location

    Args:
        audio_url: URL of the audio file (HTTP/HTTPS)
        timeout: Request timeout in seconds

    Returns:
        Path to temporary audio file, or None if download fails
    """
    try:
        # Check if it's a local file path
        if audio_url.startswith(("http://", "https://")):
            logger.info(f"Downloading audio from URL: {audio_url}")

            # Download file
            response = requests.get(audio_url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Get file extension from URL or use mp3 as default
            content_type = response.headers.get("content-type", "audio/mpeg")
            ext = ".mp3"
            if "wav" in content_type:
                ext = ".wav"
            elif "ogg" in content_type:
                ext = ".ogg"
            elif "flac" in content_type:
                ext = ".flac"

            # Create temporary file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"audio_{os.urandom(8).hex()}{ext}")

            # Write to temporary file
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Audio downloaded successfully: {temp_file}")
            return temp_file

        else:
            # It's a local file path
            if os.path.exists(audio_url):
                logger.info(f"Using local audio file: {audio_url}")
                return audio_url
            else:
                logger.error(f"Local audio file not found: {audio_url}")
                return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download audio from {audio_url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading audio: {e}")
        return None


def cleanup_temp_audio(file_path: str) -> bool:
    """
    Clean up temporary audio file

    Args:
        file_path: Path to temporary audio file

    Returns:
        True if successfully deleted, False otherwise
    """
    try:
        if os.path.exists(file_path) and os.path.basename(file_path).startswith(
            "audio_"
        ):
            os.remove(file_path)
            logger.info(f"Temporary audio file cleaned up: {file_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to cleanup temporary audio file {file_path}: {e}")
        return False
