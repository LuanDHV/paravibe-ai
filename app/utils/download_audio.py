import requests
import os
import tempfile
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def download_audio_from_url(audio_url: str, timeout: int = 30) -> Optional[str]:
    """
    Tải file âm thanh từ URL và lưu vào vị trí tạm thời

    Args:
        audio_url: URL của file âm thanh (HTTP/HTTPS)
        timeout: Thời gian chờ request timeout tính bằng giây

    Returns:
        Đường dẫn đến file âm thanh tạm thời, hoặc None nếu tải thất bại
    """
    try:
        # Kiểm tra xem có phải đường dẫn file local không
        if audio_url.startswith(("http://", "https://")):
            logger.info(f"Downloading audio from URL: {audio_url}")

            # Tải file
            response = requests.get(audio_url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Phát hiện phần mở rộng file từ URL trước (đáng tin cậy hơn cho file M4A của iTunes)
            ext = ".mp3"  # mặc định fallback

            # Kiểm tra URL cho các phần mở rộng âm thanh phổ biến
            if ".m4a" in audio_url.lower():
                ext = ".m4a"
            elif ".aac" in audio_url.lower():
                ext = ".m4a"  # File AAC thường dùng container M4A
            elif ".wav" in audio_url.lower():
                ext = ".wav"
            elif ".ogg" in audio_url.lower():
                ext = ".ogg"
            elif ".flac" in audio_url.lower():
                ext = ".flac"
            elif ".mp3" in audio_url.lower():
                ext = ".mp3"
            else:
                # Fallback để phát hiện từ content-type
                content_type = response.headers.get("content-type", "audio/mpeg")
                if "wav" in content_type:
                    ext = ".wav"
                elif "ogg" in content_type:
                    ext = ".ogg"
                elif "flac" in content_type:
                    ext = ".flac"
                elif "aac" in content_type or "m4a" in content_type:
                    ext = ".m4a"

            # Tạo file tạm thời
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"audio_{os.urandom(8).hex()}{ext}")

            # Ghi vào file tạm thời
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logger.info(f"Audio downloaded successfully: {temp_file}")
            return temp_file

        else:
            # Đây là đường dẫn file local
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


def cleanup_temp_audio(file_path: Optional[str]) -> None:
    """
    Dọn dẹp file âm thanh tạm thời

    Args:
        file_path: Đường dẫn đến file tạm thời cần xóa
    """
    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {file_path}: {e}")
