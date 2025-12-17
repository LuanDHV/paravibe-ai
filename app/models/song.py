# Các mô hình Pydantic cho API
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class SongBase(BaseModel):
    title: str
    artist: str
    duration: Optional[float] = None
    release_date: Optional[datetime] = None
    image_url: Optional[str] = None
    audio_url: Optional[str] = None
    genre: Optional[str] = None


class SongCreate(SongBase):
    pass


class SongUpdate(SongBase):
    pass


class Song(SongBase):
    id: int
    audio_vector: Optional[List[float]] = None
    metadata_vector: Optional[List[float]] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# Các mô hình yêu cầu/phản hồi embedding
class AudioEmbedRequest(BaseModel):
    audio_url: str


class MetadataEmbedRequest(BaseModel):
    """Trích xuất embedding từ metadata bài hát (không phụ thuộc lyrics)"""

    title: str
    artist: str
    genre: Optional[str] = None
    album: Optional[str] = None
    release_year: Optional[int] = None


class BatchEmbedRequest(BaseModel):
    songs: List[SongCreate]


class EmbedResponse(BaseModel):
    embedding: List[float]
    processing_time: float


class BatchEmbedResponse(BaseModel):
    embeddings: List[Optional[EmbedResponse]]
    total_time: float


class SongEmbedRequest(BaseModel):
    song_id: int


class SongEmbedResponse(BaseModel):
    song_id: int
    audio_embedding: Optional[List[float]] = None
    metadata_embedding: Optional[List[float]] = None
    processing_time: float
