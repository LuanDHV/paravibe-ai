# Các mô hình Pydantic cho User và UserHistory
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UserBase(BaseModel):
    name: str
    email: str


class UserCreate(UserBase):
    password_hash: str


class User(UserBase):
    user_id: int
    role_id: Optional[int] = None
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class UserHistoryBase(BaseModel):
    user_id: int
    song_id: int
    action: Optional[str] = None  # PLAY, SKIP, LIKE


class UserHistoryCreate(UserHistoryBase):
    pass


class UserHistory(UserHistoryBase):
    history_id: int
    listened_at: Optional[datetime] = None
    duration_listened: Optional[int] = None

    class Config:
        from_attributes = True


# Response models cho recommendation
class RecommendationItem(BaseModel):
    song_id: int
    title: str
    artist: str
    score: float


class UserRecommendationResponse(BaseModel):
    user: dict
    recommendations: List[RecommendationItem]
    processing_time: float
    listened_songs_count: Optional[int] = None
    message: Optional[str] = None


class UserTopSongsResponse(BaseModel):
    song_id: int
    title: str
    artist: str
    listened_at: Optional[str] = None
    action: Optional[str] = None
