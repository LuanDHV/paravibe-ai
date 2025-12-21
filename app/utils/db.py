from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv(".env.local")

# Cấu hình cơ sở dữ liệu
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Artist(Base):
    __tablename__ = "Artist"

    artist_id = Column("artist_id", Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=True)
    country = Column(String(255), nullable=True)
    bio = Column(Text, nullable=True)
    genre = Column(String(100), nullable=True)
    image_url = Column(String(500), nullable=True)

    # Relationship với Song
    songs = relationship("Song", back_populates="artist_rel")


class Song(Base):
    __tablename__ = "Song"

    song_id = Column("song_id", Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=True)
    artist_id = Column(
        "artist_id", Integer, ForeignKey("Artist.artist_id"), nullable=True
    )
    duration = Column(Integer, nullable=True)
    release_date = Column(String(50), nullable=True)  # date stored as string
    image_url = Column(String(500), nullable=True)
    audio_url = Column(String(500), nullable=True)
    preview_url = Column(String(255), nullable=True)
    genre = Column(String(100), nullable=True)
    audio_vector = Column(Text, nullable=True)  # JSON của audio embeddings
    metadata_vector = Column(Text, nullable=True)  # JSON của metadata embeddings
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship với Artist
    artist_rel = relationship("Artist", back_populates="songs")

    # Alias để tương thích với code cũ
    @property
    def id(self):
        return self.song_id

    @property
    def artist(self):
        """Lấy tên nghệ sĩ từ relationship"""
        return self.artist_rel.name if self.artist_rel else "Unknown Artist"


def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables"""
    Base.metadata.create_all(bind=engine)
