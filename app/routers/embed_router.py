from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import json
import logging
from typing import List
import time
import numpy as np

from ..models.song import (
    AudioEmbedRequest,
    LyricsEmbedRequest,
    BatchEmbedRequest,
    EmbedResponse,
    BatchEmbedResponse,
    SongEmbedRequest,
    SongEmbedResponse,
)
from ..services.audio_embedder import get_audio_embedder
from ..services.lyrics_embedder import get_lyrics_embedder
from ..utils.db import get_db, Song
from ..utils.cosine_similarity import cosine_similarity

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embed", tags=["embeddings"])


@router.post("/audio", response_model=EmbedResponse)
async def embed_audio(request: AudioEmbedRequest):
    """Extract audio embedding from audio file URL"""
    try:
        start_time = time.time()

        embedder = get_audio_embedder()
        embedding = embedder.get_embedding(request.audio_url)

        processing_time = time.time() - start_time

        logger.info(f"Audio embedding extracted in {processing_time:.2f}s")

        return EmbedResponse(
            embedding=embedding.tolist(), processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Audio embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio embedding failed: {str(e)}")


@router.post("/lyrics", response_model=EmbedResponse)
async def embed_lyrics(request: LyricsEmbedRequest):
    """Extract lyrics embedding from text"""
    try:
        start_time = time.time()

        embedder = get_lyrics_embedder()
        embedding = embedder.get_embedding(request.lyrics)

        processing_time = time.time() - start_time

        logger.info(f"Lyrics embedding extracted in {processing_time:.2f}s")

        return EmbedResponse(
            embedding=embedding.tolist(), processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Lyrics embedding failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Lyrics embedding failed: {str(e)}"
        )


@router.post("/song", response_model=SongEmbedResponse)
async def embed_song(request: SongEmbedRequest, db: Session = Depends(get_db)):
    """Extract both audio and lyrics embeddings for a song"""
    try:
        start_time = time.time()

        # Lấy bài hát từ cơ sở dữ liệu
        song = db.query(Song).filter(Song.id == request.song_id).first()
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")

        audio_embedding = None
        lyrics_embedding = None

        # Trích xuất embedding âm thanh nếu audio_url tồn tại
        if song.audio_url:
            try:
                audio_embedder = get_audio_embedder()
                audio_embedding = audio_embedder.get_embedding(song.audio_url).tolist()
            except Exception as e:
                logger.warning(
                    f"Audio embedding failed for song {request.song_id}: {e}"
                )

        # Trích xuất embedding lời bài hát nếu lời bài hát tồn tại
        if song.lyrics:
            try:
                lyrics_embedder = get_lyrics_embedder()
                lyrics_embedding = lyrics_embedder.get_embedding(song.lyrics).tolist()
            except Exception as e:
                logger.warning(
                    f"Lyrics embedding failed for song {request.song_id}: {e}"
                )

        processing_time = time.time() - start_time

        logger.info(f"Song embedding extracted in {processing_time:.2f}s")

        return SongEmbedResponse(
            song_id=request.song_id,
            audio_embedding=audio_embedding,
            lyrics_embedding=lyrics_embedding,
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Song embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Song embedding failed: {str(e)}")


@router.post("/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest):
    """Extract embeddings for multiple songs"""
    try:
        start_time = time.time()

        audio_embedder = get_audio_embedder()
        lyrics_embedder = get_lyrics_embedder()

        embeddings = []

        for song_data in request.songs:
            song_start = time.time()

            audio_emb = None
            lyrics_emb = None

            # Trích xuất embedding âm thanh
            if song_data.audio_url:
                try:
                    audio_emb = audio_embedder.get_embedding(
                        song_data.audio_url
                    ).tolist()
                except Exception as e:
                    logger.warning(f"Audio embedding failed for {song_data.title}: {e}")

            # Trích xuất embedding lời bài hát
            if song_data.lyrics:
                try:
                    lyrics_emb = lyrics_embedder.get_embedding(
                        song_data.lyrics
                    ).tolist()
                except Exception as e:
                    logger.warning(
                        f"Lyrics embedding failed for {song_data.title}: {e}"
                    )

            song_time = time.time() - song_start

            embeddings.append(
                EmbedResponse(
                    embedding=audio_emb or lyrics_emb or [], processing_time=song_time
                )
            )

        total_time = time.time() - start_time

        logger.info(
            f"Batch embedding completed for {len(request.songs)} songs in {total_time:.2f}s"
        )

        return BatchEmbedResponse(embeddings=embeddings, total_time=total_time)

    except Exception as e:
        logger.error(f"Batch embedding failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch embedding failed: {str(e)}")


# Các endpoint bổ sung cho khuyến nghị
@router.post("/recommend/song/{song_id}")
async def get_song_recommendations(
    song_id: int, top_k: int = 5, db: Session = Depends(get_db)
):
    """Get song recommendations based on audio and lyrics similarity"""
    try:
        # Lấy bài hát mục tiêu
        target_song = db.query(Song).filter(Song.id == song_id).first()
        if not target_song:
            raise HTTPException(status_code=404, detail="Song not found")

        # Lấy tất cả bài hát có embedding
        all_songs = (
            db.query(Song)
            .filter(Song.audio_vector.isnot(None) | Song.lyric_vector.isnot(None))
            .all()
        )

        if not all_songs:
            return {"recommendations": []}

        # Chuẩn bị các vector mục tiêu
        target_audio = (
            json.loads(target_song.audio_vector) if target_song.audio_vector else None
        )
        target_lyrics = (
            json.loads(target_song.lyric_vector) if target_song.lyric_vector else None
        )

        recommendations = []

        for song in all_songs:
            if song.id == song_id:
                continue

            score = 0.0
            count = 0

            # Độ tương tự âm thanh
            if target_audio and song.audio_vector:
                song_audio = json.loads(song.audio_vector)
                audio_sim = cosine_similarity(
                    np.array(target_audio), np.array(song_audio)
                )
                score += audio_sim
                count += 1

            # Độ tương tự lời bài hát
            if target_lyrics and song.lyric_vector:
                song_lyrics = json.loads(song.lyric_vector)
                lyrics_sim = cosine_similarity(
                    np.array(target_lyrics), np.array(song_lyrics)
                )
                score += lyrics_sim
                count += 1

            if count > 0:
                avg_score = score / count
                recommendations.append({"song_id": song.id, "score": avg_score})

        # Sắp xếp theo điểm giảm dần và lấy top_k
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        recommendations = recommendations[:top_k]

        return {"recommendations": recommendations}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")
