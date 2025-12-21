from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import json
import logging
from typing import List
import time
import numpy as np

from ..models.song import (
    AudioEmbedRequest,
    MetadataEmbedRequest,
    BatchEmbedRequest,
    EmbedResponse,
    BatchEmbedResponse,
    SongEmbedRequest,
    SongEmbedResponse,
)
from ..models.user import (
    UserRecommendationResponse,
    UserTopSongsResponse,
)
from ..services.audio_embedder import get_audio_embedder
from ..services.metadata_embedder import get_metadata_embedder
from ..utils.db import get_db, Song, User, UserHistory
from ..utils.cosine_similarity import cosine_similarity
from sqlalchemy.orm import joinedload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embed", tags=["embeddings"])


@router.post("/audio", response_model=EmbedResponse)
async def embed_audio(request: AudioEmbedRequest):
    """Trích xuất embedding âm thanh từ URL file âm thanh"""
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


@router.post("/metadata", response_model=EmbedResponse)
async def embed_metadata(request: MetadataEmbedRequest):
    """Trích xuất embedding metadata từ thông tin bài hát (không phụ thuộc lyrics)"""
    try:
        start_time = time.time()

        # Xây dựng văn bản metadata từ thông tin bài hát
        metadata_text = f"""
        Tiêu đề: {request.title}
        Nghệ sĩ: {request.artist}
        Thể loại: {request.genre or 'Không xác định'}
        Album: {request.album or 'Không xác định'}
        Năm phát hành: {request.release_year or 'Không xác định'}
        """.strip()

        # Sử dụng metadata_embedder để xử lý metadata text
        embedder = get_metadata_embedder()
        embedding = embedder.get_embedding(metadata_text)

        processing_time = time.time() - start_time

        logger.info(f"Metadata embedding extracted in {processing_time:.2f}s")

        return EmbedResponse(
            embedding=embedding.tolist(), processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Metadata embedding failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Metadata embedding failed: {str(e)}"
        )


@router.post("/song", response_model=SongEmbedResponse)
async def embed_song(request: SongEmbedRequest, db: Session = Depends(get_db)):
    """Trích xuất embedding âm thanh và metadata cho bài hát và lưu vào DB"""
    try:
        start_time = time.time()

        # Lấy bài hát từ cơ sở dữ liệu
        song = db.query(Song).filter(Song.song_id == request.song_id).first()
        if not song:
            raise HTTPException(status_code=404, detail="Song not found")

        audio_embedding = None
        metadata_embedding = None

        # Trích xuất embedding âm thanh nếu audio_url tồn tại
        if song.audio_url:
            try:
                audio_embedder = get_audio_embedder()
                audio_embedding = audio_embedder.get_embedding(song.audio_url)
                # Lưu vào DB dưới dạng JSON
                song.audio_vector = json.dumps(audio_embedding.tolist())
            except Exception as e:
                logger.warning(
                    f"Audio embedding failed for song {request.song_id}: {e}"
                )

        # Trích xuất embedding metadata từ thông tin bài hát
        try:
            metadata_text = f"""
            Tiêu đề: {song.title}
            Nghệ sĩ: {song.artist if hasattr(song, 'artist') else 'Không xác định'}
            Thể loại: {song.genre or 'Không xác định'}
            """.strip()

            metadata_embedder = get_metadata_embedder()
            metadata_embedding = metadata_embedder.get_embedding(metadata_text)
            # Lưu vào DB dưới dạng JSON
            song.metadata_vector = json.dumps(metadata_embedding.tolist())
        except Exception as e:
            logger.warning(f"Metadata embedding failed for song {request.song_id}: {e}")

        # Lưu vào database
        if audio_embedding is not None or metadata_embedding is not None:
            db.commit()
            logger.info(f"Song embeddings saved to DB for song {request.song_id}")

        processing_time = time.time() - start_time

        logger.info(f"Song embedding extracted and saved in {processing_time:.2f}s")

        return SongEmbedResponse(
            song_id=request.song_id,
            audio_embedding=(
                audio_embedding.tolist() if audio_embedding is not None else None
            ),
            metadata_embedding=(
                metadata_embedding.tolist() if metadata_embedding is not None else None
            ),
            processing_time=processing_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Song embedding failed: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Song embedding failed: {str(e)}")


@router.post("/batch", response_model=BatchEmbedResponse)
async def embed_batch(request: BatchEmbedRequest):
    """Trích xuất embeddings cho nhiều bài hát"""
    try:
        start_time = time.time()

        audio_embedder = get_audio_embedder()
        metadata_embedder = get_metadata_embedder()

        embeddings = []

        for song_data in request.songs:
            song_start = time.time()

            audio_emb = None
            metadata_emb = None

            # Trích xuất embedding âm thanh
            if song_data.audio_url:
                try:
                    audio_emb = audio_embedder.get_embedding(
                        song_data.audio_url
                    ).tolist()
                except Exception as e:
                    logger.warning(f"Audio embedding failed for {song_data.title}: {e}")

            # Trích xuất embedding metadata
            try:
                metadata_text = f"""
            Tiêu đề: {song_data.title}
            Nghệ sĩ: {song_data.artist}
            Thể loại: {song_data.genre or 'Không xác định'}
                """.strip()
                metadata_emb = metadata_embedder.get_embedding(metadata_text).tolist()
            except Exception as e:
                logger.warning(f"Metadata embedding failed for {song_data.title}: {e}")

            song_time = time.time() - song_start

            embeddings.append(
                EmbedResponse(
                    embedding=audio_emb or metadata_emb or [], processing_time=song_time
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
@router.get("/recommend/song/{song_id}")
async def get_song_recommendations(
    song_id: int, top_k: int = 5, db: Session = Depends(get_db)
):
    """Lấy khuyến nghị bài hát dựa trên độ tương tự âm thanh và metadata"""
    try:
        start_time = time.time()

        # Lấy bài hát mục tiêu với artist relationship
        target_song = (
            db.query(Song)
            .options(joinedload(Song.artist_rel))
            .filter(Song.song_id == song_id)
            .first()
        )
        if not target_song:
            raise HTTPException(status_code=404, detail="Song not found")

        # Lấy tất cả bài hát có embedding với artist relationship
        all_songs = (
            db.query(Song)
            .options(joinedload(Song.artist_rel))
            .filter(Song.audio_vector.isnot(None) | Song.metadata_vector.isnot(None))
            .all()
        )

        if not all_songs:
            return {
                "song": {
                    "song_id": target_song.id,
                    "title": target_song.title,
                    "artist": target_song.artist if target_song.artist else "Unknown",
                },
                "recommendations": [],
                "processing_time": 0,
            }

        # Chuẩn bị các vector mục tiêu
        target_audio = (
            np.array(json.loads(target_song.audio_vector))
            if target_song.audio_vector
            else None
        )
        target_metadata = (
            np.array(json.loads(target_song.metadata_vector))
            if target_song.metadata_vector
            else None
        )

        recommendations = []

        for song in all_songs:
            if song.song_id == song_id:
                continue

            # Độ tương tự âm thanh
            audio_sim = 0.0
            if target_audio is not None and song.audio_vector:
                try:
                    song_audio = np.array(json.loads(song.audio_vector))
                    audio_sim = cosine_similarity(target_audio, song_audio)
                except Exception as e:
                    logger.warning(f"Error comparing audio vectors: {e}")

            # Độ tương tự metadata
            metadata_sim = 0.0
            if target_metadata is not None and song.metadata_vector:
                try:
                    song_metadata = np.array(json.loads(song.metadata_vector))
                    metadata_sim = cosine_similarity(target_metadata, song_metadata)
                except Exception as e:
                    logger.warning(f"Error comparing metadata vectors: {e}")

            # Tính weighted score (60% audio, 40% metadata)
            weighted_score = (audio_sim * 0.6) + (metadata_sim * 0.4)

            # Thêm vào recommendations
            recommendations.append(
                {
                    "song_id": song.song_id,
                    "title": song.title,
                    "artist": song.artist if song.artist else "Unknown",
                    "score": float(weighted_score),
                }
            )

        # Sắp xếp theo điểm giảm dần và lấy top_k
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        recommendations = recommendations[:top_k]

        processing_time = time.time() - start_time

        return {
            "song": {
                "song_id": target_song.song_id,
                "title": target_song.title,
                "artist": target_song.artist if target_song.artist else "Unknown",
            },
            "recommendations": recommendations,
            "processing_time": processing_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


# User-based recommendations endpoints
@router.get("/recommend/user/{user_id}")
async def get_user_recommendations(
    user_id: int, top_k: int = 10, db: Session = Depends(get_db)
):
    """
    Lấy khuyến nghị bài hát dựa trên lịch sử nghe nhạc của user

    Algorithm:
    1. Lấy top 20 bài hát user đã nghe gần đây
    2. Tính average audio vector và metadata vector
    3. So sánh với tất cả bài hát trong database
    4. Return top_k recommendations (không bao gồm bài hát đã nghe)
    """
    try:
        start_time = time.time()

        # Kiểm tra user tồn tại
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Lấy top 20 bài hát user đã nghe gần đây
        user_history = (
            db.query(UserHistory)
            .filter(UserHistory.user_id == user_id)
            .order_by(UserHistory.listened_at.desc())
            .limit(20)
            .all()
        )

        if not user_history:
            return {
                "user": {
                    "user_id": user.user_id,
                    "name": user.name or "Unknown",
                },
                "recommendations": [],
                "processing_time": 0,
                "message": "User has no listening history",
            }

        # Lấy embeddings từ các bài hát đã nghe
        audio_vectors = []
        metadata_vectors = []

        for history in user_history:
            song = (
                db.query(Song)
                .options(joinedload(Song.artist_rel))
                .filter(Song.song_id == history.song_id)
                .first()
            )
            if song:
                if song.audio_vector:
                    audio_vectors.append(np.array(json.loads(song.audio_vector)))
                if song.metadata_vector:
                    metadata_vectors.append(np.array(json.loads(song.metadata_vector)))

        if not audio_vectors and not metadata_vectors:
            return {
                "user": {
                    "user_id": user.user_id,
                    "name": user.name or "Unknown",
                },
                "recommendations": [],
                "processing_time": 0,
                "message": "User's history has no embeddings",
            }

        # Tính average vector (user preference)
        user_audio_pref = None
        user_metadata_pref = None

        if audio_vectors:
            user_audio_pref = np.mean(audio_vectors, axis=0)

        if metadata_vectors:
            user_metadata_pref = np.mean(metadata_vectors, axis=0)

        # Lấy tất cả bài hát có embedding
        all_songs = (
            db.query(Song)
            .options(joinedload(Song.artist_rel))
            .filter(Song.audio_vector.isnot(None) | Song.metadata_vector.isnot(None))
            .all()
        )

        if not all_songs:
            return {
                "user": {
                    "user_id": user.user_id,
                    "name": user.name or "Unknown",
                },
                "recommendations": [],
                "processing_time": 0,
            }

        recommendations = []
        listened_song_ids = {h.song_id for h in user_history}

        for song in all_songs:
            # Không recommend bài hát user đã nghe
            if song.song_id in listened_song_ids:
                continue

            # Độ tương tự âm thanh
            audio_sim = 0.0
            if user_audio_pref is not None and song.audio_vector:
                try:
                    song_audio = np.array(json.loads(song.audio_vector))
                    audio_sim = cosine_similarity(user_audio_pref, song_audio)
                except Exception as e:
                    logger.warning(f"Error comparing audio vectors: {e}")

            # Độ tương tự metadata
            metadata_sim = 0.0
            if user_metadata_pref is not None and song.metadata_vector:
                try:
                    song_metadata = np.array(json.loads(song.metadata_vector))
                    metadata_sim = cosine_similarity(user_metadata_pref, song_metadata)
                except Exception as e:
                    logger.warning(f"Error comparing metadata vectors: {e}")

            # Tính weighted score (60% audio, 40% metadata)
            weighted_score = (audio_sim * 0.6) + (metadata_sim * 0.4)

            # Thêm vào recommendations
            if weighted_score > 0:
                recommendations.append(
                    {
                        "song_id": song.song_id,
                        "title": song.title,
                        "artist": song.artist if song.artist else "Unknown",
                        "score": float(weighted_score),
                    }
                )

        # Sắp xếp theo điểm giảm dần và lấy top_k
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        recommendations = recommendations[:top_k]

        processing_time = time.time() - start_time

        return {
            "user": {
                "user_id": user.user_id,
                "name": user.name or "Unknown",
            },
            "recommendations": recommendations,
            "processing_time": processing_time,
            "listened_songs_count": len(user_history),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User recommendation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/recommend/user/{user_id}/top-songs")
async def get_user_top_songs(
    user_id: int, limit: int = 10, db: Session = Depends(get_db)
):
    """
    Lấy top songs mà user đã nghe nhiều nhất
    Hữu ích để debug user preference
    """
    try:
        # Kiểm tra user tồn tại
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Lấy top songs user đã nghe
        user_history = (
            db.query(UserHistory)
            .options(joinedload(UserHistory.song).joinedload(Song.artist_rel))
            .filter(UserHistory.user_id == user_id)
            .order_by(UserHistory.listened_at.desc())
            .limit(limit)
            .all()
        )

        top_songs = []
        for history in user_history:
            song = history.song
            if song:
                top_songs.append(
                    {
                        "song_id": song.song_id,
                        "title": song.title,
                        "artist": song.artist if song.artist else "Unknown",
                        "listened_at": (
                            history.listened_at.isoformat()
                            if history.listened_at
                            else None
                        ),
                        "action": history.action,
                    }
                )

        return top_songs

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get user top songs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")
