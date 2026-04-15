from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import uvicorn

from .routers.embed_router import router as embed_router
from .utils.db import create_tables

logger = logging.getLogger(__name__)

# Cấu hình ghi log
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Paravibe AI Service",
    description="AI service for music recommendations using MERT and SBERT models",
    version="1.0.0",
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cấu hình phù hợp cho sản xuất
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bao gồm các router
app.include_router(embed_router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Create tables on startup"""
    try:
        create_tables()
        logger.info("Database tables created/verified successfully")
    except Exception as e:
        logger.warning(f"Could not create/verify database tables: {e}")


@app.get("/")
def root():
    return {"message": "Paravibe AI Service running"}


@app.get("/api/v1/health")
def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
