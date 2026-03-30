"""Health check endpoint."""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "voice-to-pcr-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
