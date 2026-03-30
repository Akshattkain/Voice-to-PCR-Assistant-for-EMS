"""Top-level API router aggregator."""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    correction,
    evaluation,
    extraction,
    gap_detection,
    health,
    pcr_export,
    sessions,
    transcription,
)
from app.api.v1.websocket import realtime as ws_realtime

api_router = APIRouter()

# Health check at root level
api_router.include_router(health.router, tags=["health"])

# V1 API endpoints
api_router.include_router(sessions.router, prefix="/api/v1")
api_router.include_router(transcription.router, prefix="/api/v1")
api_router.include_router(extraction.router, prefix="/api/v1")
api_router.include_router(correction.router, prefix="/api/v1")
api_router.include_router(gap_detection.router, prefix="/api/v1")
api_router.include_router(pcr_export.router, prefix="/api/v1")
api_router.include_router(evaluation.router, prefix="/api/v1")

# WebSocket
api_router.include_router(ws_realtime.router)
