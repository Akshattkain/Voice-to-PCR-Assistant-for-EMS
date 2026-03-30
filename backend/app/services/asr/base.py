"""ASR service abstract base class."""

from abc import ABC, abstractmethod

from app.schemas.transcription import TranscriptionResponse


class ASRService(ABC):
    """Abstract base for speech recognition services."""

    @abstractmethod
    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: str = "en",
    ) -> TranscriptionResponse:
        """Transcribe audio data to text."""
        ...
