"""Whisper API-based ASR implementation."""

from app.schemas.transcription import TranscriptionResponse, TranscriptionSegment
from app.services.asr.base import ASRService
from app.services.llm.openai_client import OpenAIClient


class WhisperAPIService(ASRService):
    """OpenAI Whisper API speech recognition."""

    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client

    async def transcribe(
        self,
        audio_data: bytes,
        audio_format: str = "wav",
        language: str = "en",
    ) -> TranscriptionResponse:
        """Transcribe audio via OpenAI Whisper API."""
        filename = f"audio.{audio_format}"

        result = await self.openai_client.transcribe_audio(
            audio_data=audio_data,
            filename=filename,
            language=language,
            response_format="verbose_json",
        )

        # Parse segments from Whisper verbose_json response
        segments = []
        if hasattr(result, "segments") and result.segments:
            for seg in result.segments:
                segments.append(
                    TranscriptionSegment(
                        text=seg.get("text", "") if isinstance(seg, dict) else seg.text,
                        start=seg.get("start", 0.0) if isinstance(seg, dict) else seg.start,
                        end=seg.get("end", 0.0) if isinstance(seg, dict) else seg.end,
                    )
                )

        transcript_text = result.text if hasattr(result, "text") else str(result)
        duration = result.duration if hasattr(result, "duration") else None

        return TranscriptionResponse(
            transcript_text=transcript_text,
            segments=segments,
            language=language,
            duration_sec=duration,
            model_used="whisper-1",
        )
