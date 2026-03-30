"""FastAPI dependency providers — singleton service instances."""

from app.config import settings
from app.core.gap_detector import GapDetector
from app.core.session_manager import SessionManager
from app.services.asr.base import ASRService
from app.services.asr.whisper_api import WhisperAPIService
from app.services.extraction.base import ExtractionService
from app.services.extraction.llm_baseline_extractor import LLMBaselineExtractor
from app.services.llm.openai_client import OpenAIClient
from app.utils.logging import logger


# Singletons
_session_manager: SessionManager | None = None
_gap_detector: GapDetector | None = None
_openai_client: OpenAIClient | None = None
_asr_service: ASRService | None = None
_llm_baseline_extractor: ExtractionService | None = None
_finetuned_extractor: ExtractionService | None = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_gap_detector() -> GapDetector:
    global _gap_detector
    if _gap_detector is None:
        _gap_detector = GapDetector()
    return _gap_detector


def get_openai_client() -> OpenAIClient:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client


def get_asr_service() -> ASRService:
    global _asr_service
    if _asr_service is None:
        _asr_service = WhisperAPIService(openai_client=get_openai_client())
    return _asr_service


def get_extraction_service(model: str = "") -> ExtractionService:
    """Get the extraction service for the specified model type."""
    model = model or settings.extraction_model_default

    if model == "llm_baseline":
        return _get_llm_baseline_extractor()
    elif model == "finetuned":
        return _get_finetuned_extractor()
    else:
        raise ValueError(f"Unknown extraction model: {model}")


def _get_llm_baseline_extractor() -> ExtractionService:
    global _llm_baseline_extractor
    if _llm_baseline_extractor is None:
        _llm_baseline_extractor = LLMBaselineExtractor(
            openai_client=get_openai_client(),
            model=settings.gpt_model,
        )
    return _llm_baseline_extractor


def _get_finetuned_extractor() -> ExtractionService:
    global _finetuned_extractor
    if _finetuned_extractor is None:
        from app.services.extraction.finetuned_extractor import FineTunedExtractor

        _finetuned_extractor = FineTunedExtractor(
            model_path=settings.finetuned_model_path,
            device=settings.finetuned_model_device,
        )
    return _finetuned_extractor
