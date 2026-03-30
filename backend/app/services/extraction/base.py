"""Extraction service abstract base class."""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.schemas.pcr import PCRDocument


class ExtractionResult(BaseModel):
    """Result of running extraction on a transcript."""

    pcr: PCRDocument
    confidence_map: dict[str, float] = {}
    raw_output: str = ""
    latency_ms: float = 0.0
    model_name: str = ""


class ExtractionService(ABC):
    """Abstract base for transcript-to-PCR extraction services."""

    @abstractmethod
    async def extract(self, transcript: str) -> ExtractionResult:
        """Extract structured PCR fields from a transcript."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name of the extraction model."""
        ...
