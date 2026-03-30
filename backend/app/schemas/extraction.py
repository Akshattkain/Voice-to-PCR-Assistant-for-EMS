"""Extraction request/response models."""

from pydantic import BaseModel, Field

from app.schemas.pcr import PCRDocument, PCRStateEnvelope


class ExtractionRequest(BaseModel):
    transcript: str
    model: str = "llm_baseline"  # "finetuned" or "llm_baseline"


class ExtractionResponse(BaseModel):
    """Result of a single extraction run."""

    extracted_pcr: PCRDocument
    confidence_map: dict[str, float] = Field(default_factory=dict)
    model_used: str
    latency_ms: float
    pcr_state: PCRStateEnvelope


class ComparisonResponse(BaseModel):
    """Side-by-side comparison of two extraction models."""

    finetuned_result: ExtractionResponse
    llm_baseline_result: ExtractionResponse
    field_diffs: dict[str, dict] = Field(default_factory=dict)
