"""Evaluation metrics models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class FieldScore(BaseModel):
    """Precision, recall, F1 for a single field."""

    precision: float
    recall: float
    f1: float


class HallucinationResult(BaseModel):
    hallucination_count: int
    total_predicted_fields: int
    hallucination_rate: float
    hallucinated_fields: list[str] = Field(default_factory=list)


class CompletenessResult(BaseModel):
    mandatory_filled: int
    mandatory_total: int
    mandatory_rate: float
    required_filled: int
    required_total: int
    required_rate: float
    overall_completeness: float
    missing_mandatory: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)


class SingleEvalResult(BaseModel):
    """Evaluation result for a single (transcript, PCR) pair."""

    field_scores: dict[str, FieldScore] = Field(default_factory=dict)
    aggregate_f1: float
    hallucination: HallucinationResult
    completeness: CompletenessResult
    model_used: str
    latency_ms: float


class AggregateMetrics(BaseModel):
    """Aggregated metrics across a dataset."""

    micro_f1: float
    macro_f1: float
    mean_hallucination_rate: float
    mean_completeness: float
    per_field_f1: dict[str, float] = Field(default_factory=dict)


class BatchEvalRequest(BaseModel):
    dataset_path: str
    models: list[str] = Field(default_factory=lambda: ["finetuned", "llm_baseline"])


class BatchComparisonReport(BaseModel):
    finetuned_metrics: Optional[AggregateMetrics] = None
    baseline_metrics: Optional[AggregateMetrics] = None
    h1_result: str = ""  # "SUPPORTED" or "NOT_SUPPORTED" with F1 values
    h2_result: str = ""  # Hallucination rate comparison
    h3_result: str = ""  # Gap detection impact
    num_samples: int = 0
    timestamp: datetime = Field(default_factory=datetime.now)
