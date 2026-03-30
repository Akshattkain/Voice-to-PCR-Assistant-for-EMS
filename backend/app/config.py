"""Application configuration loaded from environment variables."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PCR_", env_file=".env")

    # Server
    app_name: str = "Voice-to-PCR Assistant"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:3000"]

    # OpenAI (Whisper ASR + GPT-4 baseline)
    openai_api_key: str = Field(default="")
    whisper_model: str = "whisper-1"
    gpt_model: str = "gpt-4"

    # Anthropic (Claude baseline)
    anthropic_api_key: Optional[str] = None
    claude_model: str = "claude-sonnet-4-20250514"

    # Fine-tuned model
    finetuned_model_path: str = "models/t5_pcr_v1"
    finetuned_model_device: str = "cpu"
    extraction_model_default: str = "llm_baseline"  # "finetuned" or "llm_baseline"

    # PCR State
    confidence_threshold: float = 0.5
    numeric_tolerance: int = 2

    # Audio
    max_audio_duration_sec: int = 300
    audio_buffer_interval_sec: float = 5.0

    # Evaluation
    eval_dataset_path: str = "training/data/processed/test.jsonl"


settings = Settings()
