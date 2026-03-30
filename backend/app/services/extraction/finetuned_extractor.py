"""Fine-tuned T5/Llama+LoRA extractor for transcript-to-PCR extraction.

This module loads a fine-tuned model and runs inference to extract
structured PCR fields from transcripts. Requires a trained model checkpoint.
"""

import json
import time
from typing import Optional

from app.schemas.pcr import PCRDocument
from app.services.extraction.base import ExtractionResult, ExtractionService
from app.utils.logging import logger


class FineTunedExtractor(ExtractionService):
    """Fine-tuned T5-base or Llama+LoRA extraction model."""

    def __init__(self, model_path: str, device: str = "cpu"):
        self._model_path = model_path
        self._device = device
        self._model = None
        self._tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        """Load the fine-tuned model from disk. Deferred to avoid import cost at startup."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading fine-tuned model from {self._model_path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info("Fine-tuned model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            raise

    async def extract(self, transcript: str) -> ExtractionResult:
        """Extract PCR fields using the fine-tuned model."""
        if not self._loaded:
            self.load_model()

        start = time.perf_counter()

        # Format input for T5
        input_text = f"Extract PCR fields from EMS transcript: {transcript}"
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self._device)

        # Generate with output scores for confidence
        import torch

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_length=1024,
                num_beams=4,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Decode output
        raw_output = self._tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        latency_ms = (time.perf_counter() - start) * 1000

        # Parse JSON
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.error(f"Fine-tuned model output invalid JSON: {raw_output[:200]}")
            parsed = {}

        pcr = self._build_pcr(parsed)

        # Compute per-field confidence from generation scores
        confidence_map = self._compute_confidence(pcr, outputs)

        return ExtractionResult(
            pcr=pcr,
            confidence_map=confidence_map,
            raw_output=raw_output,
            latency_ms=latency_ms,
            model_name=self.model_name,
        )

    @property
    def model_name(self) -> str:
        return "finetuned_t5"

    def _build_pcr(self, data: dict) -> PCRDocument:
        """Build PCRDocument from model output."""
        clean = {}
        for field_name in PCRDocument.model_fields:
            value = data.get(field_name)
            if value is not None:
                clean[field_name] = value
        try:
            return PCRDocument(**clean)
        except Exception as e:
            logger.error(f"Failed to build PCRDocument from fine-tuned output: {e}")
            return PCRDocument()

    def _compute_confidence(self, pcr: PCRDocument, outputs) -> dict[str, float]:
        """Compute per-field confidence from token-level log probabilities.

        Uses the average log-probability of generated tokens as a proxy for
        confidence. Falls back to 0.8 if scores are unavailable.
        """
        import math

        confidence_map = {}
        pcr_dict = pcr.model_dump()

        # Try to compute average log probability from generation scores
        avg_logprob = None
        if hasattr(outputs, "scores") and outputs.scores:
            import torch

            logprobs = []
            for i, score in enumerate(outputs.scores):
                token_id = outputs.sequences[0][i + 1]  # +1 for decoder start token
                log_prob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                logprobs.append(log_prob)
            if logprobs:
                avg_logprob = sum(logprobs) / len(logprobs)

        # Convert to 0-1 confidence
        default_conf = 0.8
        if avg_logprob is not None:
            # Map log probability to confidence (exp of avg logprob)
            default_conf = min(1.0, max(0.0, math.exp(avg_logprob)))

        for field_name, value in pcr_dict.items():
            if value is not None and value != [] and value != "":
                confidence_map[field_name] = default_conf

        return confidence_map
