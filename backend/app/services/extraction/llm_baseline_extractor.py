"""LLM baseline extractor — prompted GPT-4/Claude for transcript-to-PCR extraction.

This is the baseline against which the fine-tuned model is compared (H1, H2).
"""

import json
import time

from app.schemas.pcr import PCRDocument
from app.services.extraction.base import ExtractionResult, ExtractionService
from app.services.llm.openai_client import OpenAIClient
from app.utils.logging import logger

SYSTEM_PROMPT = """You are a medical documentation assistant that extracts structured Patient Care Report (PCR) fields from EMS paramedic speech transcripts.

Given a transcript of paramedic radio communication, extract ONLY the information explicitly stated in the transcript into the following JSON structure. Do NOT infer, assume, or fabricate any values not directly supported by the transcript text.

Output a valid JSON object with these fields (use null for fields not mentioned):

{
  "age": <int or null>,
  "sex": <"male" | "female" | "unknown" | null>,
  "chief_complaint": <string or null>,
  "primary_impression": <string or null>,
  "secondary_impression": <string or null>,
  "bp_systolic": <int or null>,
  "bp_diastolic": <int or null>,
  "heart_rate": <int or null>,
  "respiratory_rate": <int or null>,
  "spo2": <int or null>,
  "gcs_total": <int or null>,
  "avpu": <"Alert" | "Verbal" | "Pain" | "Unresponsive" | null>,
  "pain_scale": <int 0-10 or null>,
  "temperature": <float or null>,
  "blood_glucose": <int or null>,
  "etco2": <int or null>,
  "cardiac_rhythm": <string or null>,
  "allergies": [<strings>],
  "medications_current": [<strings>],
  "past_medical_history": [<strings>],
  "medications_given": [
    {"drug": <string>, "dose": <float or null>, "unit": <string or null>, "route": <string or null>}
  ],
  "procedures": [<strings>],
  "signs_symptoms": [<strings>],
  "events_leading": <string or null>,
  "narrative_text": <string or null>
}

CRITICAL RULES:
1. Extract ONLY what is explicitly stated in the transcript
2. Use null for any field not mentioned - NEVER guess or infer
3. Expand common EMS abbreviations (BP = blood pressure, HR = heart rate, RR = respiratory rate, sats = SpO2, etc.)
4. For medications_given, include dose, unit, and route when stated
5. Return ONLY the JSON object, no other text"""


class LLMBaselineExtractor(ExtractionService):
    """Prompted LLM extraction for baseline comparison."""

    def __init__(self, openai_client: OpenAIClient, model: str = "gpt-4"):
        self._openai_client = openai_client
        self._model = model

    async def extract(self, transcript: str) -> ExtractionResult:
        """Extract PCR fields from transcript using prompted LLM."""
        start = time.perf_counter()

        user_message = f"Extract PCR fields from this EMS transcript:\n\n{transcript}"

        raw_output = await self._openai_client.chat_completion(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            model=self._model,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        latency_ms = (time.perf_counter() - start) * 1000

        # Parse JSON response
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {raw_output[:200]}")
            parsed = {}

        # Build PCRDocument from parsed JSON
        pcr = self._build_pcr(parsed)

        # Build confidence map (fixed 0.7 for LLM baseline — no logprobs available)
        confidence_map = {}
        pcr_dict = pcr.model_dump()
        for field_name, value in pcr_dict.items():
            if value is not None and value != [] and value != "":
                confidence_map[field_name] = 0.7

        return ExtractionResult(
            pcr=pcr,
            confidence_map=confidence_map,
            raw_output=raw_output,
            latency_ms=latency_ms,
            model_name=self.model_name,
        )

    @property
    def model_name(self) -> str:
        return f"llm_baseline_{self._model}"

    def _build_pcr(self, data: dict) -> PCRDocument:
        """Build a PCRDocument from parsed LLM output, handling type mismatches."""
        clean = {}
        for field_name in PCRDocument.model_fields:
            value = data.get(field_name)
            if value is None:
                continue
            clean[field_name] = value

        try:
            return PCRDocument(**clean)
        except Exception as e:
            logger.error(f"Failed to build PCRDocument: {e}")
            return PCRDocument()
