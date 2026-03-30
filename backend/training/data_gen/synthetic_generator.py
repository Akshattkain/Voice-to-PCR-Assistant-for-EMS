"""Synthetic data generator — LLM-driven (transcript, PCR JSON) pair generation.

Generates realistic paramedic speech transcripts paired with structured
NEMSIS-compliant PCR JSON using LLM prompting and scenario templates.
"""

import json
import random
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from app.schemas.pcr import PCRDocument
from app.services.llm.openai_client import OpenAIClient
from app.utils.logging import logger
from training.data_gen.scenario_templates import SCENARIO_TEMPLATES, ScenarioTemplate


class DataPair(BaseModel):
    """A single (transcript, PCR JSON) training pair."""

    transcript: str
    pcr_json: dict
    scenario_type: str
    difficulty: str = "standard"


GENERATION_SYSTEM_PROMPT = """You are a medical data generation assistant. Your job is to generate realistic training data for an EMS (Emergency Medical Services) voice-to-PCR extraction system.

Given a scenario type and parameters, generate:
1. A realistic paramedic radio/dictation transcript
2. The corresponding structured PCR JSON

RULES FOR TRANSCRIPT:
- Use natural paramedic speech patterns (abbreviations, shorthand, radio protocol)
- Include realistic disfluencies: "uh", "um", pauses
- Use common EMS abbreviations: BP, HR, RR, sats, GCS, c/o, Hx, NKDA, etc.
- Include radio call structure: unit ID, patient description, vitals, interventions
- Vary the speaking style and detail level

RULES FOR PCR JSON:
- Every field value MUST be derivable from the transcript
- Use null for fields not mentioned in the transcript
- Follow the exact JSON schema provided
- Expand abbreviations in structured fields (BP -> bp_systolic/bp_diastolic)

Return a JSON object with exactly two keys: "transcript" and "pcr_json"."""


class SyntheticDataGenerator:
    """Generate (transcript, PCR JSON) pairs using LLM."""

    def __init__(self, openai_client: OpenAIClient, model: str = "gpt-4"):
        self._client = openai_client
        self._model = model

    async def generate_pair(
        self,
        scenario_type: str = "cardiac",
        difficulty: str = "standard",
    ) -> DataPair:
        """Generate a single (transcript, PCR JSON) pair."""
        template = SCENARIO_TEMPLATES.get(scenario_type)
        if not template:
            raise ValueError(f"Unknown scenario type: {scenario_type}")

        # Randomize parameters from template
        params = self._randomize_params(template, difficulty)

        user_message = self._build_generation_prompt(template, params, difficulty)

        raw = await self._client.chat_completion(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_message=user_message,
            model=self._model,
            temperature=0.8,  # Higher temp for diversity
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse generated pair: {raw[:200]}")
            raise ValueError("LLM returned invalid JSON")

        return DataPair(
            transcript=parsed.get("transcript", ""),
            pcr_json=parsed.get("pcr_json", {}),
            scenario_type=scenario_type,
            difficulty=difficulty,
        )

    async def generate_batch(
        self,
        n: int,
        scenario_distribution: Optional[dict[str, float]] = None,
        difficulty_distribution: Optional[dict[str, float]] = None,
    ) -> list[DataPair]:
        """Generate n pairs with specified scenario/difficulty distribution."""
        if scenario_distribution is None:
            scenarios = list(SCENARIO_TEMPLATES.keys())
            scenario_distribution = {s: 1.0 / len(scenarios) for s in scenarios}

        if difficulty_distribution is None:
            difficulty_distribution = {
                "easy": 0.2, "standard": 0.5, "hard": 0.2, "incomplete": 0.1
            }

        pairs = []
        for i in range(n):
            scenario = self._weighted_choice(scenario_distribution)
            difficulty = self._weighted_choice(difficulty_distribution)
            try:
                pair = await self.generate_pair(scenario, difficulty)
                pairs.append(pair)
                logger.info(f"Generated pair {i+1}/{n}: {scenario}/{difficulty}")
            except Exception as e:
                logger.error(f"Failed to generate pair {i+1}/{n}: {e}")

        return pairs

    def split_dataset(
        self, pairs: list[DataPair], train: float = 0.8, val: float = 0.1, test: float = 0.1
    ) -> tuple[list[DataPair], list[DataPair], list[DataPair]]:
        """Split dataset into train/val/test."""
        random.shuffle(pairs)
        n = len(pairs)
        train_end = int(n * train)
        val_end = train_end + int(n * val)
        return pairs[:train_end], pairs[train_end:val_end], pairs[val_end:]

    def save_dataset(self, pairs: list[DataPair], output_path: str) -> None:
        """Save pairs as JSONL file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair.model_dump()) + "\n")
        logger.info(f"Saved {len(pairs)} pairs to {output_path}")

    def _randomize_params(self, template: ScenarioTemplate, difficulty: str) -> dict:
        """Randomize scenario parameters from template distributions."""
        age = random.randint(*template.age_range)
        sex = self._weighted_choice(template.sex_distribution)
        chief_complaint = random.choice(template.common_chief_complaints)
        impression = random.choice(template.common_impressions)

        # Randomize vitals within ranges
        vitals = {}
        for field, (low, high) in template.typical_vitals.items():
            if low == high == 0:
                vitals[field] = 0
            else:
                vitals[field] = random.randint(low, high)

        # Select medications and procedures
        n_meds = random.randint(0, min(3, len(template.common_medications_given)))
        n_procs = random.randint(1, min(3, len(template.common_procedures)))
        meds = random.sample(template.common_medications_given, n_meds)
        procs = random.sample(template.common_procedures, n_procs)
        allergy = random.choice(template.common_allergies)
        pmh = random.sample(template.common_pmh, min(2, len(template.common_pmh)))
        symptoms = random.sample(
            template.common_signs_symptoms,
            min(3, len(template.common_signs_symptoms)),
        )

        return {
            "age": age, "sex": sex, "chief_complaint": chief_complaint,
            "impression": impression, "vitals": vitals, "medications": meds,
            "procedures": procs, "allergy": allergy, "pmh": pmh,
            "symptoms": symptoms,
        }

    def _build_generation_prompt(
        self, template: ScenarioTemplate, params: dict, difficulty: str
    ) -> str:
        """Build the user prompt for pair generation."""
        difficulty_instructions = {
            "easy": "Use clear, complete speech with no abbreviations. State all values explicitly.",
            "standard": "Use typical EMS radio speech with common abbreviations (BP, HR, sats, etc.). Include some natural disfluencies.",
            "hard": "Use heavy EMS jargon, many abbreviations, interrupted speech patterns, mid-sentence corrections, and background noise references.",
            "incomplete": "Deliberately omit 2-3 required fields from the transcript. The PCR JSON should have null for those omitted fields.",
        }

        return f"""Generate a realistic (transcript, PCR JSON) pair for this EMS scenario:

Scenario: {template.type} - {template.description}
Difficulty: {difficulty}
Style instructions: {difficulty_instructions[difficulty]}

Parameters:
- Age: {params['age']}, Sex: {params['sex']}
- Chief complaint: {params['chief_complaint']}
- Impression: {params['impression']}
- Vitals: {json.dumps(params['vitals'])}
- Medications given: {params['medications']}
- Procedures: {params['procedures']}
- Allergies: {params['allergy']}
- PMH: {params['pmh']}
- Signs/symptoms: {params['symptoms']}

PCR JSON schema (follow exactly):
{{
  "age": int, "sex": str, "chief_complaint": str, "primary_impression": str,
  "bp_systolic": int, "bp_diastolic": int, "heart_rate": int,
  "respiratory_rate": int, "spo2": int, "gcs_total": int, "avpu": str,
  "pain_scale": int or null, "allergies": [str], "medications_current": [str],
  "past_medical_history": [str],
  "medications_given": [{{"drug": str, "dose": float, "unit": str, "route": str}}],
  "procedures": [str], "signs_symptoms": [str], "events_leading": str or null
}}

Return {{"transcript": "...", "pcr_json": {{...}}}}"""

    @staticmethod
    def _weighted_choice(distribution: dict[str, float]) -> str:
        """Choose a key from a weighted distribution."""
        keys = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(keys, weights=weights, k=1)[0]
