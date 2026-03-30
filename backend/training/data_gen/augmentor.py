"""Transcript augmentor — apply realistic noise and variations."""

import random
import re

FILLER_WORDS = ["uh", "um", "like", "you know", "so"]

EMS_ABBREVIATIONS = {
    "blood pressure": "BP",
    "heart rate": "HR",
    "respiratory rate": "RR",
    "oxygen saturation": "sats",
    "pulse oximetry": "pulse ox",
    "chest pain": "CP",
    "shortness of breath": "SOB",
    "complaining of": "c/o",
    "history of": "Hx of",
    "no known drug allergies": "NKDA",
    "years old": "y/o",
    "milligrams": "mg",
    "intravenous": "IV",
    "sublingual": "SL",
    "by mouth": "PO",
    "intramuscular": "IM",
    "normal sinus rhythm": "NSR",
    "Glasgow Coma Scale": "GCS",
    "electrocardiogram": "EKG",
    "motor vehicle collision": "MVC",
}

RADIO_PREFIXES = [
    "Medic {unit} to County,",
    "Unit {unit}, en route with",
    "Ambulance {unit} to dispatch,",
    "Medic {unit}, we're coming in with",
    "{unit} to base,",
]


class TranscriptAugmentor:
    """Apply realistic noise and variations to transcripts."""

    def add_filler_words(self, transcript: str, probability: float = 0.1) -> str:
        """Insert filler words at random positions."""
        words = transcript.split()
        augmented = []
        for word in words:
            if random.random() < probability:
                augmented.append(random.choice(FILLER_WORDS))
            augmented.append(word)
        return " ".join(augmented)

    def abbreviation_substitution(self, transcript: str, to_abbrev: bool = True) -> str:
        """Replace full terms with EMS abbreviations or vice versa."""
        result = transcript
        for full, abbrev in EMS_ABBREVIATIONS.items():
            if to_abbrev:
                result = re.sub(re.escape(full), abbrev, result, flags=re.IGNORECASE)
            else:
                result = re.sub(re.escape(abbrev), full, result)
        return result

    def add_corrections(self, transcript: str, probability: float = 0.05) -> str:
        """Simulate mid-stream corrections on numerical values."""
        words = transcript.split()
        augmented = []
        for word in words:
            augmented.append(word)
            if word.isdigit() and random.random() < probability:
                correction = int(word) + random.choice([-2, -1, 1, 2])
                augmented.extend([", uh, sorry,", str(correction)])
        return " ".join(augmented)

    def add_radio_prefix(self, transcript: str) -> str:
        """Add a radio protocol prefix."""
        unit = random.randint(1, 20)
        prefix = random.choice(RADIO_PREFIXES).format(unit=unit)
        return f"{prefix} {transcript}"

    def augment(self, transcript: str, intensity: str = "standard") -> str:
        """Apply all augmentations at specified intensity."""
        if intensity == "easy":
            return transcript

        result = transcript

        if intensity in ("standard", "hard"):
            result = self.add_filler_words(result, probability=0.08)

        if intensity == "hard":
            result = self.add_corrections(result, probability=0.1)
            result = self.abbreviation_substitution(result, to_abbrev=True)

        return result
