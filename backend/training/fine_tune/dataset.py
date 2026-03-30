"""HuggingFace Dataset builder for training data."""

import json
from pathlib import Path

from app.utils.logging import logger


def load_jsonl(path: str) -> list[dict]:
    """Load pairs from a JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def prepare_t5_examples(pairs: list[dict]) -> list[dict]:
    """Convert (transcript, PCR JSON) pairs to T5 seq2seq format.

    Source: "Extract PCR fields from EMS transcript: <transcript>"
    Target: "<PCR JSON string>"
    """
    examples = []
    for pair in pairs:
        transcript = pair.get("transcript", "")
        pcr_json = pair.get("pcr_json", {})

        source = f"Extract PCR fields from EMS transcript: {transcript}"
        target = json.dumps(pcr_json, separators=(",", ":"))

        examples.append({"source": source, "target": target})

    return examples


def build_hf_dataset(data_dir: str) -> dict:
    """Build HuggingFace datasets from JSONL files in data_dir.

    Expects: train.jsonl, val.jsonl, test.jsonl in data_dir.
    Returns: dict with 'train', 'validation', 'test' splits.
    """
    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("Install 'datasets' package: pip install datasets")

    data_path = Path(data_dir)
    splits = {}

    for split_name, filename in [
        ("train", "train.jsonl"),
        ("validation", "val.jsonl"),
        ("test", "test.jsonl"),
    ]:
        filepath = data_path / filename
        if not filepath.exists():
            logger.warning(f"Split file not found: {filepath}")
            continue

        pairs = load_jsonl(str(filepath))
        examples = prepare_t5_examples(pairs)
        splits[split_name] = Dataset.from_list(examples)
        logger.info(f"Loaded {len(examples)} examples for {split_name}")

    return splits
