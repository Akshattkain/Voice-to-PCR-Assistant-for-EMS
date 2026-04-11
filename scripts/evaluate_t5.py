"""
MEDIC T5 Test Set Evaluation
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer

MODEL_PATH   = "/Users/thesilentreaper/Documents/Projects/EMS/models"
TEST_PATH    = "/Users/thesilentreaper/Documents/Projects/EMS/data/medic-synthetic/test.jsonl"
OUTPUT_PATH  = "/Users/thesilentreaper/Documents/Projects/EMS/data/eval_results.json"
MAX_INPUT_LEN  = 512
MAX_TARGET_LEN = 768
BATCH_SIZE     = 4
DEVICE         = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Device: {DEVICE}")

SCALAR_FIELDS = [
    "age", "sex", "chief_complaint", "primary_impression",
    "secondary_impression", "incident_location", "initial_acuity",
    "protocol_used", "bp_systolic", "bp_diastolic", "heart_rate",
    "respiratory_rate", "spo2", "gcs_total", "avpu", "pain_scale",
    "events_leading",
]
ARRAY_FIELDS = [
    "allergies", "medications_current", "past_medical_history",
    "procedures", "signs_symptoms",
]
NUMERIC_FIELDS = {
    "age", "bp_systolic", "bp_diastolic", "heart_rate",
    "respiratory_rate", "spo2", "gcs_total", "pain_scale"
}

def pcr_to_target_string(pcr):
    """Convert PCR dict to flat target string for T5."""
    parts = []
    for field in SCALAR_FIELDS:
        val = pcr.get(field)
        parts.append(f"{field}: {val}" if val is not None else f"{field}: null")
    for field in ARRAY_FIELDS:
        val = pcr.get(field)
        if val and isinstance(val, list):
            parts.append(f"{field}: {' | '.join(str(v) for v in val)}")
        else:
            parts.append(f"{field}: null")
    meds = pcr.get("medications_given")
    if meds and isinstance(meds, list):
        med_strs = []
        for m in meds:
            if isinstance(m, dict):
                med_strs.append(
                    f"{m.get('drug','?')} {m.get('dose','?')}{m.get('unit','?')} {m.get('route','?')}"
                )
        parts.append(f"medications_given: {' | '.join(med_strs)}")
    else:
        parts.append("medications_given: null")
    return " ; ".join(parts)

def target_string_to_pcr(text):
    """Parse T5 output string back to PCR dict."""
    pcr = {}
    for part in text.split(" ; "):
        part = part.strip()
        if ": " not in part:
            continue
        key, val = part.split(": ", 1)
        key = key.strip()
        val = val.strip()
        if val == "null":
            pcr[key] = None
        elif key in ARRAY_FIELDS:
            pcr[key] = [v.strip() for v in val.split(" | ")]
        elif key in NUMERIC_FIELDS:
            try:
                pcr[key] = float(val) if "." in val else int(val)
            except ValueError:
                pcr[key] = val
        else:
            pcr[key] = val
    return pcr

def compute_field_metrics(pred_pcr, gold_pcr):
    """Compute per-field exact match and F1."""
    results = {}
    for field in SCALAR_FIELDS + ARRAY_FIELDS:
        gold = gold_pcr.get(field)
        pred = pred_pcr.get(field)
        if gold is None:
            results[field] = None
            continue
        if isinstance(gold, list):
            gold_set = set(str(g).lower().strip() for g in gold)
            pred_set = set(str(p).lower().strip() for p in pred) if isinstance(pred, list) else set()
            if gold_set or pred_set:
                precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0
                recall    = len(gold_set & pred_set) / len(gold_set) if gold_set else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 1.0
            results[field] = {"precision": precision, "recall": recall, "f1": f1}
        else:
            match = str(pred).lower().strip() == str(gold).lower().strip()
            results[field] = {"exact_match": float(match)}
    return results

# load model and tokenizer
print(f"Loading model from {MODEL_PATH}...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model     = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

# load test samples
print(f"Loading test set...")
samples = []
with open(TEST_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))
print(f"  {len(samples)} test samples")

# run inference batch by batch
all_field_metrics = {f: [] for f in SCALAR_FIELDS + ARRAY_FIELDS}
all_pred_pcrs     = []
all_gold_pcrs     = []

print("\nRunning inference...")
for i in range(0, len(samples), BATCH_SIZE):
    batch     = samples[i:i + BATCH_SIZE]
    inputs    = [f"extract pcr: {s['transcript']}" for s in batch]
    gold_pcrs = [s["pcr_json"] for s in batch]

    enc = tokenizer(
        inputs,
        max_length=MAX_INPUT_LEN,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        preds = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=MAX_TARGET_LEN,
            num_beams=4,
        )

    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)

    for pred_text, gold_pcr in zip(pred_texts, gold_pcrs):
        pred_pcr = target_string_to_pcr(pred_text)
        all_pred_pcrs.append(pred_pcr)
        all_gold_pcrs.append(gold_pcr)
        metrics = compute_field_metrics(pred_pcr, gold_pcr)
        for field, result in metrics.items():
            if result is not None:
                all_field_metrics[field].append(result)

    if (i // BATCH_SIZE + 1) % 10 == 0:
        print(f"  [{min(i + BATCH_SIZE, len(samples))}/{len(samples)}]")

# aggregate and print results
print("\nTEST SET RESULTS\n")
summary     = {}
overall_f1s = []
overall_ems = []

for field in SCALAR_FIELDS + ARRAY_FIELDS:
    results = all_field_metrics[field]
    if not results:
        continue
    if "f1" in results[0]:
        avg_f1 = np.mean([r["f1"] for r in results])
        avg_p  = np.mean([r["precision"] for r in results])
        avg_r  = np.mean([r["recall"] for r in results])
        summary[field] = {"precision": avg_p, "recall": avg_r, "f1": avg_f1, "n": len(results)}
        overall_f1s.append(avg_f1)
        print(f"  {field:<30} F1={avg_f1:.3f}  P={avg_p:.3f}  R={avg_r:.3f}  (n={len(results)})")
    elif "exact_match" in results[0]:
        avg_em = np.mean([r["exact_match"] for r in results])
        summary[field] = {"exact_match": avg_em, "n": len(results)}
        overall_ems.append(avg_em)
        print(f"  {field:<30} EM={avg_em:.3f}  (n={len(results)})")

print(f"\nOverall Scalar Exact Match : {np.mean(overall_ems):.4f}")
print(f"Overall Array F1           : {np.mean(overall_f1s):.4f}")
print(f"Combined                   : {np.mean(overall_ems + overall_f1s):.4f}")

# save full results including sample predictions
output = {
    "summary": summary,
    "overall": {
        "scalar_exact_match": float(np.mean(overall_ems)),
        "array_f1":           float(np.mean(overall_f1s)),
        "combined":           float(np.mean(overall_ems + overall_f1s)),
    },
    "n_samples": len(samples),
    "sample_predictions": [
        {"pred": p, "gold": g}
        for p, g in zip(all_pred_pcrs, all_gold_pcrs)
    ]
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to: {OUTPUT_PATH}")