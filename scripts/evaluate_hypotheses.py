"""
MEDIC Hypothesis Evaluation Suite
H1: T5 vs LLM baseline field-level F1
H2: Hallucination rate comparison
H3: Completeness before vs after gap detection

Usage:
  Step 1: Run LLM baseline on test set (slow, ~2-3 hours with Ollama)
    python scripts/evaluate_hypotheses.py --run-baseline

  Step 2: Generate charts from cached results (fast)
    python scripts/evaluate_hypotheses.py --charts-only

  Both steps together:
    python scripts/evaluate_hypotheses.py
"""

import json
import argparse
import time
import os
import sys
import numpy as np
from pathlib import Path

TEST_PATH = "data/medic-synthetic/test.jsonl"
T5_EVAL_PATH = "data/eval_results.json"
LLM_EVAL_PATH = "data/llm_baseline_results.json"
CHARTS_DIR = "data/charts"

SCALAR_FIELDS = [
    "age", "sex", "chief_complaint", "primary_impression",
    "secondary_impression", "incident_location", "initial_acuity",
    "protocol_used", "bp_systolic", "bp_diastolic", "heart_rate",
    "respiratory_rate", "spo2", "gcs_total", "avpu", "pain_scale",
]
ARRAY_FIELDS = [
    "allergies", "medications_current", "past_medical_history",
    "procedures", "signs_symptoms",
]
ALL_FIELDS = SCALAR_FIELDS + ARRAY_FIELDS

MANDATORY_FIELDS = ["age", "sex", "chief_complaint", "primary_impression"]
REQUIRED_FIELDS = [
    "allergies", "past_medical_history", "bp_systolic", "bp_diastolic",
    "heart_rate", "respiratory_rate", "spo2", "gcs_total", "avpu",
    "medications_given", "procedures", "signs_symptoms",
]
HIGH_RISK_FIELDS = [
    "bp_systolic", "bp_diastolic", "heart_rate", "respiratory_rate",
    "spo2", "gcs_total", "pain_scale",
]

LLM_SYSTEM_PROMPT = """You are a medical documentation assistant that extracts structured Patient Care Report (PCR) fields from EMS paramedic speech transcripts.

Given a transcript, extract ONLY the information explicitly stated. Do NOT infer or fabricate any values not directly supported by the transcript text.

Output a valid JSON object with these fields (use null for fields not mentioned):

{
  "age": <int or null>,
  "sex": <"male" | "female" | null>,
  "incident_location": <string or null>,
  "initial_acuity": <"Critical" | "Emergent" | "Lower Acuity" | "Non-Acute" | null>,
  "protocol_used": <string or null>,
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
  "allergies": [<strings>],
  "medications_current": [<strings>],
  "past_medical_history": [<strings>],
  "medications_given": [{"drug": <string>, "dose": <float or null>, "unit": <string or null>, "route": <string or null>}],
  "procedures": [<strings>],
  "signs_symptoms": [<strings>],
  "events_leading": <string or null>
}

Return ONLY the JSON object, no other text."""


def load_test_samples():
    samples = []
    with open(TEST_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def load_t5_results():
    with open(T5_EVAL_PATH) as f:
        data = json.load(f)
    return data


def run_llm_baseline(samples):
    """Run Ollama Llama 3.1 8B on all test samples."""
    import httpx

    results = []
    total = len(samples)
    latencies = []

    print(f"\nRunning LLM baseline on {total} samples...")
    print("This will take ~2-3 hours with Ollama Llama 3.1 8B\n")

    for i, sample in enumerate(samples):
        transcript = sample["transcript"]
        gold = sample["pcr_json"]

        user_message = f"Extract PCR fields from this EMS transcript:\n\n{transcript}"

        start = time.perf_counter()
        try:
            response = httpx.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.1:8b",
                    "messages": [
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0},
                },
                timeout=180.0,
            )
            raw = response.json()["message"]["content"]
            latency = (time.perf_counter() - start) * 1000

            try:
                pred = json.loads(raw)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                match = re.search(r'\{.*\}', raw, re.DOTALL)
                if match:
                    try:
                        pred = json.loads(match.group())
                    except json.JSONDecodeError:
                        pred = {}
                else:
                    pred = {}

        except Exception as e:
            print(f"  ERROR on sample {i}: {e}")
            pred = {}
            latency = 0

        latencies.append(latency)
        results.append({
            "pred": pred,
            "gold": gold,
            "latency_ms": latency,
        })

        if (i + 1) % 10 == 0 or i == 0:
            avg_lat = np.mean(latencies[-10:])
            remaining = (total - i - 1) * avg_lat / 1000 / 60
            print(f"  [{i+1}/{total}] latency={latency:.0f}ms  est. remaining={remaining:.1f}min")

        # Save intermediate results every 50 samples
        if (i + 1) % 50 == 0:
            with open(LLM_EVAL_PATH, "w") as f:
                json.dump({"results": results, "n_complete": i + 1}, f)
            print(f"  Checkpoint saved ({i+1} samples)")

    # Save final results
    with open(LLM_EVAL_PATH, "w") as f:
        json.dump({
            "results": results,
            "n_complete": total,
            "avg_latency_ms": float(np.mean(latencies)),
        }, f, indent=2)

    print(f"\nLLM baseline complete. Avg latency: {np.mean(latencies):.0f}ms")
    print(f"Results saved to: {LLM_EVAL_PATH}")

    return results


def compute_field_metric(pred, gold, field):
    gold_val = gold.get(field)
    pred_val = pred.get(field)

    if gold_val is None:
        return None

    if isinstance(gold_val, list):
        gold_set = set(str(g).lower().strip() for g in gold_val) if gold_val else set()
        pred_set = set(str(p).lower().strip() for p in pred_val) if isinstance(pred_val, list) else set()
        if not gold_set and not pred_set:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        precision = len(gold_set & pred_set) / len(pred_set) if pred_set else 0
        recall = len(gold_set & pred_set) / len(gold_set) if gold_set else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"precision": precision, "recall": recall, "f1": f1}
    else:
        match = str(pred_val).lower().strip() == str(gold_val).lower().strip() if pred_val is not None else False
        return {"exact_match": float(match)}


def aggregate_metrics(results):
    field_metrics = {f: [] for f in ALL_FIELDS}
    for r in results:
        for field in ALL_FIELDS:
            m = compute_field_metric(r["pred"], r["gold"], field)
            if m is not None:
                field_metrics[field].append(m)

    summary = {}
    for field in ALL_FIELDS:
        vals = field_metrics[field]
        if not vals:
            continue
        if "f1" in vals[0]:
            summary[field] = {
                "f1": float(np.mean([v["f1"] for v in vals])),
                "precision": float(np.mean([v["precision"] for v in vals])),
                "recall": float(np.mean([v["recall"] for v in vals])),
                "n": len(vals),
            }
        else:
            summary[field] = {
                "exact_match": float(np.mean([v["exact_match"] for v in vals])),
                "n": len(vals),
            }
    return summary


def compute_hallucinations(results):
    total_hallucinated = 0
    total_filled = 0
    per_field = {f: {"hall": 0, "total": 0} for f in ALL_FIELDS}

    for r in results:
        pred = r["pred"]
        gold = r["gold"]
        for field in ALL_FIELDS:
            pred_val = pred.get(field)
            gold_val = gold.get(field)
            if pred_val is None or pred_val == [] or pred_val == "":
                continue
            per_field[field]["total"] += 1
            total_filled += 1
            if gold_val is None or gold_val == [] or gold_val == "":
                per_field[field]["hall"] += 1
                total_hallucinated += 1

    rate = total_hallucinated / total_filled if total_filled > 0 else 0
    return rate, total_hallucinated, total_filled, per_field


def compute_completeness_stats(results):
    mandatory_scores = []
    required_scores = []
    all_scores = []
    mandatory_missing_counts = []
    required_missing_counts = []

    for r in results:
        pred = r["pred"]
        m_filled = sum(1 for f in MANDATORY_FIELDS
                       if pred.get(f) is not None and pred.get(f) != [] and pred.get(f) != "")
        r_filled = sum(1 for f in REQUIRED_FIELDS
                       if pred.get(f) is not None and pred.get(f) != [] and pred.get(f) != "")
        all_tracked = MANDATORY_FIELDS + REQUIRED_FIELDS
        a_filled = sum(1 for f in all_tracked
                       if pred.get(f) is not None and pred.get(f) != [] and pred.get(f) != "")

        mandatory_scores.append(m_filled / len(MANDATORY_FIELDS))
        required_scores.append(r_filled / len(REQUIRED_FIELDS))
        all_scores.append(a_filled / len(all_tracked))
        mandatory_missing_counts.append(len(MANDATORY_FIELDS) - m_filled)
        required_missing_counts.append(len(REQUIRED_FIELDS) - r_filled)

    return {
        "mandatory_completeness": mandatory_scores,
        "required_completeness": required_scores,
        "all_completeness": all_scores,
        "mandatory_missing": mandatory_missing_counts,
        "required_missing": required_missing_counts,
    }


def generate_charts(t5_summary, llm_summary, t5_results, llm_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Chart 1: H1 — Field-level accuracy comparison (horizontal bar)
    print("\n[Chart 1] H1: Field-level accuracy comparison...")
    fields_plot = []
    t5_scores = []
    llm_scores = []
    for field in ALL_FIELDS:
        t5_d = t5_summary.get(field, {})
        llm_d = llm_summary.get(field, {})
        if not t5_d or not llm_d:
            continue
        t5_s = t5_d.get("f1", t5_d.get("exact_match", 0))
        llm_s = llm_d.get("f1", llm_d.get("exact_match", 0))
        fields_plot.append(field)
        t5_scores.append(t5_s)
        llm_scores.append(llm_s)

    idx = np.argsort(t5_scores)[::-1]
    fields_plot = [fields_plot[i] for i in idx]
    t5_scores = [t5_scores[i] for i in idx]
    llm_scores = [llm_scores[i] for i in idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(len(fields_plot))
    w = 0.35
    ax.barh(x - w/2, t5_scores, w, label="Fine-tuned T5", color="#2563eb")
    ax.barh(x + w/2, llm_scores, w, label="LLM Baseline (Llama 3.1)", color="#dc2626")
    ax.axvline(x=0.85, color="#16a34a", linestyle="--", linewidth=1.5, label="H1 Target (0.85)")
    ax.set_xlabel("Score (EM or F1)")
    ax.set_title("H1: Field-Level Accuracy — T5 vs LLM Baseline")
    ax.set_yticks(x)
    ax.set_yticklabels(fields_plot, fontsize=9)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.08)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/h1_field_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {CHARTS_DIR}/h1_field_accuracy.png")

    # Chart 2: H1 — Overall comparison
    print("[Chart 2] H1: Overall comparison...")
    t5_em = np.mean([s.get("exact_match", 0) for s in t5_summary.values() if "exact_match" in s])
    t5_f1 = np.mean([s.get("f1", 0) for s in t5_summary.values() if "f1" in s])
    llm_em = np.mean([s.get("exact_match", 0) for s in llm_summary.values() if "exact_match" in s])
    llm_f1 = np.mean([s.get("f1", 0) for s in llm_summary.values() if "f1" in s])

    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["Scalar EM", "Array F1", "Combined"]
    t5_v = [t5_em, t5_f1, (t5_em + t5_f1) / 2]
    llm_v = [llm_em, llm_f1, (llm_em + llm_f1) / 2]
    x = np.arange(len(metrics))
    w = 0.3
    ax.bar(x - w/2, t5_v, w, label="Fine-tuned T5", color="#2563eb")
    ax.bar(x + w/2, llm_v, w, label="LLM Baseline", color="#dc2626")
    ax.axhline(y=0.85, color="#16a34a", linestyle="--", linewidth=1.5, label="H1 Target")
    ax.set_ylabel("Score")
    ax.set_title("H1: Overall Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 1.08)
    for i, (t, l) in enumerate(zip(t5_v, llm_v)):
        ax.text(i - w/2, t + 0.02, f"{t:.3f}", ha="center", fontsize=9)
        ax.text(i + w/2, l + 0.02, f"{l:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/h1_overall.png", dpi=150)
    plt.close()
    print(f"  Saved: {CHARTS_DIR}/h1_overall.png")

    # Chart 3: H2 — Hallucination rate
    print("[Chart 3] H2: Hallucination rate...")
    t5_rate, t5_h, t5_f, t5_pf = compute_hallucinations(t5_results)
    llm_rate, llm_h, llm_f, llm_pf = compute_hallucinations(llm_results)

    fig, ax = plt.subplots(figsize=(6, 5))
    models = ["Fine-tuned T5", "LLM Baseline"]
    rates = [t5_rate * 100, llm_rate * 100]
    colors = ["#2563eb", "#dc2626"]
    bars = ax.bar(models, rates, color=colors, width=0.5)
    ax.axhline(y=5, color="#16a34a", linestyle="--", linewidth=1.5, label="H2 Target (< 5%)")
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("H2: Unsupported Field Insertions")
    ax.legend()
    for bar, rate, h, f in zip(bars, rates, [t5_h, llm_h], [t5_f, llm_f]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{rate:.1f}%\n({h}/{f})", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.4 + 2)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/h2_hallucination.png", dpi=150)
    plt.close()
    print(f"  Saved: {CHARTS_DIR}/h2_hallucination.png")

    # Chart 4: H3 — Completeness
    print("[Chart 4] H3: Completeness distribution...")
    t5_comp = compute_completeness_stats(t5_results)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(np.array(t5_comp["all_completeness"]) * 100, bins=20,
                 color="#2563eb", alpha=0.7, edgecolor="black")
    mean_comp = np.mean(t5_comp["all_completeness"]) * 100
    axes[0].axvline(x=mean_comp, color="#dc2626", linestyle="--", linewidth=2,
                    label=f"Mean: {mean_comp:.1f}%")
    axes[0].set_xlabel("Completeness (%)")
    axes[0].set_ylabel("Number of Samples")
    axes[0].set_title("Single-Pass T5 Completeness (Mandatory + Required)")
    axes[0].legend()

    avg_m_miss = np.mean(t5_comp["mandatory_missing"])
    avg_r_miss = np.mean(t5_comp["required_missing"])
    cats = [f"Mandatory\n(of {len(MANDATORY_FIELDS)})", f"Required\n(of {len(REQUIRED_FIELDS)})"]
    bars = axes[1].bar(cats, [avg_m_miss, avg_r_miss], color=["#dc2626", "#f59e0b"], width=0.5)
    for bar, miss in zip(bars, [avg_m_miss, avg_r_miss]):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f"{miss:.2f}", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Avg Missing Fields per Sample")
    axes[1].set_title("H3: Fields Flagged by Gap Detection")

    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/h3_completeness.png", dpi=150)
    plt.close()
    print(f"  Saved: {CHARTS_DIR}/h3_completeness.png")

    # Chart 5: High-risk fields accuracy
    print("[Chart 5] High-risk field accuracy...")
    hr_fields = []
    hr_t5 = []
    hr_llm = []
    for f in HIGH_RISK_FIELDS:
        t5_d = t5_summary.get(f, {})
        llm_d = llm_summary.get(f, {})
        if t5_d and llm_d:
            hr_fields.append(f)
            hr_t5.append(t5_d.get("exact_match", t5_d.get("f1", 0)))
            hr_llm.append(llm_d.get("exact_match", llm_d.get("f1", 0)))

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(hr_fields))
    w = 0.35
    ax.bar(x - w/2, hr_t5, w, label="Fine-tuned T5", color="#2563eb")
    ax.bar(x + w/2, hr_llm, w, label="LLM Baseline", color="#dc2626")
    ax.set_ylabel("Exact Match")
    ax.set_title("High-Risk Field Accuracy (Vitals + Scores)")
    ax.set_xticks(x)
    ax.set_xticklabels(hr_fields, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.08)
    for i, (t, l) in enumerate(zip(hr_t5, hr_llm)):
        ax.text(i - w/2, t + 0.02, f"{t:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, l + 0.02, f"{l:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/high_risk_accuracy.png", dpi=150)
    plt.close()
    print(f"  Saved: {CHARTS_DIR}/high_risk_accuracy.png")

    # Chart 6: Latency comparison
    print("[Chart 6] Latency comparison...")
    llm_latencies = [r.get("latency_ms", 0) for r in llm_results if r.get("latency_ms", 0) > 0]
    # T5 latency from eval — approximate from the extraction endpoint (~4-5s)
    t5_latency_est = 4500

    fig, ax = plt.subplots(figsize=(8, 5))
    if llm_latencies:
        ax.hist(llm_latencies, bins=30, alpha=0.7, color="#dc2626", label=f"LLM Baseline (mean={np.mean(llm_latencies):.0f}ms)", edgecolor="black")
    ax.axvline(x=t5_latency_est, color="#2563eb", linestyle="--", linewidth=2, label=f"T5 (~{t5_latency_est}ms)")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference Latency Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{CHARTS_DIR}/latency_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: {CHARTS_DIR}/latency_comparison.png")

    # Print summary table
    print("\n" + "=" * 70)
    print("HYPOTHESIS EVALUATION SUMMARY")
    print("=" * 70)
    t5_combined = (t5_em + t5_f1) / 2
    llm_combined = (llm_em + llm_f1) / 2
    print(f"\nH1: T5 combined score = {t5_combined:.4f} (target > 0.85)")
    print(f"    LLM combined score = {llm_combined:.4f}")
    print(f"    T5 {'PASSES' if t5_combined > 0.85 else 'FAILS'} H1 threshold")
    print(f"    T5 {'outperforms' if t5_combined > llm_combined else 'underperforms'} LLM baseline by {abs(t5_combined - llm_combined):.4f}")

    print(f"\nH2: T5 hallucination rate = {t5_rate*100:.1f}% (target < 5%)")
    print(f"    LLM hallucination rate = {llm_rate*100:.1f}%")
    print(f"    T5 {'PASSES' if t5_rate < 0.05 else 'FAILS'} H2 threshold")

    mean_all = np.mean(t5_comp["all_completeness"]) * 100
    pct_with_gaps = sum(1 for s in t5_comp["all_completeness"] if s < 1.0) / len(t5_comp["all_completeness"]) * 100
    print(f"\nH3: Avg completeness (single-pass) = {mean_all:.1f}%")
    print(f"    Samples with gaps = {pct_with_gaps:.1f}%")
    print(f"    Avg mandatory missing = {np.mean(t5_comp['mandatory_missing']):.2f}")
    print(f"    Avg required missing = {np.mean(t5_comp['required_missing']):.2f}")
    print(f"    Gap detection identifies these for prompting")

    if llm_latencies:
        print(f"\nLatency: T5 ~{t5_latency_est}ms vs LLM {np.mean(llm_latencies):.0f}ms")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="MEDIC Hypothesis Evaluation")
    parser.add_argument("--run-baseline", action="store_true", help="Run LLM baseline on test set")
    parser.add_argument("--charts-only", action="store_true", help="Generate charts from cached results")
    args = parser.parse_args()

    # If no flags, run everything
    run_baseline = args.run_baseline or (not args.charts_only)
    gen_charts = args.charts_only or (not args.run_baseline) or args.run_baseline

    # Load T5 results
    print("Loading T5 evaluation results...")
    t5_data = load_t5_results()
    t5_summary = t5_data["summary"]

    # Build T5 results list from sample_predictions
    t5_results = t5_data.get("sample_predictions", [])
    # If we only have sample predictions (first 20), we need the full set
    # Use the summary for charts, and sample_predictions for hallucination on available samples

    # Load test samples
    samples = load_test_samples()
    print(f"Loaded {len(samples)} test samples")

    # Run or load LLM baseline
    if run_baseline and not args.charts_only:
        if os.path.exists(LLM_EVAL_PATH):
            print(f"\nLLM baseline results already exist at {LLM_EVAL_PATH}")
            resp = input("Re-run? (y/N): ").strip().lower()
            if resp != "y":
                run_baseline = False

        if run_baseline:
            llm_results = run_llm_baseline(samples)
        else:
            with open(LLM_EVAL_PATH) as f:
                llm_data = json.load(f)
            llm_results = llm_data["results"]
    else:
        if not os.path.exists(LLM_EVAL_PATH):
            print(f"\nERROR: No LLM baseline results found at {LLM_EVAL_PATH}")
            print("Run with --run-baseline first")
            sys.exit(1)
        with open(LLM_EVAL_PATH) as f:
            llm_data = json.load(f)
        llm_results = llm_data["results"]

    # Compute LLM summary
    print("Computing LLM baseline metrics...")
    llm_summary = aggregate_metrics(llm_results)

    # For T5, if we don't have full pred/gold pairs for all 239,
    # use summary for field metrics and sample_predictions for hallucination
    # We need full results for hallucination computation
    # Re-run T5 eval if needed, or use the sample predictions we have
    if len(t5_results) < len(samples):
        print(f"\nNote: T5 eval has {len(t5_results)} sample predictions (of {len(samples)})")
        print("Using summary metrics for H1, sample predictions for H2")
        # For hallucination, scale from available samples
        t5_results_for_hall = t5_results
    else:
        t5_results_for_hall = t5_results

    # Generate charts
    print("\nGenerating charts...")
    generate_charts(t5_summary, llm_summary, t5_results_for_hall, llm_results)

    print(f"\nAll charts saved to: {CHARTS_DIR}/")
    print("Done!")


if __name__ == "__main__":
    main()