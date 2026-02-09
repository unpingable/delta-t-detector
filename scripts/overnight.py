#!/usr/bin/env python3
"""
Overnight harness: load model once, run canary + stability sweep.

Usage:
    python3 scripts/overnight.py                       # full suite
    python3 scripts/overnight.py --canary-only         # just the nightly canary
    python3 scripts/overnight.py --sweep-only          # just the stability sweep
    python3 scripts/overnight.py --corpus path.jsonl   # custom corpus
"""

import argparse
import json
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.config import DetectorConfig, RISK_PROFILES
from detector.core import DeltaTDetector
from detector.eval import load_jsonl, compute_guard_flags
from detector.run_store import PredictionRecord, store_run, _utcnow_iso

# Defaults
CORPUS = "data/eval_seed_v2.jsonl"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda"
PROFILE_NAME = "general"

# Stability sweep grid (small and targeted)
TEMP_GRID = [0.5, 0.7, 0.9]
NPERTURB_GRID = [3, 5]


def run_one_eval(detector, corpus_path, profile_name, label=""):
    """Run a single eval pass, return (prediction_records, started_at)."""
    profile = RISK_PROFILES[profile_name]
    started_at = _utcnow_iso()
    prediction_records = []

    for item in load_jsonl(corpus_path):
        result = detector.detect_multi_invariant(item.prompt, test_semantic=True)
        features = result.features or {}
        anomaly_score = None
        flags = compute_guard_flags(
            features, profile, result.temporal_debt, anomaly_score, result.prediction
        )

        inv_results_dict = None
        n_violated = None
        aggregate_score = None

        if result.report:
            report_inv = result.report.invariant_results
            if isinstance(report_inv, dict):
                inv_results_dict = {}
                for inv_name, inv_data in report_inv.items():
                    if hasattr(inv_data, "to_dict"):
                        inv_results_dict[inv_name] = inv_data.to_dict()
                    elif isinstance(inv_data, dict):
                        inv_results_dict[inv_name] = inv_data
            n_violated = result.report.n_invariants_violated
            aggregate_score = getattr(result.report, "confidence", None)

        prediction_records.append(PredictionRecord(
            id=item.id,
            prompt=item.prompt,
            expected_risk=item.expected_risk,
            prediction=result.prediction,
            confidence=result.confidence,
            temporal_debt=result.temporal_debt,
            anomaly_score=anomaly_score,
            features=features,
            guard_flags=flags,
            notes=label,
            invariant_results=inv_results_dict,
            n_violated=n_violated,
            aggregate_score=aggregate_score,
        ))

    return prediction_records, started_at


def main():
    parser = argparse.ArgumentParser(description="Overnight eval harness")
    parser.add_argument("--corpus", default=CORPUS, help="JSONL corpus path")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--profile", default=PROFILE_NAME)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--sweep-only", action="store_true")
    args = parser.parse_args()

    run_canary = not args.sweep_only
    run_sweep = not args.canary_only

    # Load model once
    config = DetectorConfig()
    config.device = args.device
    detector = DeltaTDetector(model_name=args.model, config=config)
    detector.set_risk_profile(args.profile)

    results = []
    wall_start = time.time()

    # --- Canary run (default config) ---
    if run_canary:
        print(f"\n{'='*60}")
        print("CANARY RUN (default config)")
        print(f"{'='*60}")
        t0 = time.time()
        preds, started = run_one_eval(detector, args.corpus, args.profile, label="canary")
        rd = store_run(
            preds, args.corpus, args.model, args.profile,
            device=args.device, started_at=started, multi_invariant=True,
        )
        elapsed = time.time() - t0
        results.append(("canary", rd, elapsed))
        print(f"  Stored: {rd}  ({elapsed:.0f}s)")

    # --- Stability sweep ---
    if run_sweep:
        original_temp_base = config.temperature_base
        original_temp_step = config.temperature_step
        profile = RISK_PROFILES[args.profile]
        original_n_perturb = profile.n_perturbations

        for temp_base in TEMP_GRID:
            for n_perturb in NPERTURB_GRID:
                label = f"sweep_t{temp_base}_n{n_perturb}"
                print(f"\n{'='*60}")
                print(f"SWEEP: temp_base={temp_base}, n_perturbations={n_perturb}")
                print(f"{'='*60}")

                # Patch config in-place (no model reload)
                config.temperature_base = temp_base
                profile.n_perturbations = n_perturb

                t0 = time.time()
                preds, started = run_one_eval(
                    detector, args.corpus, args.profile, label=label
                )
                rd = store_run(
                    preds, args.corpus, args.model, args.profile,
                    device=args.device, started_at=started, multi_invariant=True,
                )
                elapsed = time.time() - t0
                results.append((label, rd, elapsed))
                print(f"  Stored: {rd}  ({elapsed:.0f}s)")

        # Restore originals
        config.temperature_base = original_temp_base
        config.temperature_step = original_temp_step
        profile.n_perturbations = original_n_perturb

    # --- Summary ---
    wall_total = time.time() - wall_start
    print(f"\n\n{'='*60}")
    print(f"OVERNIGHT SUMMARY  ({wall_total:.0f}s total)")
    print(f"{'='*60}")

    for label, rd, elapsed in results:
        summary = json.loads((rd / "summary.json").read_text())
        pc = summary["prediction_counts"]
        print(f"\n  {label} ({elapsed:.0f}s)")
        print(f"    Predictions: {pc}")
        print(f"    HR recall: {summary.get('high_risk_recall', 'N/A')}")
        print(f"    LP precision: {summary.get('low_risk_precision', 'N/A')}")
        inv_v = summary.get("invariant_violation_counts", {})
        if inv_v:
            print(f"    Invariant violations: {inv_v}")


if __name__ == "__main__":
    main()
