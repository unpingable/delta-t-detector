#!/usr/bin/env python3
"""
TC threshold ablation: tokens-only, accel-only, both.
Patches invariant testers in-place to avoid reloading the model.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.config import DetectorConfig, RISK_PROFILES
from detector.core import DeltaTDetector
from detector.eval import load_jsonl, compute_guard_flags
from detector.run_store import PredictionRecord, store_run, _utcnow_iso
from detector.invariants import TemporalCoherenceTest

CORPUS = "data/eval_seed_v2.jsonl"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda"
PROFILE_NAME = "general"

# Load model once
config = DetectorConfig()
config.device = DEVICE
detector = DeltaTDetector(model_name=MODEL, config=config)
detector.set_risk_profile(PROFILE_NAME)

profile = RISK_PROFILES[PROFILE_NAME]

ablations = {
    "A_tokens_only": {"tokens": 4, "accel": 0.45},   # re-enable tokens, accel unchanged
    "B_accel_only":  {"tokens": 0, "accel": 0.30},    # tokens disabled, accel lowered
    "C_both":        {"tokens": 4, "accel": 0.30},     # both enabled
}

results = {}

for label, params in ablations.items():
    print(f"\n{'='*60}")
    print(f"Run {label}: tokens_threshold={params['tokens']}, accel_threshold={params['accel']}")
    print(f"{'='*60}")

    # Patch the TC tester in-place (no model reload)
    detector.temporal_coherence_test = TemporalCoherenceTest(
        threshold=profile.temporal_threshold,
        tokens_to_conf_threshold=params["tokens"],
        acceleration_threshold=params["accel"],
        zscore_threshold=profile.baseline_zscore_threshold,
    )

    started_at = _utcnow_iso()
    prediction_records = []
    items = list(load_jsonl(CORPUS))

    for item in items:
        result = detector.detect_multi_invariant(item.prompt, test_semantic=True)
        features = result.features or {}

        anomaly_score = None
        flags = compute_guard_flags(features, profile, result.temporal_debt, anomaly_score, result.prediction)

        inv_results_dict = None
        n_violated = None
        aggregate_score = None

        if result.report:
            report_inv = result.report.invariant_results
            if isinstance(report_inv, dict):
                inv_results_dict = {}
                for inv_name, inv_data in report_inv.items():
                    if hasattr(inv_data, 'to_dict'):
                        inv_results_dict[inv_name] = inv_data.to_dict()
                    elif isinstance(inv_data, dict):
                        inv_results_dict[inv_name] = inv_data
            n_violated = result.report.n_invariants_violated
            aggregate_score = getattr(result.report, 'confidence', None)

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
            notes=f"ablation={label}",
            invariant_results=inv_results_dict,
            n_violated=n_violated,
            aggregate_score=aggregate_score,
        ))

    run_dir = store_run(
        predictions=prediction_records,
        corpus_path=CORPUS,
        model=MODEL,
        profile=PROFILE_NAME,
        device=DEVICE,
        started_at=started_at,
        multi_invariant=True,
    )

    results[label] = run_dir
    print(f"Stored: {run_dir}")

# Print summary
print(f"\n\n{'='*60}")
print("ABLATION SUMMARY")
print(f"{'='*60}")

for label, rd in results.items():
    summary = json.loads((rd / "summary.json").read_text())
    manifest = json.loads((rd / "manifest.json").read_text())
    inv_v = summary.get("invariant_violation_counts", {})
    print(f"\n--- {label} (run_id={manifest['run_id']}) ---")
    print(f"  Predictions: {summary['prediction_counts']}")
    print(f"  High-risk recall: {summary['high_risk_recall']}")
    print(f"  Low-risk precision: {summary['low_risk_precision']}")
    print(f"  Invariant violations: {inv_v}")
    cm = summary['confusion_matrix']
    print(f"  Confusion matrix:")
    for exp, preds in sorted(cm.items()):
        print(f"    {exp}: {preds}")
