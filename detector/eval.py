"""
Δt Hallucination Detector - Evaluation Harness
Minimal JSONL corpus runner with CSV output.
"""

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional

from .config import DetectorConfig, RISK_PROFILES, RiskProfile


@dataclass
class EvalItem:
    id: str
    prompt: str
    expected_risk: str
    notes: str = ""


def load_jsonl(path: str) -> Iterable[EvalItem]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield EvalItem(
                id=str(data.get("id", "")),
                prompt=str(data.get("prompt", "")),
                expected_risk=str(data.get("expected_risk", "")),
                notes=str(data.get("notes", ""))
            )


def compute_guard_flags(
    features: Dict[str, Any],
    profile: RiskProfile,
    temporal_debt: float,
    anomaly_score: Optional[float],
    prediction: str
) -> Dict[str, bool]:
    low_evidence = (
        features.get("entropy_variance", 0.0) < profile.low_evidence_entropy_threshold and
        features.get("tokens_to_high_conf", 0.0) < profile.low_evidence_tokens_threshold
    )
    anomaly = anomaly_score is not None and anomaly_score > profile.anomaly_score_uncertain_threshold
    debt_cap = prediction in ("truthful", "temporal_stable", "CLEAN", "WARN") and temporal_debt > profile.temporal_debt_confidence_cap_threshold
    return {
        "low_evidence": low_evidence,
        "anomaly": anomaly,
        "debt_cap": debt_cap
    }


def run_eval(
    jsonl_path: str,
    model: str,
    profile_name: str,
    device: str = "auto",
    baseline: bool = False,
    output_csv: Optional[str] = None,
    store: bool = True,
    multi_invariant: bool = False,
) -> Optional[Path]:
    from .core import DeltaTDetector
    config = DetectorConfig()
    config.device = device

    detector = DeltaTDetector(model_name=model, config=config)
    detector.set_risk_profile(profile_name)

    if baseline:
        detector.load_baseline(model)

    profile = RISK_PROFILES.get(profile_name, RISK_PROFILES["general"])

    fieldnames = [
        "id",
        "expected_risk",
        "prediction",
        "confidence",
        "temporal_debt",
        "anomaly_score",
        "entropy_variance",
        "tokens_to_high_conf",
        "final_confidence",
        "max_confidence_slope",
        "confidence_acceleration",
        "perturbation_sensitivity",
        "low_evidence_triggered",
        "anomaly_triggered",
        "debt_cap_triggered",
        "notes"
    ]
    if multi_invariant:
        fieldnames.extend(["n_violated", "aggregate_score"])

    started_at: Optional[str] = None
    if store:
        from .run_store import PredictionRecord, store_run, _utcnow_iso
        started_at = _utcnow_iso()

    prediction_records: List["PredictionRecord"] = [] if store else []  # type: ignore[name-defined]

    out_f = open(output_csv, "w", newline="") if output_csv else None
    writer = csv.DictWriter(out_f if out_f is not None else sys.stdout, fieldnames=fieldnames)

    writer.writeheader()

    for item in load_jsonl(jsonl_path):
        if multi_invariant:
            result = detector.detect_multi_invariant(item.prompt, test_semantic=True)
        else:
            result = detector.detect(item.prompt, generate_report=False)
        features = result.features or {}

        anomaly_score = None
        if detector._current_baseline is not None and features:
            try:
                from .features import TemporalFeatures
                tf = TemporalFeatures(**features)
                from .baseline import compute_anomaly_score
                anomaly_score = compute_anomaly_score(tf, detector._current_baseline)
            except Exception:
                anomaly_score = None

        flags = compute_guard_flags(features, profile, result.temporal_debt, anomaly_score, result.prediction)

        # Extract multi-invariant data from report
        inv_results_dict: Optional[Dict[str, Any]] = None
        n_violated: Optional[int] = None
        aggregate_score: Optional[float] = None

        if multi_invariant and result.report:
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

        row = {
            "id": item.id,
            "expected_risk": item.expected_risk,
            "prediction": result.prediction,
            "confidence": result.confidence,
            "temporal_debt": result.temporal_debt,
            "anomaly_score": anomaly_score,
            "entropy_variance": features.get("entropy_variance"),
            "tokens_to_high_conf": features.get("tokens_to_high_conf"),
            "final_confidence": features.get("final_confidence"),
            "max_confidence_slope": features.get("max_confidence_slope"),
            "confidence_acceleration": features.get("confidence_acceleration"),
            "perturbation_sensitivity": features.get("perturbation_sensitivity"),
            "low_evidence_triggered": flags["low_evidence"],
            "anomaly_triggered": flags["anomaly"],
            "debt_cap_triggered": flags["debt_cap"],
            "notes": item.notes
        }
        if multi_invariant:
            row["n_violated"] = n_violated
            row["aggregate_score"] = aggregate_score
        writer.writerow(row)

        if store:
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
                notes=item.notes,
                invariant_results=inv_results_dict,
                n_violated=n_violated,
                aggregate_score=aggregate_score,
                fail_type=getattr(result, 'fail_type', None),
                fail_subjects=getattr(result, 'fail_subjects', None),
            ))

    if out_f:
        out_f.close()

    if store:
        run_dir = store_run(
            predictions=prediction_records,
            corpus_path=jsonl_path,
            model=model,
            profile=profile_name,
            device=device,
            baseline_used=baseline,
            started_at=started_at,
            multi_invariant=multi_invariant,
        )
        return run_dir

    return None
