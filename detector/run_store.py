"""
Δt Hallucination Detector - Run Store
Append-only, hash-pinned eval run storage.

Every run bundle is immutable once written: corpus hash + git commit +
profile + model together derive the run_id, and the predictions file
is SHA-256 verified against the manifest.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Optional


RUN_STORE_VERSION = "1.0"

_RUNS_DIR = "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def corpus_hash(path: str) -> str:
    """SHA-256 of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head_sha() -> str:
    """Current HEAD commit (or 'unknown')."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """Working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return len(out) > 0
    except Exception:
        return False


def derive_run_id(corpus_sha: str, git_commit: str, profile: str, model: str) -> str:
    """First 12 hex chars of SHA-256(corpus_sha|git_commit|profile|model)."""
    payload = f"{corpus_sha}|{git_commit}|{profile}|{model}"
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PredictionRecord:
    """One eval item's result."""
    id: str
    prompt: str
    expected_risk: str
    prediction: str
    confidence: float
    temporal_debt: float
    anomaly_score: Optional[float]
    features: Dict[str, Any]
    guard_flags: Dict[str, bool]
    notes: str = ""
    # Multi-invariant fields (None for single-invariant runs)
    invariant_results: Optional[Dict[str, Dict[str, Any]]] = None
    n_violated: Optional[int] = None
    aggregate_score: Optional[float] = None
    # FAIL classification
    fail_type: Optional[str] = None          # 'FABRICATED_IDENTIFIER' or None
    fail_subjects: Optional[List[str]] = None  # e.g. ['doi:10.xxxx']


@dataclass
class RunManifest:
    """Immutable run metadata — written last as the crash-safety gate."""
    run_store_version: str
    run_id: str
    started_at: str
    finished_at: str
    corpus_hash: str
    corpus_path: str
    predictions_hash: str
    model: str
    profile: str
    device: str
    baseline_used: bool
    detector_version: str
    git_commit: str
    git_dirty: bool
    corpus_size: int
    multi_invariant: bool = False


@dataclass
class RunSummary:
    """Aggregate statistics for a run."""
    prediction_counts: Dict[str, int]
    mean_confidence: float
    median_confidence: float
    mean_temporal_debt: float
    median_temporal_debt: float
    guard_flag_counts: Dict[str, int]
    high_risk_recall: Optional[float]
    low_risk_precision: Optional[float]
    confusion_matrix: Dict[str, Dict[str, int]]
    # Multi-invariant (empty for single-invariant runs)
    invariant_violation_counts: Dict[str, int] = field(default_factory=dict)
    invariant_mean_scores: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_summary(predictions: List[PredictionRecord]) -> RunSummary:
    pred_counts: Dict[str, int] = {}
    confs: List[float] = []
    debts: List[float] = []
    flag_counts: Dict[str, int] = {}

    # confusion: {expected: {predicted: count}}
    confusion: Dict[str, Dict[str, int]] = {}

    # Multi-invariant aggregation
    inv_violation_counts: Dict[str, int] = {}
    inv_scores: Dict[str, List[float]] = {}

    for p in predictions:
        pred_counts[p.prediction] = pred_counts.get(p.prediction, 0) + 1
        confs.append(p.confidence)
        debts.append(p.temporal_debt)

        for flag_name, flag_val in p.guard_flags.items():
            if flag_val:
                flag_counts[flag_name] = flag_counts.get(flag_name, 0) + 1

        if p.expected_risk not in confusion:
            confusion[p.expected_risk] = {}
        bucket = confusion[p.expected_risk]
        bucket[p.prediction] = bucket.get(p.prediction, 0) + 1

        if p.invariant_results:
            for inv_name, inv_data in p.invariant_results.items():
                if inv_data.get("violated"):
                    inv_violation_counts[inv_name] = inv_violation_counts.get(inv_name, 0) + 1
                score = inv_data.get("score")
                if score is not None:
                    inv_scores.setdefault(inv_name, []).append(score)

    # High-risk recall: of items expected high risk, how many predicted unstable/FAIL?
    high_risk_labels = {"high", "hallucination", "temporal_unstable"}
    unstable_preds = {"temporal_unstable", "hallucination", "FAIL", "high"}
    hr_total = sum(
        ct
        for exp, preds in confusion.items()
        if exp in high_risk_labels
        for ct in preds.values()
    )
    hr_correct = sum(
        ct
        for exp, preds in confusion.items()
        if exp in high_risk_labels
        for pred, ct in preds.items()
        if pred in unstable_preds
    )
    high_risk_recall = hr_correct / hr_total if hr_total > 0 else None

    # Low-risk precision: of items predicted stable, how many actually low risk?
    stable_preds = {"temporal_stable", "truthful", "CLEAN", "low"}
    low_risk_labels = {"low", "truthful", "temporal_stable"}
    lr_predicted = sum(
        ct
        for preds in confusion.values()
        for pred, ct in preds.items()
        if pred in stable_preds
    )
    lr_correct = sum(
        ct
        for exp, preds in confusion.items()
        if exp in low_risk_labels
        for pred, ct in preds.items()
        if pred in stable_preds
    )
    low_risk_precision = lr_correct / lr_predicted if lr_predicted > 0 else None

    inv_mean_scores = {name: mean(scores) for name, scores in inv_scores.items()}

    return RunSummary(
        prediction_counts=pred_counts,
        mean_confidence=mean(confs) if confs else 0.0,
        median_confidence=median(confs) if confs else 0.0,
        mean_temporal_debt=mean(debts) if debts else 0.0,
        median_temporal_debt=median(debts) if debts else 0.0,
        guard_flag_counts=flag_counts,
        high_risk_recall=high_risk_recall,
        low_risk_precision=low_risk_precision,
        confusion_matrix=confusion,
        invariant_violation_counts=inv_violation_counts,
        invariant_mean_scores=inv_mean_scores,
    )


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def store_run(
    predictions: List[PredictionRecord],
    corpus_path: str,
    model: str,
    profile: str,
    device: str = "auto",
    baseline_used: bool = False,
    runs_root: Optional[str] = None,
    started_at: Optional[str] = None,
    multi_invariant: bool = False,
) -> Path:
    """Write a complete run bundle.  Returns the run directory path.

    Write order (crash-safety): predictions.jsonl -> summary.json -> manifest.json.
    Manifest written *last* — its presence signals a complete run.
    """
    from detector import __version__ as detector_version

    if runs_root is None:
        runs_root = _RUNS_DIR

    c_hash = corpus_hash(corpus_path)
    commit = git_head_sha()
    dirty = git_dirty()
    run_id = derive_run_id(c_hash, commit, profile, model)

    ts = _utcnow_iso()
    dir_name = f"{run_id}_{ts}"
    run_dir = Path(runs_root) / dir_name

    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)

    # 1. predictions.jsonl
    pred_path = run_dir / "predictions.jsonl"
    pred_lines: List[bytes] = []
    for p in predictions:
        line = json.dumps(asdict(p), default=str).encode()
        pred_lines.append(line)
    pred_bytes = b"\n".join(pred_lines) + b"\n" if pred_lines else b""
    pred_path.write_bytes(pred_bytes)
    p_hash = _sha256_bytes(pred_bytes)

    # 2. summary.json
    summary = _compute_summary(predictions)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2, default=str))

    # 3. manifest.json (last — crash-safety gate)
    manifest = RunManifest(
        run_store_version=RUN_STORE_VERSION,
        run_id=run_id,
        started_at=started_at or ts,
        finished_at=ts,
        corpus_hash=c_hash,
        corpus_path=corpus_path,
        predictions_hash=p_hash,
        model=model,
        profile=profile,
        device=device,
        baseline_used=baseline_used,
        detector_version=detector_version,
        git_commit=commit,
        git_dirty=dirty,
        corpus_size=len(predictions),
        multi_invariant=multi_invariant,
    )
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2, default=str))

    return run_dir


def list_runs(runs_root: Optional[str] = None) -> List[dict]:
    """Scan runs/ for directories containing manifest.json.

    Returns list of manifest dicts, sorted by finished_at descending.
    """
    if runs_root is None:
        runs_root = _RUNS_DIR

    root = Path(runs_root)
    if not root.is_dir():
        return []

    manifests = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        mf = entry / "manifest.json"
        if mf.exists():
            manifests.append(json.loads(mf.read_text()))

    manifests.sort(key=lambda m: m.get("finished_at", ""), reverse=True)
    return manifests


def load_run(run_dir: str) -> dict:
    """Load manifest + predictions + summary from a run directory."""
    rd = Path(run_dir)

    manifest = json.loads((rd / "manifest.json").read_text())

    predictions = []
    pred_path = rd / "predictions.jsonl"
    for line in pred_path.read_text().splitlines():
        line = line.strip()
        if line:
            predictions.append(json.loads(line))

    summary = json.loads((rd / "summary.json").read_text())

    return {
        "manifest": manifest,
        "predictions": predictions,
        "summary": summary,
    }


def verify_run(run_dir: str) -> bool:
    """Recompute predictions_hash and check against manifest."""
    rd = Path(run_dir)
    manifest = json.loads((rd / "manifest.json").read_text())
    pred_bytes = (rd / "predictions.jsonl").read_bytes()
    actual = _sha256_bytes(pred_bytes)
    return actual == manifest["predictions_hash"]


def diff_runs(dir_a: str, dir_b: str) -> dict:
    """Compare two runs: config changes, prediction changes, summary deltas."""
    a = load_run(dir_a)
    b = load_run(dir_b)

    # Config changes
    config_keys = [
        "model", "profile", "device", "baseline_used",
        "corpus_hash", "git_commit", "detector_version",
    ]
    config_changes = {}
    for key in config_keys:
        va = a["manifest"].get(key)
        vb = b["manifest"].get(key)
        if va != vb:
            config_changes[key] = {"a": va, "b": vb}

    # Prediction changes (match by id)
    preds_a = {p["id"]: p for p in a["predictions"]}
    preds_b = {p["id"]: p for p in b["predictions"]}
    all_ids = sorted(set(preds_a) | set(preds_b))

    prediction_changes = []
    for pid in all_ids:
        pa = preds_a.get(pid)
        pb = preds_b.get(pid)
        if pa is None:
            prediction_changes.append({"id": pid, "change": "added_in_b"})
        elif pb is None:
            prediction_changes.append({"id": pid, "change": "removed_in_b"})
        elif pa.get("prediction") != pb.get("prediction"):
            prediction_changes.append({
                "id": pid,
                "change": "prediction_changed",
                "a": pa.get("prediction"),
                "b": pb.get("prediction"),
            })

    # Summary deltas
    summary_deltas = {}
    numeric_keys = [
        "mean_confidence", "median_confidence",
        "mean_temporal_debt", "median_temporal_debt",
        "high_risk_recall", "low_risk_precision",
    ]
    for key in numeric_keys:
        va = a["summary"].get(key)
        vb = b["summary"].get(key)
        if va is not None and vb is not None:
            summary_deltas[key] = {"a": va, "b": vb, "delta": vb - va}
        elif va != vb:
            summary_deltas[key] = {"a": va, "b": vb}

    return {
        "config_changes": config_changes,
        "prediction_changes": prediction_changes,
        "summary_deltas": summary_deltas,
    }
