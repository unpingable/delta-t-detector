"""
Δt Hallucination Detector - Governor Signal Emitter

Produces the DetectorSignalFile JSON consumed by agent_gov's
detector_integration.py.  The detector stays zero-knowledge about governor
internals: it writes a JSON file, the governor reads it.
"""

import hashlib
import json
from typing import Any, Dict, Optional

from .features import TemporalFeatures

GOVERNOR_SIGNAL_VERSION = "1.0"

# Exactly 19 raw fields the governor expects
_RAW_FIELD_ORDER = [
    "temporal_debt",
    "confidence_slope",
    "acceleration",
    "max_entropy_jump",
    "entropy_recovery_detected",
    "entropy_variance",
    "token_jaccard",
    "perturbation_sensitivity",
    "tokens_to_high_conf",
    "confidence_monotonicity",
    "early_phase_entropy",
    "mid_phase_entropy",
    "late_phase_entropy",
    "early_phase_confidence",
    "mid_phase_confidence",
    "late_phase_confidence",
    "mean_logprob",
    "response_length",
    "unique_token_ratio",
]


def _full_sha256(text: str) -> str:
    """Full 64-char hex SHA-256 digest (governor expects untruncated)."""
    return hashlib.sha256(text.encode()).hexdigest()


def _build_raw(features: TemporalFeatures, temporal_debt: float) -> Dict[str, float]:
    """
    Map detector TemporalFeatures to the governor's 19-field ``raw`` dict.

    Field renaming is intentional — the governor schema uses shorter names.
    """
    return {
        "temporal_debt": float(temporal_debt),
        "confidence_slope": float(features.max_confidence_slope),
        "acceleration": float(features.confidence_acceleration),
        "max_entropy_jump": float(features.max_entropy_jump),
        "entropy_recovery_detected": float(features.entropy_recovery_detected),
        "entropy_variance": float(features.entropy_variance),
        "token_jaccard": float(features.token_jaccard),
        "perturbation_sensitivity": float(features.perturbation_sensitivity),
        "tokens_to_high_conf": float(features.tokens_to_high_conf),
        "confidence_monotonicity": float(features.confidence_monotonicity),
        "early_phase_entropy": float(features.early_phase_entropy),
        "mid_phase_entropy": float(features.middle_phase_entropy),
        "late_phase_entropy": float(features.late_phase_entropy),
        "early_phase_confidence": float(features.early_phase_confidence),
        "mid_phase_confidence": float(features.middle_phase_confidence),
        "late_phase_confidence": float(features.late_phase_confidence),
        "mean_logprob": float(features.mean_logprob),
        "response_length": float(features.generation_length),
        "unique_token_ratio": float(features.unique_token_ratio),
    }


def build_governor_signal(
    result: Any,
    run_id: str = "",
    turn_id: str = "",
    model_id: str = "",
    profile: str = "general",
) -> Dict[str, Any]:
    """
    Build a full ``DetectorSignalFile`` dict from a DetectionResult.

    Parameters
    ----------
    result : DetectionResult
        The detection result (must have .features dict, .temporal_debt,
        .prediction, .confidence, and .report with .generation).
    run_id, turn_id : str
        Optional identifiers for the governor session / turn.
    model_id : str
        Model identifier string.
    profile : str
        Risk profile name used for this detection.

    Returns
    -------
    dict
        Ready-to-serialize DetectorSignalFile.
    """
    # Reconstruct TemporalFeatures from the result's feature dict so we can
    # read the typed fields used by _build_raw().
    feat_dict = result.features
    generation_text = feat_dict.get("generation", "")
    features_obj = TemporalFeatures(**feat_dict)

    raw = _build_raw(features_obj, result.temporal_debt)

    return {
        "key": {
            "run_id": run_id,
            "turn_id": turn_id,
            "model_id": model_id,
            "response_hash": _full_sha256(generation_text),
        },
        "raw": raw,
        "collapsed": None,
        "detector_version": "2.0.0",
        "profile": profile,
    }


def write_governor_signal(signal: Dict[str, Any], path: str) -> None:
    """Write a governor signal dict to *path* as JSON."""
    with open(path, "w") as f:
        json.dump(signal, f, indent=2)
