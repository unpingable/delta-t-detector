"""
Controller Decision Writer — controller_decision/v1 serialization.

Serializes the 3-way controller's output into the governor handoff contract
format (controller_decision/v1). Write-only; does not change controller behavior.

Spec: agent_gov/specs/core/DETECTOR_HANDOFF_SPEC.md
Contract: one JSON file per task, read by governor's DetectorHandoffGate.

Key rules:
  - Floats in hashable fields → integer milliunits (round-half-to-even)
  - output_hash = SHA-256 of UTF-8 encoded model output text
  - If governor didn't pass a binding (nonce, oracle_profile), don't invent defaults
  - controller_version = git SHA of this repo
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from detector.run_store import git_head_sha


# =============================================================================
# Constants
# =============================================================================

SCHEMA = "controller_decision/v1"


# =============================================================================
# Vocabulary mapping: controller internals → v1 enums
# =============================================================================

def _to_milliunits(value: float | None) -> int | None:
    """Convert float to integer milliunits using round-half-to-even (Python default)."""
    if value is None:
        return None
    return round(value * 1000)


def _map_controller_action(policy: str) -> str:
    """Map controller's internal policy name to v1 controller_action.

    Controller vocabulary: FAST_PATH, LOW_MARGIN_RETRY, CONFIDENT_WRONG
    v1 vocabulary: FAST_PATH, LOW_MARGIN_RETRY, GROUND, STOP

    CONFIDENT_WRONG is a diagnosis (final_status), not an action.
    The action that produced CONFIDENT_WRONG is STOP (halt local attempt).
    """
    mapping = {
        "FAST_PATH": "FAST_PATH",
        "LOW_MARGIN_RETRY": "LOW_MARGIN_RETRY",
        "CONFIDENT_WRONG": "STOP",
    }
    return mapping.get(policy, policy)


def _map_final_status(policy: str, final_verdict: str, oracle_error: bool = False) -> str:
    """Map controller's (policy, final_verdict) pair to v1 final_status.

    Controller verdicts: CLEAN, WARN, FAIL
    v1 final_status: CLEAN, LOW_MARGIN_RECOVERED, CONFIDENT_WRONG,
                     KNOWLEDGE_BOUNDARY, ORACLE_ERROR

    Logic:
      - oracle_error=True → ORACLE_ERROR (oracle itself failed to execute)
      - policy=LOW_MARGIN_RETRY + verdict=CLEAN/WARN → LOW_MARGIN_RECOVERED
      - policy=CONFIDENT_WRONG (verdict must be FAIL) → CONFIDENT_WRONG
      - policy=FAST_PATH + verdict=FAIL → KNOWLEDGE_BOUNDARY
        (high margin but wrong = model doesn't know this namespace)
      - policy=FAST_PATH + verdict=CLEAN/WARN → CLEAN
    """
    if oracle_error:
        return "ORACLE_ERROR"

    if policy == "LOW_MARGIN_RETRY" and final_verdict in ("CLEAN", "WARN"):
        return "LOW_MARGIN_RECOVERED"

    if policy == "CONFIDENT_WRONG":
        return "CONFIDENT_WRONG"

    if policy == "LOW_MARGIN_RETRY" and final_verdict == "FAIL":
        # Retried at greedy, still failed → knowledge boundary
        return "KNOWLEDGE_BOUNDARY"

    if policy == "FAST_PATH" and final_verdict == "FAIL":
        # High margin, oracle fails → model is confidently wrong but we
        # didn't detect it via margin (marginal edge case). Still block.
        return "KNOWLEDGE_BOUNDARY"

    # FAST_PATH + CLEAN/WARN → CLEAN
    return "CLEAN"


def _map_handoff_recommendation(final_status: str) -> str:
    """Derive handoff_recommendation from final_status."""
    mapping = {
        "CLEAN": "NONE",
        "LOW_MARGIN_RECOVERED": "NONE",
        "CONFIDENT_WRONG": "ESCALATE_REMOTE",
        "KNOWLEDGE_BOUNDARY": "ESCALATE_REMOTE",
        "ORACLE_ERROR": "ESCALATE_MODEL",
    }
    return mapping.get(final_status, "NONE")


def _map_oracle_result(verdict: str) -> str:
    """Map EG verdict to v1 oracle result enum.

    EG verdicts: CLEAN, WARN, FAIL
    v1: PASS, FAIL, ERROR
    """
    if verdict == "CLEAN":
        return "PASS"
    if verdict in ("FAIL", "WARN"):
        return "FAIL"
    return "ERROR"


def _map_failure_kind(fail_type: str | None) -> str | None:
    """Map EG fail_type to v1 failure_kind enum."""
    if not fail_type:
        return None
    # EG fail_types: "fabricated", "not_found", "evasion", "sentinel", "format"
    mapping = {
        "fabricated": "NOT_FOUND",
        "not_found": "NOT_FOUND",
        "evasion": "MALFORMED",
        "sentinel": "NOT_FOUND",
        "format": "MALFORMED",
    }
    return mapping.get(fail_type, "NOT_FOUND")


# =============================================================================
# Output hash
# =============================================================================

def compute_output_hash(text: str) -> str:
    """SHA-256 of UTF-8 encoded model output text.

    This is the exact bytes that the governor will verify.
    No normalization (newlines, whitespace) — hash what was generated.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# =============================================================================
# Core serializer
# =============================================================================

def build_controller_decision(
    *,
    # Controller outputs
    record: Dict[str, Any],
    final_text: str,
    final_result: Dict[str, Any],
    # Model info
    model_id: str,
    quant: str = "none",
    temperature: float = 0.7,
    retry_temperature: float = 0.0,
    seed: int = 42,
    # Task info
    task_id: str,
    task_class: str = "citation",
    namespace: str = "pypi",
    prompt_text: str = "",
    # Governor-provided bindings (DO NOT invent defaults)
    request_nonce: str | None = None,
    oracle_profile: str | None = None,
    decision_ttl_ms: int | None = None,
    # Oracle error flag
    oracle_error: bool = False,
) -> Dict[str, Any]:
    """Build a controller_decision/v1 JSON dict.

    Args:
        record: The controller's internal record dict from run_prompt().
        final_text: The model's final output text (for output_hash).
        final_result: The scoring result dict (verdict, fail_type, etc.).
        model_id: HuggingFace model ID.
        task_id: Unique task identifier (usually prompt_id).
        task_class: Task classification (citation, codegen, qa, ...).
        namespace: Identifier namespace (pypi, cve, doi, rfc, code, ...).
        prompt_text: Original prompt text (for prompt_hash).
        request_nonce: Governor-generated nonce. None if governor didn't provide one.
        oracle_profile: Oracle chain identifier. None if governor didn't specify.
        decision_ttl_ms: TTL in ms. None if governor didn't specify.
        oracle_error: True if oracle itself failed to execute.

    Returns:
        Dict ready for json.dumps().
    """
    policy = record["policy"]
    final_verdict = record.get("final_verdict", final_result.get("verdict", ""))
    fork_risk = record.get("fork_risk")
    tau = record.get("tau", 0.05)

    # Map to v1 vocabulary
    controller_action = _map_controller_action(policy)
    final_status = _map_final_status(policy, final_verdict, oracle_error)
    handoff_recommendation = _map_handoff_recommendation(final_status)

    # Task section
    task: Dict[str, Any] = {
        "task_id": task_id,
        "task_class": task_class,
        "namespace": namespace,
        "prompt_hash": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "output_hash": compute_output_hash(final_text),
    }
    if request_nonce is not None:
        task["request_nonce"] = request_nonce
    if oracle_profile is not None:
        task["oracle_profile"] = oracle_profile

    # Decision section
    decision: Dict[str, Any] = {
        "fork_risk": {
            "metric": "prompt_min_margin",
            "value_milliunits": _to_milliunits(fork_risk),
            "window_k": record.get("n_id_windows", 0),
        },
        "tau_milliunits": _to_milliunits(tau),
        "controller_action": controller_action,
        "final_status": final_status,
        "handoff_recommendation": handoff_recommendation,
    }
    if decision_ttl_ms is not None:
        decision["decision_ttl_ms"] = decision_ttl_ms

    # Evidence section
    candidates = _extract_candidates(final_text, namespace)
    oracle_checks = _build_oracle_checks(final_result, namespace, oracle_error)
    failures = _build_failures(final_result)

    evidence: Dict[str, Any] = {
        "candidates": candidates,
        "oracle_checks": oracle_checks,
        "failures": failures,
    }

    # Economics section
    attempt1 = record.get("attempt1", {})
    attempt2 = record.get("attempt2")
    ms1 = attempt1.get("infer_ms", 0)
    ms2 = attempt2.get("infer_ms", 0) if attempt2 else 0
    oracle_ms = sum(c.get("latency_ms", 0) for c in oracle_checks)

    economics: Dict[str, Any] = {
        "local_tokens_spent": 0,  # Will be filled from trace if available
        "retry_tokens_spent": 0,
        "retry_count": 1 if attempt2 else 0,
        "latency_ms": round(ms1 + ms2 + oracle_ms),
        "oracle_latency_ms": round(oracle_ms),
    }

    # Build complete decision
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "task": task,
        "local_model": {
            "model_id": model_id,
            "quant": quant,
            "decoding": {
                "temperature": temperature,
                "retry_temperature": retry_temperature,
                "seed": seed,
            },
            "controller_version": git_head_sha(),
        },
        "decision": decision,
        "evidence": evidence,
        "economics": economics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return result


def _extract_candidates(text: str, namespace: str) -> List[str]:
    """Extract identifier candidates from model output."""
    try:
        from detector.utils import extract_citations
        citations = extract_citations(text)

        # Map namespace to citation key
        ns_map = {
            "pypi": "pypi",
            "cve": "cves",
            "doi": "dois",
            "rfc": "rfcs",
            "arxiv": "arxiv",
        }
        key = ns_map.get(namespace, namespace)
        raw = citations.get(key, [])

        # Format as namespace:identifier
        prefix_map = {
            "pypi": "pypi:",
            "cve": "CVE-",
            "doi": "doi:",
            "rfc": "RFC ",
            "arxiv": "arxiv:",
        }
        prefix = prefix_map.get(namespace, f"{namespace}:")
        return [f"{prefix}{c}" if not c.startswith(prefix) else c for c in raw]
    except Exception:
        return []


def _build_oracle_checks(
    result: Dict[str, Any],
    namespace: str,
    oracle_error: bool,
) -> List[Dict[str, Any]]:
    """Build oracle_checks from EG scoring result."""
    if oracle_error:
        return [{
            "oracle": f"{namespace}_validator",
            "query": "",
            "result": "ERROR",
            "error_kind": "NETWORK",
            "error": "Oracle execution failed",
            "latency_ms": 0,
        }]

    eg = result.get("eg", {})
    details = eg.get("details", {})

    checks: List[Dict[str, Any]] = []

    # Extract per-anchor validation results from EG details
    validations = details.get("validation_results", {})
    for anchor_type, anchor_results in validations.items():
        if not isinstance(anchor_results, dict):
            continue
        for anchor_id, status in anchor_results.items():
            check: Dict[str, Any] = {
                "oracle": f"{namespace}_validator",
                "query": str(anchor_id),
                "result": "PASS" if status in (True, "valid", "found") else "FAIL",
                "latency_ms": 0,  # EG doesn't track per-check latency
            }
            if check["result"] == "FAIL":
                check["failure_kind"] = "NOT_FOUND"
            checks.append(check)

    # If no per-anchor results, use aggregate verdict
    if not checks:
        verdict = result.get("verdict", "CLEAN")
        check = {
            "oracle": f"{namespace}_validator",
            "query": "aggregate",
            "result": _map_oracle_result(verdict),
            "latency_ms": 0,
        }
        if check["result"] == "FAIL":
            fail_type = result.get("fail_type")
            check["failure_kind"] = _map_failure_kind(fail_type) or "NOT_FOUND"
        checks.append(check)

    return checks


def _build_failures(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build failures list from EG scoring result."""
    failures: List[Dict[str, Any]] = []

    fail_type = result.get("fail_type")
    fail_subjects = result.get("fail_subjects") or []

    if fail_type and result.get("verdict") == "FAIL":
        for i, subject in enumerate(fail_subjects):
            failures.append({
                "kind": _map_failure_kind(fail_type) or "NOT_FOUND",
                "message": f"{fail_type}: {subject}",
                "attempt": 1,
            })

        # If no specific subjects but we have a failure
        if not fail_subjects:
            failures.append({
                "kind": _map_failure_kind(fail_type) or "NOT_FOUND",
                "message": fail_type,
                "attempt": 1,
            })

    return failures


# =============================================================================
# File I/O
# =============================================================================


def write_decision(
    decision: Dict[str, Any],
    output_dir: Path,
    task_id: str | None = None,
) -> Path:
    """Write controller_decision.json to disk.

    Args:
        decision: The v1 decision dict from build_controller_decision().
        output_dir: Directory to write to (will be created).
        task_id: Optional task_id for filename. Defaults to decision's task_id.

    Returns:
        Path to written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tid = task_id or decision.get("task", {}).get("task_id", "unknown")
    # Sanitize for filesystem
    safe_tid = "".join(c if c.isalnum() or c in "-_." else "_" for c in tid)
    path = output_dir / f"controller_decision_{safe_tid}.json"
    with open(path, "w") as f:
        json.dump(decision, f, indent=2, default=str)
    return path
