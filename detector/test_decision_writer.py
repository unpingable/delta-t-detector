"""Tests for decision_writer.py — controller_decision/v1 serialization."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from detector.decision_writer import (
    SCHEMA,
    _map_controller_action,
    _map_final_status,
    _map_handoff_recommendation,
    _map_oracle_result,
    _map_failure_kind,
    _to_milliunits,
    build_controller_decision,
    compute_output_hash,
    write_decision,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_record(
    policy: str = "FAST_PATH",
    fork_risk: float | None = 0.08,
    tau: float = 0.05,
    n_id_windows: int = 2,
    final_verdict: str = "CLEAN",
    ms1: float = 800.0,
) -> dict:
    return {
        "prompt_id": "test-1",
        "fork_risk": fork_risk,
        "tau": tau,
        "n_id_windows": n_id_windows,
        "policy": policy,
        "attempt1": {
            "verdict": "CLEAN" if policy != "CONFIDENT_WRONG" else "FAIL",
            "fail_type": None,
            "fail_subjects": None,
            "margin_stats": {"m_min": fork_risk, "m_mean": 0.12},
            "infer_ms": ms1,
        },
        "final_verdict": final_verdict,
    }


def _make_result(
    verdict: str = "CLEAN",
    fail_type: str | None = None,
    fail_subjects: list | None = None,
) -> dict:
    return {
        "eg": {"details": {}},
        "verdict": verdict,
        "confidence": 0.9,
        "fail_type": fail_type,
        "fail_subjects": fail_subjects,
        "warn_type": None,
    }


# =============================================================================
# Milliunits conversion
# =============================================================================


class TestToMilliunits:

    def test_basic(self):
        assert _to_milliunits(0.05) == 50
        assert _to_milliunits(0.03) == 30

    def test_none(self):
        assert _to_milliunits(None) is None

    def test_round_half_to_even(self):
        # Python's round() uses banker's rounding
        assert _to_milliunits(0.0005) == 0  # 0.5 rounds to 0 (even)
        assert _to_milliunits(0.0015) == 2  # 1.5 rounds to 2 (even)
        assert _to_milliunits(0.0025) == 2  # 2.5 rounds to 2 (even)
        assert _to_milliunits(0.0035) == 4  # 3.5 rounds to 4 (even)

    def test_exact_values(self):
        assert _to_milliunits(1.0) == 1000
        assert _to_milliunits(0.0) == 0
        assert _to_milliunits(0.123) == 123


# =============================================================================
# Vocabulary mapping
# =============================================================================


class TestMapControllerAction:

    def test_fast_path(self):
        assert _map_controller_action("FAST_PATH") == "FAST_PATH"

    def test_low_margin_retry(self):
        assert _map_controller_action("LOW_MARGIN_RETRY") == "LOW_MARGIN_RETRY"

    def test_confident_wrong_becomes_stop(self):
        assert _map_controller_action("CONFIDENT_WRONG") == "STOP"

    def test_unknown_passthrough(self):
        assert _map_controller_action("UNKNOWN") == "UNKNOWN"


class TestMapFinalStatus:

    def test_fast_path_clean(self):
        assert _map_final_status("FAST_PATH", "CLEAN") == "CLEAN"

    def test_fast_path_warn(self):
        assert _map_final_status("FAST_PATH", "WARN") == "CLEAN"

    def test_fast_path_fail(self):
        assert _map_final_status("FAST_PATH", "FAIL") == "KNOWLEDGE_BOUNDARY"

    def test_low_margin_retry_clean(self):
        assert _map_final_status("LOW_MARGIN_RETRY", "CLEAN") == "LOW_MARGIN_RECOVERED"

    def test_low_margin_retry_warn(self):
        assert _map_final_status("LOW_MARGIN_RETRY", "WARN") == "LOW_MARGIN_RECOVERED"

    def test_low_margin_retry_fail(self):
        assert _map_final_status("LOW_MARGIN_RETRY", "FAIL") == "KNOWLEDGE_BOUNDARY"

    def test_confident_wrong(self):
        assert _map_final_status("CONFIDENT_WRONG", "FAIL") == "CONFIDENT_WRONG"

    def test_oracle_error(self):
        assert _map_final_status("FAST_PATH", "CLEAN", oracle_error=True) == "ORACLE_ERROR"
        assert _map_final_status("LOW_MARGIN_RETRY", "FAIL", oracle_error=True) == "ORACLE_ERROR"


class TestMapHandoffRecommendation:

    def test_clean_no_handoff(self):
        assert _map_handoff_recommendation("CLEAN") == "NONE"

    def test_recovered_no_handoff(self):
        assert _map_handoff_recommendation("LOW_MARGIN_RECOVERED") == "NONE"

    def test_confident_wrong_escalate(self):
        assert _map_handoff_recommendation("CONFIDENT_WRONG") == "ESCALATE_REMOTE"

    def test_knowledge_boundary_escalate(self):
        assert _map_handoff_recommendation("KNOWLEDGE_BOUNDARY") == "ESCALATE_REMOTE"

    def test_oracle_error_escalate_model(self):
        assert _map_handoff_recommendation("ORACLE_ERROR") == "ESCALATE_MODEL"


class TestMapOracleResult:

    def test_clean_pass(self):
        assert _map_oracle_result("CLEAN") == "PASS"

    def test_fail_fail(self):
        assert _map_oracle_result("FAIL") == "FAIL"

    def test_warn_fail(self):
        assert _map_oracle_result("WARN") == "FAIL"

    def test_unknown_error(self):
        assert _map_oracle_result("UNKNOWN") == "ERROR"


class TestMapFailureKind:

    def test_fabricated(self):
        assert _map_failure_kind("fabricated") == "NOT_FOUND"

    def test_not_found(self):
        assert _map_failure_kind("not_found") == "NOT_FOUND"

    def test_evasion(self):
        assert _map_failure_kind("evasion") == "MALFORMED"

    def test_none(self):
        assert _map_failure_kind(None) is None

    def test_unknown(self):
        assert _map_failure_kind("unknown_type") == "NOT_FOUND"


# =============================================================================
# Output hash
# =============================================================================


class TestComputeOutputHash:

    def test_deterministic(self):
        h1 = compute_output_hash("hello world")
        h2 = compute_output_hash("hello world")
        assert h1 == h2

    def test_utf8_encoding(self):
        text = "hello world"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert compute_output_hash(text) == expected

    def test_different_text_different_hash(self):
        assert compute_output_hash("a") != compute_output_hash("b")


# =============================================================================
# build_controller_decision
# =============================================================================


class TestBuildControllerDecision:

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_basic_clean_decision(self, mock_sha):
        record = _make_record()
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="ANSWER: requests is a Python HTTP library.\nREFS:\n1. pypi:requests==2.31.0",
            final_result=result,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            task_id="test-1",
            prompt_text="What is the requests library?",
        )
        assert decision["schema"] == SCHEMA
        assert decision["task"]["task_id"] == "test-1"
        assert decision["task"]["task_class"] == "citation"
        assert decision["task"]["namespace"] == "pypi"
        assert len(decision["task"]["prompt_hash"]) == 64
        assert len(decision["task"]["output_hash"]) == 64
        assert decision["decision"]["controller_action"] == "FAST_PATH"
        assert decision["decision"]["final_status"] == "CLEAN"
        assert decision["decision"]["fork_risk"]["value_milliunits"] == 80
        assert decision["decision"]["tau_milliunits"] == 50
        assert decision["local_model"]["model_id"] == "Qwen/Qwen2.5-3B-Instruct"
        assert decision["local_model"]["controller_version"] == "abc123"

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_confident_wrong_becomes_stop(self, mock_sha):
        record = _make_record(policy="CONFIDENT_WRONG", final_verdict="FAIL")
        result = _make_result(verdict="FAIL", fail_type="fabricated", fail_subjects=["flask==9.9.9"])
        decision = build_controller_decision(
            record=record,
            final_text="fake output",
            final_result=result,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            task_id="test-2",
            prompt_text="test prompt",
        )
        assert decision["decision"]["controller_action"] == "STOP"
        assert decision["decision"]["final_status"] == "CONFIDENT_WRONG"
        assert decision["decision"]["handoff_recommendation"] == "ESCALATE_REMOTE"

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_low_margin_retry_recovered(self, mock_sha):
        record = _make_record(
            policy="LOW_MARGIN_RETRY",
            fork_risk=0.03,
            final_verdict="CLEAN",
        )
        record["attempt2"] = {
            "verdict": "CLEAN",
            "fail_type": None,
            "fail_subjects": None,
            "margin_stats": {"m_min": 0.12},
            "infer_ms": 900.0,
        }
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="recovered output",
            final_result=result,
            model_id="Qwen/Qwen2.5-3B-Instruct",
            task_id="test-3",
            prompt_text="test prompt",
        )
        assert decision["decision"]["controller_action"] == "LOW_MARGIN_RETRY"
        assert decision["decision"]["final_status"] == "LOW_MARGIN_RECOVERED"
        assert decision["economics"]["retry_count"] == 1

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_governor_bindings_included(self, mock_sha):
        record = _make_record()
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="output",
            final_result=result,
            model_id="test-model",
            task_id="test-4",
            prompt_text="prompt",
            request_nonce="nonce-abc",
            oracle_profile="pypi_locked",
            decision_ttl_ms=30000,
        )
        assert decision["task"]["request_nonce"] == "nonce-abc"
        assert decision["task"]["oracle_profile"] == "pypi_locked"
        assert decision["decision"]["decision_ttl_ms"] == 30000

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_no_bindings_when_not_provided(self, mock_sha):
        record = _make_record()
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="output",
            final_result=result,
            model_id="test-model",
            task_id="test-5",
            prompt_text="prompt",
        )
        assert "request_nonce" not in decision["task"]
        assert "oracle_profile" not in decision["task"]
        assert "decision_ttl_ms" not in decision["decision"]

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_oracle_error(self, mock_sha):
        record = _make_record()
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="output",
            final_result=result,
            model_id="test-model",
            task_id="test-6",
            prompt_text="prompt",
            oracle_error=True,
        )
        assert decision["decision"]["final_status"] == "ORACLE_ERROR"
        assert decision["decision"]["handoff_recommendation"] == "ESCALATE_MODEL"
        assert decision["evidence"]["oracle_checks"][0]["result"] == "ERROR"

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_output_hash_matches_text(self, mock_sha):
        text = "the actual model output"
        record = _make_record()
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text=text,
            final_result=result,
            model_id="test-model",
            task_id="test-7",
            prompt_text="prompt",
        )
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert decision["task"]["output_hash"] == expected_hash

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_null_fork_risk(self, mock_sha):
        record = _make_record(fork_risk=None, n_id_windows=0)
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="output",
            final_result=result,
            model_id="test-model",
            task_id="test-8",
            prompt_text="prompt",
        )
        assert decision["decision"]["fork_risk"]["value_milliunits"] is None

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_all_six_sections_present(self, mock_sha):
        record = _make_record()
        result = _make_result()
        decision = build_controller_decision(
            record=record,
            final_text="output",
            final_result=result,
            model_id="test-model",
            task_id="test-9",
            prompt_text="prompt",
        )
        for section in ("schema", "task", "local_model", "decision", "evidence", "economics", "created_at"):
            assert section in decision, f"Missing section: {section}"

    @patch("detector.decision_writer.git_head_sha", return_value="abc123")
    def test_failures_populated_on_fail(self, mock_sha):
        record = _make_record(policy="CONFIDENT_WRONG", final_verdict="FAIL")
        result = _make_result(
            verdict="FAIL",
            fail_type="fabricated",
            fail_subjects=["flask==9.9.9", "django==99.0"],
        )
        decision = build_controller_decision(
            record=record,
            final_text="output",
            final_result=result,
            model_id="test-model",
            task_id="test-10",
            prompt_text="prompt",
        )
        failures = decision["evidence"]["failures"]
        assert len(failures) == 2
        assert failures[0]["kind"] == "NOT_FOUND"
        assert "flask" in failures[0]["message"]


# =============================================================================
# write_decision
# =============================================================================


class TestWriteDecision:

    def test_writes_file(self, tmp_path):
        decision = {"schema": SCHEMA, "task": {"task_id": "test-1"}}
        path = write_decision(decision, tmp_path)
        assert path.exists()
        loaded = json.loads(path.read_text())
        assert loaded["schema"] == SCHEMA

    def test_creates_directory(self, tmp_path):
        out_dir = tmp_path / "sub" / "dir"
        decision = {"schema": SCHEMA, "task": {"task_id": "test-1"}}
        path = write_decision(decision, out_dir)
        assert path.exists()

    def test_filename_contains_task_id(self, tmp_path):
        decision = {"schema": SCHEMA, "task": {"task_id": "my-task"}}
        path = write_decision(decision, tmp_path, task_id="my-task")
        assert "my-task" in path.name

    def test_sanitizes_task_id(self, tmp_path):
        decision = {"schema": SCHEMA, "task": {"task_id": "bad/id with spaces"}}
        path = write_decision(decision, tmp_path, task_id="bad/id with spaces")
        assert "/" not in path.name
        assert " " not in path.name
