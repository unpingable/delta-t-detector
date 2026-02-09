"""
Δt Hallucination Detector - GPU Integration Tests

These tests require a CUDA GPU with enough VRAM for Qwen2.5-3B-Instruct.
They exercise the real model end-to-end and are skipped when CUDA is
unavailable.

Run:
    pytest detector/tests_gpu.py -v
"""

import json
import pytest
import numpy as np

try:
    import torch
    _cuda_available = torch.cuda.is_available()
except ImportError:
    _cuda_available = False

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _cuda_available, reason="CUDA GPU not available"),
]

# ---------------------------------------------------------------------------
# Module-scoped fixtures (load model + eval data once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def detector():
    from detector import DeltaTDetector
    from detector.config import DetectorConfig
    config = DetectorConfig()
    config.device = "cuda"
    d = DeltaTDetector(model_name="Qwen/Qwen2.5-3B-Instruct", config=config)
    return d


@pytest.fixture(scope="module")
def eval_prompts():
    import os, json
    seed_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "eval_seed.jsonl",
    )
    prompts = []
    with open(seed_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


# ============================================================================
# TestGenerateWithMetrics
# ============================================================================

class TestGenerateWithMetrics:
    """Test low-level token generation with metrics."""

    MAX_TOKENS = 40

    def test_trace_shape(self, detector):
        """tokens, logprobs, entropies should have the same length"""
        trace = detector.generate_with_metrics(
            "What is 2+2?", max_new_tokens=self.MAX_TOKENS
        )
        assert len(trace.tokens) == len(trace.logprobs) == len(trace.entropies)
        assert len(trace.tokens) > 0

    def test_logprobs_leq_zero(self, detector):
        """All log-probabilities must be <= 0"""
        trace = detector.generate_with_metrics(
            "Name three colors.", max_new_tokens=self.MAX_TOKENS
        )
        assert all(lp <= 0.0 for lp in trace.logprobs)

    def test_entropies_geq_zero(self, detector):
        """Entropies must be >= 0"""
        trace = detector.generate_with_metrics(
            "What is the capital of France?", max_new_tokens=self.MAX_TOKENS
        )
        assert all(e >= 0.0 for e in trace.entropies)

    def test_confidence_range(self, detector):
        """Confidence scores must be in (0, 1]"""
        trace = detector.generate_with_metrics(
            "Explain gravity briefly.", max_new_tokens=self.MAX_TOKENS
        )
        for c in trace.confidence:
            assert 0.0 < c <= 1.0

    def test_text_nonempty(self, detector):
        """Generated text should not be empty"""
        trace = detector.generate_with_metrics(
            "Say hello.", max_new_tokens=self.MAX_TOKENS
        )
        assert len(trace.text.strip()) > 0


# ============================================================================
# TestComputeFeatures
# ============================================================================

class TestComputeFeatures:
    """Test feature extraction on a real model."""

    def test_returns_temporal_features(self, detector):
        """compute_features should return a TemporalFeatures object"""
        from detector.features import TemporalFeatures
        features = detector.compute_features(
            "What is the speed of light?", max_new_tokens=40
        )
        assert isinstance(features, TemporalFeatures)

    def test_positive_generation_length(self, detector):
        """generation_length must be > 0"""
        features = detector.compute_features(
            "Name a planet.", max_new_tokens=40
        )
        assert features.generation_length > 0

    def test_plausible_confidence_range(self, detector):
        """mean_confidence should be in (0, 1]"""
        features = detector.compute_features(
            "What language is Python named after?", max_new_tokens=40
        )
        assert 0.0 < features.mean_confidence <= 1.0

    def test_nonempty_generation(self, detector):
        """generation text should be non-empty"""
        features = detector.compute_features(
            "What is water made of?", max_new_tokens=40
        )
        assert len(features.generation.strip()) > 0

    def test_to_dict_completeness(self, detector):
        """to_dict should include all feature_columns plus metadata"""
        features = detector.compute_features(
            "Who wrote Hamlet?", max_new_tokens=40
        )
        d = features.to_dict()
        from detector.features import TemporalFeatures
        for col in TemporalFeatures.feature_columns():
            assert col in d


# ============================================================================
# TestDetectEndToEnd
# ============================================================================

class TestDetectEndToEnd:
    """End-to-end detect() tests."""

    def test_valid_prediction(self, detector):
        """prediction should be one of the expected labels"""
        result = detector.detect("What is the capital of Japan?")
        assert result.prediction in ('truthful', 'hallucination', 'uncertain', 'unknown')

    def test_confidence_range(self, detector):
        """confidence should be in [0, 1]"""
        result = detector.detect("What is photosynthesis?")
        assert 0.0 <= result.confidence <= 1.0

    def test_temporal_debt_range(self, detector):
        """temporal_debt should be in [0, 1]"""
        result = detector.detect("How many continents are there?")
        assert 0.0 <= result.temporal_debt <= 1.0

    def test_features_populated(self, detector):
        """features dict should be non-empty"""
        result = detector.detect("What is an electron?")
        assert len(result.features) > 0
        assert 'max_confidence_slope' in result.features

    def test_report_attached(self, detector):
        """detect() should attach a report by default"""
        result = detector.detect("What is DNA?")
        assert result.report is not None
        assert result.report.prediction == result.prediction

    def test_all_seed_prompts_no_crash(self, detector, eval_prompts):
        """All 5 eval seed prompts should complete without error"""
        for entry in eval_prompts:
            result = detector.detect(entry['prompt'])
            assert result.prediction in ('truthful', 'hallucination', 'uncertain', 'unknown')


# ============================================================================
# TestDetectMultiInvariant
# ============================================================================

class TestDetectMultiInvariant:
    """Multi-invariant detection tests."""

    def test_valid_result(self, detector):
        """detect_multi_invariant should return a valid result"""
        result = detector.detect_multi_invariant(
            "What is the boiling point of water?",
            validate_citations=False,
            test_semantic=False,
        )
        assert result.prediction in ('truthful', 'hallucination', 'uncertain', 'unknown')
        assert 0.0 <= result.confidence <= 1.0

    def test_report_has_invariant_entries(self, detector):
        """Report should contain at least temporal_coherence invariant"""
        result = detector.detect_multi_invariant(
            "Name the largest ocean.",
            validate_citations=False,
            test_semantic=False,
        )
        assert result.report is not None
        assert 'temporal_coherence' in result.report.invariant_results
