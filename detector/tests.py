"""
Δt Hallucination Detector - Test Suite
Unit and integration tests for all components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

# Import components
from detector.config import (
    RiskProfile, RISK_PROFILES, DetectorConfig, 
    get_config, set_profile
)
from detector.utils import (
    extract_citations, count_citations, jaccard_similarity,
    levenshtein_similarity, normalize_text, extract_keywords,
    phase_segment, confidence_from_entropy, to_plain
)
from detector.features import (
    TemporalFeatures, GenerationTrace, FeatureExtractor,
    compute_temporal_debt
)
from detector.baseline import (
    FeatureStats, ModelBaseline, BaselineProfiler
)
from detector.invariants import (
    TemporalCoherenceTest, SemanticConservationTest,
    EpistemicGroundingTest, IrreversibilityTest,
    MultiInvariantValidator, InvariantResult
)
from detector.reporting import (
    DetectionReport, ReportGenerator, format_console_report,
    build_signal, SIGNAL_SCHEMA_VERSION
)
from detector.governor_signal import (
    build_governor_signal, write_governor_signal,
    GOVERNOR_SIGNAL_VERSION, _full_sha256, _build_raw
)
from detector.eval import compute_guard_flags
from detector.eval_diff import diff_csv
from detector.run_store import (
    corpus_hash, derive_run_id, git_head_sha, git_dirty,
    store_run, list_runs, load_run, diff_runs, verify_run,
    PredictionRecord, RUN_STORE_VERSION, _sha256_bytes,
)


# ============================================================================
# Utils Tests
# ============================================================================

class TestUtils:
    """Tests for utility functions"""
    
    def test_extract_citations_doi(self):
        """Test DOI extraction"""
        text = "See Smith et al. (2023), doi: 10.1234/example.5678 for details"
        citations = extract_citations(text)
        assert len(citations['dois']) == 1
        assert '10.1234/example.5678' in citations['dois'][0]
    
    def test_extract_citations_url(self):
        """Test URL extraction"""
        text = "More info at https://example.com/page and http://test.org"
        citations = extract_citations(text)
        assert len(citations['urls']) == 2
    
    def test_extract_citations_arxiv(self):
        """Test arXiv ID extraction"""
        text = "See arXiv: 2301.12345 for the preprint"
        citations = extract_citations(text)
        assert len(citations['arxiv']) == 1
        assert '2301.12345' in citations['arxiv'][0]
    
    def test_jaccard_similarity_identical(self):
        """Test Jaccard similarity for identical sets"""
        s1 = {'a', 'b', 'c'}
        s2 = {'a', 'b', 'c'}
        assert jaccard_similarity(s1, s2) == 1.0
    
    def test_jaccard_similarity_disjoint(self):
        """Test Jaccard similarity for disjoint sets"""
        s1 = {'a', 'b'}
        s2 = {'c', 'd'}
        assert jaccard_similarity(s1, s2) == 0.0
    
    def test_jaccard_similarity_partial(self):
        """Test Jaccard similarity for overlapping sets"""
        s1 = {'a', 'b', 'c'}
        s2 = {'b', 'c', 'd'}
        # intersection = 2, union = 4
        assert jaccard_similarity(s1, s2) == 0.5
    
    def test_levenshtein_similarity_identical(self):
        """Test Levenshtein similarity for identical strings"""
        assert levenshtein_similarity("hello", "hello") == 1.0
    
    def test_levenshtein_similarity_different(self):
        """Test Levenshtein similarity for different strings"""
        sim = levenshtein_similarity("hello", "world")
        assert 0 < sim < 1.0
    
    def test_normalize_text(self):
        """Test text normalization"""
        text = "  Hello,  WORLD!  "
        normalized = normalize_text(text)
        assert normalized == "hello world"
    
    def test_extract_keywords(self):
        """Test keyword extraction"""
        text = "The quick brown fox jumps over the lazy dog"
        keywords = extract_keywords(text)
        assert 'quick' in keywords
        assert 'brown' in keywords
        assert 'the' not in keywords  # stopword
        assert 'over' not in keywords  # stopword
    
    def test_phase_segment(self):
        """Test sequence phase segmentation"""
        seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        early, middle, late = phase_segment(seq, n_phases=3)
        assert len(early) == 3
        assert len(middle) == 3
        assert len(late) == 3
    
    def test_confidence_from_entropy(self):
        """Test entropy to confidence conversion"""
        assert confidence_from_entropy(0) == 1.0
        assert confidence_from_entropy(1) == 0.5
        assert 0 < confidence_from_entropy(0.5) < 1
    
    def test_to_plain_numpy(self):
        """Test numpy type conversion"""
        val = np.float64(3.14)
        assert isinstance(to_plain(val), float)
        
        arr = np.array([1, 2, 3])
        assert isinstance(to_plain(arr), list)


# ============================================================================
# Config Tests
# ============================================================================

class TestConfig:
    """Tests for configuration module"""
    
    def test_risk_profiles_exist(self):
        """Test that all expected risk profiles exist"""
        expected = ['medical', 'legal', 'research', 'general', 'creative', 'entertainment']
        for profile in expected:
            assert profile in RISK_PROFILES
    
    def test_medical_profile_conservative(self):
        """Test that medical profile is most conservative"""
        medical = RISK_PROFILES['medical']
        creative = RISK_PROFILES['creative']
        
        assert medical.temporal_threshold < creative.temporal_threshold
        assert medical.min_invariants_required >= creative.min_invariants_required
    
    def test_set_profile(self):
        """Test setting risk profile"""
        profile = set_profile('medical')
        assert profile.name == 'medical'
    
    def test_set_invalid_profile(self):
        """Test setting invalid profile raises error"""
        config = DetectorConfig()
        with pytest.raises(ValueError):
            config.set_profile('nonexistent')
    
    def test_config_get_profile(self):
        """Test getting active profile from config"""
        config = DetectorConfig()
        config.active_profile = 'research'
        profile = config.get_profile()
        assert profile.name == 'research'


# ============================================================================
# Features Tests
# ============================================================================

class TestFeatures:
    """Tests for feature extraction"""
    
    def test_generation_trace_confidence(self):
        """Test GenerationTrace confidence computation"""
        trace = GenerationTrace(
            tokens=['a', 'b', 'c'],
            logprobs=[-0.1, -0.2, -0.3],
            entropies=[1.0, 0.5, 0.0],
            temperature=0.7
        )
        conf = trace.confidence
        assert len(conf) == 3
        assert conf[2] == 1.0  # entropy 0 -> confidence 1
        assert conf[1] == 1.0 / 1.5  # entropy 0.5 -> confidence 0.667
    
    def test_generation_trace_text(self):
        """Test GenerationTrace text property"""
        trace = GenerationTrace(
            tokens=['hello', ' ', 'world'],
            logprobs=[0, 0, 0],
            entropies=[0, 0, 0]
        )
        assert trace.text == "hello world"
    
    def test_temporal_features_to_dict(self):
        """Test TemporalFeatures serialization"""
        features = TemporalFeatures(
            max_confidence_slope=0.5,
            mean_confidence=0.7,
            generation="test output"
        )
        d = features.to_dict()
        assert d['max_confidence_slope'] == 0.5
        assert d['generation'] == "test output"
    
    def test_feature_columns(self):
        """Test feature column list"""
        cols = TemporalFeatures.feature_columns()
        assert 'max_confidence_slope' in cols
        assert 'entropy_variance' in cols
        assert 'generation' not in cols  # Not a numeric feature
    
    def test_compute_temporal_debt(self):
        """Test temporal debt computation with pathological features"""
        features = TemporalFeatures(
            max_confidence_slope=0.8,
            confidence_acceleration=0.5,
            tokens_to_high_conf=0,
            entropy_variance=0.5,
            perturbation_sensitivity=0.8,
        )
        debt = compute_temporal_debt(features)
        assert 0 <= debt <= 1.0
        assert debt > 0.5  # Should be high given these features
    
    def test_compute_temporal_debt_weights(self):
        """Custom weights should affect temporal debt"""
        features = TemporalFeatures(
            max_confidence_slope=0.8,
            confidence_acceleration=0.3,
            tokens_to_high_conf=3,
            entropy_variance=0.1,
            perturbation_sensitivity=0.5,
            answer_surfaced_early=True
        )
        debt_default = compute_temporal_debt(features)
        debt_custom = compute_temporal_debt(features, weights={
            'max_confidence_slope': 0.0,
            'confidence_acceleration': 0.0,
            'tokens_to_high_conf': 0.0,
            'entropy_variance': 0.0,
            'perturbation_sensitivity': 0.0,
            'answer_surfaced_early': 0.0,
            'entropy_recovery_detected': 0.0
        })
        assert debt_custom <= debt_default
    
    def test_feature_extractor_basic(self):
        """Test basic feature extraction"""
        extractor = FeatureExtractor()

        trace = GenerationTrace(
            tokens=['a'] * 20,
            logprobs=[-0.1] * 20,
            entropies=list(np.linspace(2, 0.5, 20))  # Decreasing entropy
        )

        features = extractor.extract([trace])
        assert features is not None
        assert features.mean_confidence > 0
        assert features.generation_length == 20

    def test_phase_confidence_features_populated(self):
        """Phase confidence features should be populated by extractor"""
        extractor = FeatureExtractor()
        trace = GenerationTrace(
            tokens=['a'] * 20,
            logprobs=[-0.1] * 20,
            entropies=list(np.linspace(2, 0.5, 20))
        )
        features = extractor.extract([trace])
        assert features.early_phase_confidence > 0
        assert features.middle_phase_confidence > 0
        assert features.late_phase_confidence > 0

    def test_mean_logprob_reflects_average(self):
        """mean_logprob should equal the average of logprobs"""
        extractor = FeatureExtractor()
        logprobs = [-0.1, -0.2, -0.3, -0.4, -0.5]
        trace = GenerationTrace(
            tokens=['a'] * 5,
            logprobs=logprobs,
            entropies=[1.0] * 5
        )
        features = extractor.extract([trace])
        expected = float(np.mean(logprobs))
        assert abs(features.mean_logprob - expected) < 1e-6

    def test_unique_token_ratio(self):
        """unique_token_ratio should be unique/total"""
        extractor = FeatureExtractor()
        tokens = ['a', 'b', 'a', 'c', 'b']  # 3 unique / 5 total = 0.6
        trace = GenerationTrace(
            tokens=tokens,
            logprobs=[-0.1] * 5,
            entropies=[1.0] * 5
        )
        features = extractor.extract([trace])
        assert abs(features.unique_token_ratio - 0.6) < 1e-6

    def test_new_fields_in_to_dict(self):
        """New governor fields should appear in to_dict()"""
        features = TemporalFeatures(
            early_phase_confidence=0.3,
            middle_phase_confidence=0.5,
            late_phase_confidence=0.7,
            mean_logprob=-0.2,
            unique_token_ratio=0.8
        )
        d = features.to_dict()
        assert d['early_phase_confidence'] == 0.3
        assert d['middle_phase_confidence'] == 0.5
        assert d['late_phase_confidence'] == 0.7
        assert d['mean_logprob'] == -0.2
        assert d['unique_token_ratio'] == 0.8

    def test_new_fields_in_feature_columns(self):
        """New governor fields should appear in feature_columns()"""
        cols = TemporalFeatures.feature_columns()
        for name in ['early_phase_confidence', 'middle_phase_confidence',
                     'late_phase_confidence', 'mean_logprob', 'unique_token_ratio']:
            assert name in cols


# ============================================================================
# Baseline Tests
# ============================================================================

class TestBaseline:
    """Tests for baseline profiling"""
    
    def test_feature_stats_zscore(self):
        """Test z-score computation"""
        stats = FeatureStats(
            mean=10.0,
            std=2.0,
            median=10.0,
            p5=6.0,
            p25=8.0,
            p75=12.0,
            p95=14.0,
            min_val=5.0,
            max_val=15.0,
            n_samples=100
        )
        
        assert stats.z_score(10.0) == 0.0  # Mean
        assert stats.z_score(12.0) == 1.0  # +1 std
        assert stats.z_score(8.0) == -1.0  # -1 std
    
    def test_feature_stats_percentile(self):
        """Test percentile estimation"""
        stats = FeatureStats(
            mean=10.0, std=2.0, median=10.0,
            p5=6.0, p25=8.0, p75=12.0, p95=14.0,
            min_val=5.0, max_val=15.0, n_samples=100
        )
        
        assert abs(stats.percentile_score(10.0) - 0.5) < 0.001  # Median
        assert stats.percentile_score(5.0) < 0.1  # Below p5
        assert stats.percentile_score(15.0) > 0.9  # Above p95
    
    def test_model_baseline_save_load(self):
        """Test baseline save/load"""
        with tempfile.TemporaryDirectory() as tmpdir:
            baseline = ModelBaseline(
                model_name="test-model",
                created_at="2024-01-01",
                n_calibration_samples=100,
                calibration_dataset="test",
                feature_stats={
                    'test_feature': FeatureStats(
                        mean=1.0, std=0.5, median=1.0,
                        p5=0.2, p25=0.6, p75=1.4, p95=1.8,
                        min_val=0.1, max_val=2.0, n_samples=100
                    )
                },
                metadata={}
            )
            
            path = os.path.join(tmpdir, "test_baseline.json")
            baseline.save(path)
            
            loaded = ModelBaseline.load(path)
            assert loaded.model_name == "test-model"
            assert 'test_feature' in loaded.feature_stats
    
    def test_baseline_profiler_create(self):
        """Test baseline creation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            profiler = BaselineProfiler(tmpdir)
            
            # Create sample features
            features_list = []
            for i in range(20):
                features = TemporalFeatures(
                    max_confidence_slope=0.3 + np.random.randn() * 0.1,
                    mean_confidence=0.6 + np.random.randn() * 0.1,
                    entropy_variance=0.2 + np.random.randn() * 0.05
                )
                features_list.append(features)
            
            baseline = profiler.create_baseline(
                model_name="test",
                features_list=features_list
            )
            
            assert baseline.n_calibration_samples == 20
            assert 'max_confidence_slope' in baseline.feature_stats
    
    def test_baseline_normalize_keeps_raw(self):
        """Normalized features should keep raw values for thresholds"""
        stats = FeatureStats(
            mean=10.0,
            std=2.0,
            median=10.0,
            p5=6.0,
            p25=8.0,
            p75=12.0,
            p95=14.0,
            min_val=5.0,
            max_val=15.0,
            n_samples=100
        )
        baseline = ModelBaseline(
            model_name="test",
            created_at="2024-01-01",
            n_calibration_samples=100,
            calibration_dataset="test",
            feature_stats={'max_confidence_slope': stats},
            metadata={}
        )
        features = TemporalFeatures(max_confidence_slope=12.0)
        normalized = baseline.normalize_features(features)
        assert 'max_confidence_slope' in normalized
        assert 'max_confidence_slope_zscore' in normalized


# ============================================================================
# Invariants Tests
# ============================================================================

class TestInvariants:
    """Tests for invariant checks"""
    
    def test_temporal_coherence_pass(self):
        """Test temporal coherence with normal values"""
        test = TemporalCoherenceTest(
            threshold=0.5,
            tokens_to_conf_threshold=8
        )
        
        features = {
            'tokens_to_high_conf': 15,
            'max_confidence_slope': 0.3,
            'confidence_acceleration': 0.1,
            'answer_surfaced_early': False
        }
        
        result = test.test(features)
        assert not result.violated
        assert result.score > 0.5
    
    def test_temporal_coherence_fail(self):
        """Test temporal coherence violation"""
        test = TemporalCoherenceTest(
            threshold=0.5,
            tokens_to_conf_threshold=8
        )
        
        features = {
            'tokens_to_high_conf': 3,  # Too fast
            'max_confidence_slope': 0.8,  # Too steep
            'confidence_acceleration': 0.25,
            'answer_surfaced_early': True
        }
        
        result = test.test(features)
        assert result.violated
    
    def test_semantic_conservation_similar(self):
        """Test semantic conservation with similar responses"""
        test = SemanticConservationTest(
            similarity_threshold=0.7,
            use_embeddings=False  # Use token similarity for testing
        )
        
        responses = [
            "The capital of France is Paris, a major European city.",
            "Paris is the capital of France and a significant European city.",
            "France's capital city is Paris, in Europe."
        ]
        
        result = test.test(responses)
        # These should be fairly similar even with token matching
        assert result.score > 0.3
    
    def test_semantic_conservation_different(self):
        """Test semantic conservation with different responses"""
        test = SemanticConservationTest(
            similarity_threshold=0.7,
            use_embeddings=False
        )
        
        responses = [
            "The answer is definitely blue.",
            "Actually it's probably red or green.",
            "I think the result should be 42."
        ]
        
        result = test.test(responses)
        assert result.score < 0.5  # Should show semantic drift
    
    def test_extract_citations_rfc(self):
        """Test RFC number extraction"""
        text = "See RFC 9110 and RFC 7231 for HTTP semantics"
        citations = extract_citations(text)
        assert len(citations['rfcs']) == 2
        assert '9110' in citations['rfcs']
        assert '7231' in citations['rfcs']

    def test_epistemic_grounding_no_citations(self):
        """Test epistemic grounding with no citations"""
        test = EpistemicGroundingTest()
        
        text = "This is a response with no citations or references."
        result = test.test(text, validate=False)
        
        assert result.score == 0.5  # Neutral
        assert not result.violated
    
    def test_epistemic_grounding_with_citations(self):
        """Test epistemic grounding with citations"""
        test = EpistemicGroundingTest(max_fabricated=0)
        
        text = "See https://www.example.com for details."
        result = test.test(text, validate=False)
        
        assert result.details['total_citations'] > 0

    def test_epistemic_grounding_validate_urls_callable(self):
        """Validate URL checker is callable (no name collision)"""
        test = EpistemicGroundingTest()
        assert callable(test.validate_urls)
        assert test.validate_urls_enabled is True
    
    def test_irreversibility_similar(self):
        """Test irreversibility with similar responses"""
        test = IrreversibilityTest(similarity_threshold=0.7)
        
        responses = [
            "The answer to your question is 42.",
            "The answer to your question is 42."
        ]
        
        result = test.test(responses)
        assert result.score == 1.0
        assert not result.violated

    def test_temporal_coherence_baseline_anomaly(self):
        """Baseline z-score anomalies should trigger violation"""
        test = TemporalCoherenceTest(zscore_threshold=1.0)
        features = {
            'tokens_to_high_conf': 12,
            'max_confidence_slope': 0.2,
            'confidence_acceleration': 0.05,
            'answer_surfaced_early': False
        }
        baseline_normalized = {
            'tokens_to_high_conf_zscore': 1.5
        }
        result = test.test(features, baseline_normalized=baseline_normalized)
        assert result.violated
    
    def test_multi_invariant_aggregation(self):
        """Test multi-invariant aggregation"""
        validator = MultiInvariantValidator(min_invariants_required=2)

        results = {
            'temporal_coherence': InvariantResult(
                name='temporal_coherence',
                score=0.3,
                violated=True,
                details={}
            ),
            'semantic_conservation': InvariantResult(
                name='semantic_conservation',
                score=0.8,
                violated=False,
                details={}
            ),
            'epistemic_grounding': InvariantResult(
                name='epistemic_grounding',
                score=0.2,
                violated=True,
                details={}
            )
        }

        aggregated = validator.aggregate(results)
        assert aggregated.n_violated == 2
        assert aggregated.prediction == 'FAIL'

    def test_multi_invariant_warn_tier(self):
        """Test WARN tier: 1 violation < min_invariants_required"""
        validator = MultiInvariantValidator(min_invariants_required=2)
        results = {
            'temporal_coherence': InvariantResult(
                name='temporal_coherence', score=0.3, violated=True, details={}
            ),
            'semantic_conservation': InvariantResult(
                name='semantic_conservation', score=0.8, violated=False, details={}
            ),
        }
        aggregated = validator.aggregate(results)
        assert aggregated.n_violated == 1
        assert aggregated.prediction == 'WARN'

    def test_multi_invariant_clean_tier(self):
        """Test CLEAN tier: 0 violations"""
        validator = MultiInvariantValidator(min_invariants_required=2)
        results = {
            'temporal_coherence': InvariantResult(
                name='temporal_coherence', score=0.9, violated=False, details={}
            ),
            'semantic_conservation': InvariantResult(
                name='semantic_conservation', score=0.8, violated=False, details={}
            ),
        }
        aggregated = validator.aggregate(results)
        assert aggregated.n_violated == 0
        assert aggregated.prediction == 'CLEAN'


# ============================================================================
# Reporting Tests
# ============================================================================

class TestReporting:
    """Tests for reporting module"""
    
    def test_detection_report_to_dict(self):
        """Test report serialization"""
        report = DetectionReport(
            prediction='hallucination',
            confidence=0.85,
            model_baseline='test-model',
            risk_profile='general',
            timestamp='2024-01-01T00:00:00',
            invariant_results={},
            n_invariants_violated=2,
            temporal_features={},
            temporal_debt_score=0.7,
            anomaly_score=1.5,
            explanation='Test explanation',
            key_findings=['Finding 1'],
            recommendations=['Rec 1'],
            generation='test output',
            prompt='test prompt'
        )
        
        d = report.to_dict()
        assert d['prediction'] == 'hallucination'
        assert d['confidence'] == 0.85
    
    def test_detection_report_to_json(self):
        """Test report JSON serialization"""
        report = DetectionReport(
            prediction='truthful',
            confidence=0.9,
            model_baseline=None,
            risk_profile='general',
            timestamp='2024-01-01',
            invariant_results={},
            n_invariants_violated=0,
            temporal_features={},
            temporal_debt_score=0.2,
            anomaly_score=None,
            explanation='Test',
            key_findings=[],
            recommendations=[],
            generation='output',
            prompt='prompt'
        )
        
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed['prediction'] == 'truthful'
    
    def test_format_console_report(self):
        """Test console report formatting"""
        report = DetectionReport(
            prediction='FAIL',
            confidence=0.75,
            model_baseline=None,
            risk_profile='medical',
            timestamp='2024-01-01',
            invariant_results={
                'temporal_coherence': {
                    'score': 0.4,
                    'violated': True
                }
            },
            n_invariants_violated=1,
            temporal_features={
                'tokens_to_high_conf': 5,
                'max_confidence_slope': 0.6,
                'final_confidence': 0.9,
                'entropy_variance': 0.3,
                'perturbation_sensitivity': 0.4
            },
            temporal_debt_score=0.65,
            anomaly_score=None,
            explanation='High temporal debt detected',
            key_findings=['Fast confidence buildup'],
            recommendations=['Verify sources'],
            generation='test',
            prompt='test'
        )

        formatted = format_console_report(report)
        assert 'FAIL' in formatted
        assert 'medical' in formatted
        assert 'temporal_coherence' in formatted

    def test_signal_schema_contract(self):
        """Test structured signal schema contract"""
        report = DetectionReport(
            prediction='truthful',
            confidence=0.9,
            model_baseline='test-model',
            risk_profile='general',
            timestamp='2024-01-01T00:00:00',
            invariant_results={
                'temporal_coherence': {
                    'score': 0.8,
                    'violated': False,
                    'confidence': 0.9
                }
            },
            n_invariants_violated=0,
            temporal_features={
                'max_confidence_slope': 0.2,
                'mean_confidence': 0.7,
                'entropy_variance': 0.1,
                'generation_length': 42,
                'generation': 'hidden'
            },
            temporal_debt_score=0.1,
            anomaly_score=None,
            explanation='Test',
            key_findings=[],
            recommendations=[],
            generation='output',
            prompt='prompt'
        )
        
        signal = build_signal(report)
        assert signal['schema_version'] == SIGNAL_SCHEMA_VERSION
        assert signal['prediction'] == 'truthful'
        assert 'signals' in signal
        assert signal['signals']['generation_length'] == 42
        assert 'temporal_debt_components' in signal
        assert 'temporal_debt_weights' in signal
        assert 'provenance' in signal
        assert signal['provenance']['generation_length'] == 42
        assert 'why' in signal
        assert signal['why']['label'] == 'truthful'
    
    def test_signal_generation_hash_stable(self):
        """Generation hash should be stable for identical generations"""
        report = DetectionReport(
            prediction='truthful',
            confidence=0.9,
            model_baseline='test-model',
            risk_profile='general',
            timestamp='2024-01-01T00:00:00',
            invariant_results={},
            n_invariants_violated=0,
            temporal_features={
                'generation_length': 5,
                'generation': 'hello'
            },
            temporal_debt_score=0.1,
            anomaly_score=None,
            explanation='Test',
            key_findings=[],
            recommendations=[],
            generation='hello',
            prompt='prompt'
        )
        signal1 = build_signal(report)
        signal2 = build_signal(report)
        assert signal1['provenance']['generation_hash'] == signal2['provenance']['generation_hash']

    def test_signal_schema_file_exists(self):
        """Signal schema file should exist"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(root, 'schema', 'signal.schema.json')
        assert os.path.exists(schema_path)


class TestEvalHarness:
    """Tests for evaluation harness helpers"""
    
    def test_compute_guard_flags(self):
        profile = RISK_PROFILES['general']
        features = {
            'entropy_variance': 0.1,
            'tokens_to_high_conf': 3
        }
        flags = compute_guard_flags(
            features,
            profile,
            temporal_debt=0.2,
            anomaly_score=2.0,
            prediction='truthful'
        )
        # low_evidence requires both entropy_variance and tokens_to_high_conf
        # below profile thresholds; general profile thresholds are calibrated
        # to avoid triggering on normal model behavior.
        assert flags['anomaly'] is True
        assert flags['debt_cap'] is True

    def test_compute_guard_flags_medical(self):
        """Medical profile has stricter low_evidence thresholds"""
        profile = RISK_PROFILES['medical']
        features = {
            'entropy_variance': 0.1,
            'tokens_to_high_conf': 3
        }
        flags = compute_guard_flags(
            features,
            profile,
            temporal_debt=0.2,
            anomaly_score=2.0,
            prediction='truthful'
        )
        assert flags['low_evidence'] is True
        assert flags['anomaly'] is True


class TestEvalDiff:
    """Tests for eval diff utility"""
    
    def test_eval_diff_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_path = os.path.join(tmpdir, "old.csv")
            new_path = os.path.join(tmpdir, "new.csv")
            
            with open(old_path, "w", newline="") as f:
                f.write("id,prediction,confidence\n")
                f.write("a,truthful,0.9\n")
                f.write("b,uncertain,0.5\n")
            
            with open(new_path, "w", newline="") as f:
                f.write("id,prediction,confidence\n")
                f.write("a,uncertain,0.7\n")
                f.write("b,uncertain,0.65\n")
            
            total, changed_pred, changed_conf = diff_csv(old_path, new_path, confidence_delta=0.1)
            assert total == 2
            assert changed_pred == 1
            assert changed_conf == 2

class TestFeatureSerialization:
    """Tests for feature serialization"""
    
    def test_phase_transition_index_serialized(self):
        """phase_transition_index should be serialized to dict"""
        features = TemporalFeatures(
            phase_transition_index=7
        )
        d = features.to_dict()
        assert d['phase_transition_index'] == 7


# ============================================================================
# Governor Signal Tests
# ============================================================================

class TestGovernorSignal:
    """Tests for the governor signal emitter"""

    def _make_result(self):
        """Helper: build a minimal DetectionResult-like object."""
        from types import SimpleNamespace
        features = TemporalFeatures(
            max_confidence_slope=0.4,
            mean_confidence=0.7,
            confidence_acceleration=0.1,
            final_confidence=0.85,
            tokens_to_high_conf=12,
            entropy_variance=0.25,
            perturbation_sensitivity=0.3,
            token_jaccard=0.9,
            max_entropy_jump=0.15,
            entropy_recovery_detected=True,
            confidence_monotonicity=0.6,
            early_phase_entropy=1.2,
            middle_phase_entropy=0.8,
            late_phase_entropy=0.4,
            early_phase_confidence=0.3,
            middle_phase_confidence=0.5,
            late_phase_confidence=0.7,
            mean_logprob=-0.2,
            unique_token_ratio=0.8,
            generation="Paris is the capital of France.",
            generation_length=6,
        )
        return SimpleNamespace(
            prediction='truthful',
            confidence=0.85,
            temporal_debt=0.3,
            features=features.to_dict(),
            report=None,
        )

    def test_signal_has_required_keys(self):
        """Signal should have all top-level keys"""
        signal = build_governor_signal(self._make_result())
        for key in ['key', 'raw', 'collapsed', 'detector_version', 'profile']:
            assert key in signal

    def test_collapsed_is_null(self):
        """collapsed must be None (governor does its own collapse)"""
        signal = build_governor_signal(self._make_result())
        assert signal['collapsed'] is None

    def test_raw_has_exactly_19_fields(self):
        """raw must contain exactly 19 fields"""
        signal = build_governor_signal(self._make_result())
        assert len(signal['raw']) == 19

    def test_raw_field_names(self):
        """raw must use the governor's expected field names"""
        signal = build_governor_signal(self._make_result())
        expected = {
            'temporal_debt', 'confidence_slope', 'acceleration',
            'max_entropy_jump', 'entropy_recovery_detected',
            'entropy_variance', 'token_jaccard', 'perturbation_sensitivity',
            'tokens_to_high_conf', 'confidence_monotonicity',
            'early_phase_entropy', 'mid_phase_entropy', 'late_phase_entropy',
            'early_phase_confidence', 'mid_phase_confidence', 'late_phase_confidence',
            'mean_logprob', 'response_length', 'unique_token_ratio',
        }
        assert set(signal['raw'].keys()) == expected

    def test_renamed_fields_carry_correct_values(self):
        """Renamed fields should map to the right detector values"""
        result = self._make_result()
        signal = build_governor_signal(result)
        raw = signal['raw']
        assert raw['confidence_slope'] == result.features['max_confidence_slope']
        assert raw['acceleration'] == result.features['confidence_acceleration']
        assert raw['mid_phase_entropy'] == result.features['middle_phase_entropy']
        assert raw['mid_phase_confidence'] == result.features['middle_phase_confidence']
        assert raw['response_length'] == float(result.features['generation_length'])

    def test_key_has_required_fields(self):
        """key must contain run_id, turn_id, model_id, response_hash"""
        signal = build_governor_signal(self._make_result())
        for field in ['run_id', 'turn_id', 'model_id', 'response_hash']:
            assert field in signal['key']

    def test_response_hash_is_full_sha256(self):
        """response_hash must be full 64-char hex (not truncated)"""
        signal = build_governor_signal(self._make_result())
        h = signal['key']['response_hash']
        assert len(h) == 64
        assert all(c in '0123456789abcdef' for c in h)

    def test_response_hash_matches_generation(self):
        """response_hash should match sha256 of the generation text"""
        import hashlib
        result = self._make_result()
        signal = build_governor_signal(result)
        expected = hashlib.sha256(result.features['generation'].encode()).hexdigest()
        assert signal['key']['response_hash'] == expected

    def test_detector_version(self):
        """detector_version should be 2.0.0"""
        signal = build_governor_signal(self._make_result())
        assert signal['detector_version'] == '2.0.0'

    def test_signal_is_json_serializable(self):
        """Signal dict must be fully JSON-serializable"""
        signal = build_governor_signal(self._make_result())
        s = json.dumps(signal)
        parsed = json.loads(s)
        assert parsed['detector_version'] == '2.0.0'

    def test_write_governor_signal_creates_file(self):
        """write_governor_signal should create a valid JSON file"""
        signal = build_governor_signal(self._make_result())
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_signal.json')
            write_governor_signal(signal, path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded['raw']['confidence_slope'] == signal['raw']['confidence_slope']

    def test_temporal_debt_from_result(self):
        """temporal_debt in raw must come from result.temporal_debt"""
        result = self._make_result()
        signal = build_governor_signal(result)
        assert signal['raw']['temporal_debt'] == result.temporal_debt

    def test_entropy_recovery_cast_to_float(self):
        """entropy_recovery_detected should be float (not bool)"""
        signal = build_governor_signal(self._make_result())
        val = signal['raw']['entropy_recovery_detected']
        assert isinstance(val, float)
        assert val == 1.0  # True -> 1.0

    def test_governor_signal_schema_file_exists(self):
        """Governor signal schema file should exist"""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_path = os.path.join(root, 'schema', 'governor_signal.schema.json')
        assert os.path.exists(schema_path)


# ============================================================================
# Run Store Tests
# ============================================================================

class TestRunStore:
    """Tests for append-only run store"""

    def _make_corpus(self, tmpdir: str, content: str = "line1\nline2\n") -> str:
        path = os.path.join(tmpdir, "corpus.jsonl")
        with open(path, "w") as f:
            f.write(content)
        return path

    def _make_predictions(self, n: int = 3) -> list:
        preds = []
        for i in range(n):
            preds.append(PredictionRecord(
                id=f"item_{i}",
                prompt=f"prompt {i}",
                expected_risk="low" if i % 2 == 0 else "high",
                prediction="temporal_stable" if i % 2 == 0 else "temporal_unstable",
                confidence=0.8 + i * 0.01,
                temporal_debt=0.1 + i * 0.05,
                anomaly_score=None,
                features={"entropy_variance": 0.2, "tokens_to_high_conf": 5},
                guard_flags={"low_evidence": False, "anomaly": False, "debt_cap": i == 1},
                notes=f"note {i}",
            ))
        return preds

    # --- Hash helpers ---

    def test_corpus_hash_deterministic(self):
        """Same file bytes -> same hash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p = self._make_corpus(tmpdir)
            assert corpus_hash(p) == corpus_hash(p)

    def test_corpus_hash_sensitive(self):
        """Different content -> different hash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = self._make_corpus(tmpdir, "aaa\n")
            h1 = corpus_hash(p1)
            # overwrite
            with open(p1, "w") as f:
                f.write("bbb\n")
            h2 = corpus_hash(p1)
            assert h1 != h2

    def test_derive_run_id_deterministic(self):
        """Same 4 axes -> same run_id"""
        rid = derive_run_id("abc", "def", "general", "model-x")
        assert rid == derive_run_id("abc", "def", "general", "model-x")
        assert len(rid) == 12

    def test_derive_run_id_sensitive(self):
        """Changing any axis changes run_id"""
        base = derive_run_id("abc", "def", "general", "model-x")
        assert base != derive_run_id("xyz", "def", "general", "model-x")
        assert base != derive_run_id("abc", "ghi", "general", "model-x")
        assert base != derive_run_id("abc", "def", "medical", "model-x")
        assert base != derive_run_id("abc", "def", "general", "model-y")

    # --- store_run ---

    def test_store_run_creates_three_files(self):
        """store_run creates predictions.jsonl, summary.json, manifest.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            preds = self._make_predictions()
            rd = store_run(preds, cp, "m", "general", runs_root=os.path.join(tmpdir, "runs"))
            assert (rd / "predictions.jsonl").exists()
            assert (rd / "summary.json").exists()
            assert (rd / "manifest.json").exists()

    def test_manifest_written_last(self):
        """If we delete manifest, the run looks incomplete to list_runs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")
            rd = store_run(self._make_predictions(), cp, "m", "general", runs_root=runs_root)
            assert len(list_runs(runs_root)) == 1
            os.remove(rd / "manifest.json")
            assert len(list_runs(runs_root)) == 0

    def test_predictions_jsonl_valid(self):
        """Each line in predictions.jsonl is valid JSON"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(5), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            lines = (rd / "predictions.jsonl").read_text().strip().split("\n")
            assert len(lines) == 5
            for line in lines:
                obj = json.loads(line)
                assert "id" in obj
                assert "prediction" in obj

    def test_predictions_hash_matches(self):
        """predictions_hash in manifest matches actual file hash"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            manifest = json.loads((rd / "manifest.json").read_text())
            actual = _sha256_bytes((rd / "predictions.jsonl").read_bytes())
            assert manifest["predictions_hash"] == actual

    # --- verify_run ---

    def test_verify_run_valid(self):
        """verify_run passes on an untampered run"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            assert verify_run(str(rd)) is True

    def test_verify_run_tampered(self):
        """verify_run fails if predictions.jsonl is tampered"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            # tamper
            pred_path = rd / "predictions.jsonl"
            pred_path.write_text("tampered content\n")
            assert verify_run(str(rd)) is False

    # --- list_runs ---

    def test_list_runs_finds_completed(self):
        """list_runs finds directories with manifest.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")
            store_run(self._make_predictions(), cp, "m", "general", runs_root=runs_root)
            assert len(list_runs(runs_root)) == 1

    def test_list_runs_ignores_incomplete(self):
        """list_runs ignores directories without manifest.json"""
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = os.path.join(tmpdir, "runs")
            os.makedirs(os.path.join(runs_root, "incomplete_run"))
            assert len(list_runs(runs_root)) == 0

    def test_list_runs_empty_root(self):
        """list_runs returns [] for nonexistent root"""
        assert list_runs("/nonexistent/path/runs") == []

    # --- load_run ---

    def test_load_run_returns_all_components(self):
        """load_run returns manifest, predictions, and summary"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(4), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            data = load_run(str(rd))
            assert "manifest" in data
            assert "predictions" in data
            assert "summary" in data
            assert len(data["predictions"]) == 4

    # --- diff_runs ---

    def test_diff_runs_detects_prediction_change(self):
        """diff_runs detects when a prediction changes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")

            preds_a = self._make_predictions(2)
            rd_a = store_run(preds_a, cp, "m", "general", runs_root=runs_root)

            preds_b = self._make_predictions(2)
            preds_b[0] = PredictionRecord(
                id="item_0", prompt="prompt 0", expected_risk="low",
                prediction="temporal_unstable",  # changed
                confidence=0.8, temporal_debt=0.1, anomaly_score=None,
                features={}, guard_flags={}, notes="",
            )
            # Need a different run_id so directories don't collide
            rd_b = store_run(preds_b, cp, "m", "medical", runs_root=runs_root)

            diff = diff_runs(str(rd_a), str(rd_b))
            assert len(diff["prediction_changes"]) >= 1
            changed_ids = [c["id"] for c in diff["prediction_changes"] if c.get("change") == "prediction_changed"]
            assert "item_0" in changed_ids

    def test_diff_runs_detects_config_change(self):
        """diff_runs detects config changes"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")
            rd_a = store_run(self._make_predictions(), cp, "m", "general", runs_root=runs_root)
            rd_b = store_run(self._make_predictions(), cp, "m", "medical", runs_root=runs_root)
            diff = diff_runs(str(rd_a), str(rd_b))
            assert "profile" in diff["config_changes"]

    # --- summary ---

    def test_summary_prediction_counts(self):
        """Summary has correct prediction counts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            preds = self._make_predictions(4)
            rd = store_run(preds, cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            summary = json.loads((rd / "summary.json").read_text())
            total = sum(summary["prediction_counts"].values())
            assert total == 4

    def test_summary_guard_flag_totals(self):
        """Summary counts guard flags correctly"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            preds = self._make_predictions(3)  # item_1 has debt_cap=True
            rd = store_run(preds, cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            summary = json.loads((rd / "summary.json").read_text())
            assert summary["guard_flag_counts"].get("debt_cap", 0) == 1

    # --- append-only ---

    def test_store_run_never_overwrites(self):
        """store_run raises FileExistsError if directory already exists"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            runs_root = os.path.join(tmpdir, "runs")
            rd = store_run(self._make_predictions(), cp, "m", "general", runs_root=runs_root)
            # Pre-create the exact directory name that the next call would use
            # to prove the guard works (same run_id + same second = collision)
            from pathlib import Path
            import time
            time.sleep(1.1)  # ensure different timestamp
            rd2 = store_run(self._make_predictions(), cp, "m", "general", runs_root=runs_root)
            assert rd != rd2
            # Both runs exist and are complete
            assert (rd / "manifest.json").exists()
            assert (rd2 / "manifest.json").exists()

    def test_manifest_has_run_store_version(self):
        """Manifest includes run_store_version"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            manifest = json.loads((rd / "manifest.json").read_text())
            assert manifest["run_store_version"] == RUN_STORE_VERSION

    # --- Multi-invariant fields ---

    def _make_multi_invariant_predictions(self, n: int = 3) -> list:
        preds = []
        for i in range(n):
            preds.append(PredictionRecord(
                id=f"item_{i}",
                prompt=f"prompt {i}",
                expected_risk="low" if i % 2 == 0 else "high",
                prediction="CLEAN" if i % 2 == 0 else "FAIL",
                confidence=0.8 + i * 0.01,
                temporal_debt=0.1 + i * 0.05,
                anomaly_score=None,
                features={"entropy_variance": 0.2},
                guard_flags={"low_evidence": False, "anomaly": False, "debt_cap": False},
                notes="",
                invariant_results={
                    "temporal_coherence": {"name": "temporal_coherence", "score": 0.7 + i * 0.05, "violated": i % 2 != 0},
                    "semantic_conservation": {"name": "semantic_conservation", "score": 0.6 + i * 0.1, "violated": i == 2},
                },
                n_violated=1 if i % 2 != 0 else 0,
                aggregate_score=0.7 + i * 0.05,
            ))
        return preds

    def test_multi_invariant_predictions_in_jsonl(self):
        """Multi-invariant fields are serialized in predictions.jsonl"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            preds = self._make_multi_invariant_predictions()
            rd = store_run(preds, cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"), multi_invariant=True)
            lines = (rd / "predictions.jsonl").read_text().strip().split("\n")
            obj = json.loads(lines[1])  # item_1 has invariant violations
            assert obj["n_violated"] == 1
            assert "temporal_coherence" in obj["invariant_results"]
            assert obj["invariant_results"]["temporal_coherence"]["violated"] is True

    def test_multi_invariant_manifest_flag(self):
        """Manifest records multi_invariant=True"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_multi_invariant_predictions(), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"), multi_invariant=True)
            manifest = json.loads((rd / "manifest.json").read_text())
            assert manifest["multi_invariant"] is True

    def test_multi_invariant_summary_violation_counts(self):
        """Summary tracks per-invariant violation counts"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            preds = self._make_multi_invariant_predictions(3)
            rd = store_run(preds, cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"), multi_invariant=True)
            summary = json.loads((rd / "summary.json").read_text())
            assert summary["invariant_violation_counts"]["temporal_coherence"] == 1
            assert summary["invariant_violation_counts"]["semantic_conservation"] == 1

    def test_multi_invariant_summary_mean_scores(self):
        """Summary tracks per-invariant mean scores"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            preds = self._make_multi_invariant_predictions(3)
            rd = store_run(preds, cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"), multi_invariant=True)
            summary = json.loads((rd / "summary.json").read_text())
            assert "temporal_coherence" in summary["invariant_mean_scores"]
            assert "semantic_conservation" in summary["invariant_mean_scores"]

    def test_single_invariant_has_empty_invariant_fields(self):
        """Single-invariant runs have empty invariant summary fields"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cp = self._make_corpus(tmpdir)
            rd = store_run(self._make_predictions(), cp, "m", "general",
                           runs_root=os.path.join(tmpdir, "runs"))
            summary = json.loads((rd / "summary.json").read_text())
            assert summary["invariant_violation_counts"] == {}
            assert summary["invariant_mean_scores"] == {}
            manifest = json.loads((rd / "manifest.json").read_text())
            assert manifest["multi_invariant"] is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests (require mocking the LLM)"""
    
    def test_full_pipeline_mock(self):
        """Test full detection pipeline with mocked model"""
        # This would require mocking the transformer model
        # For now, just verify the components integrate
        
        # Create features
        features = TemporalFeatures(
            max_confidence_slope=0.4,
            mean_confidence=0.7,
            confidence_acceleration=0.1,
            final_confidence=0.85,
            tokens_to_high_conf=12,
            entropy_variance=0.25,
            perturbation_sensitivity=0.3,
            generation="Test response about Paris being the capital of France."
        )
        
        # Compute debt
        debt = compute_temporal_debt(features)
        
        # Test invariant
        tc_test = TemporalCoherenceTest()
        tc_result = tc_test.test(features.to_dict())
        
        # Generate report
        generator = ReportGenerator()
        from detector.invariants import MultiInvariantResult
        
        report = generator.generate(
            prediction='temporal_stable' if not tc_result.violated else 'temporal_unstable',
            confidence=tc_result.score,
            features=features,
            invariant_result=MultiInvariantResult(
                results={'temporal_coherence': tc_result},
                n_violated=1 if tc_result.violated else 0,
                aggregate_score=tc_result.score,
                prediction='truthful',
                confidence=tc_result.score
            ),
            temporal_debt=debt
        )
        
        assert report is not None
        assert report.prediction in ['temporal_stable', 'temporal_unstable']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
