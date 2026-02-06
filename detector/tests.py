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
        """Test temporal debt computation"""
        features = TemporalFeatures(
            max_confidence_slope=0.8,
            confidence_acceleration=0.3,
            tokens_to_high_conf=3,
            entropy_variance=0.1,
            perturbation_sensitivity=0.5,
            answer_surfaced_early=True
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
        assert aggregated.prediction == 'hallucination'


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
            prediction='hallucination',
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
        assert 'HALLUCINATION' in formatted
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
            prediction='truthful' if not tc_result.violated else 'hallucination',
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
        assert report.prediction in ['truthful', 'hallucination']


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
