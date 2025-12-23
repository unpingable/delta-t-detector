"""
Δt Hallucination Detector

A framework for detecting hallucinations in LLM outputs by measuring
temporal coherence violations - when belief change rate (dC/dt) exceeds
evidence accumulation rate (dE/dt).

Core concept: Hallucinations accumulate "temporal debt" when models
reach high confidence faster than evidence gathering can justify.

Usage:
    from detector import DeltaTDetector
    
    # Basic detection
    detector = DeltaTDetector()
    result = detector.detect("What is the capital of France?")
    print(result.prediction)  # 'truthful' or 'hallucination'
    
    # With risk profile
    detector.set_risk_profile('medical')  # More conservative
    result = detector.detect("What medication treats X?")
    
    # Multi-invariant detection
    result = detector.detect_multi_invariant(
        "Explain quantum entanglement",
        validate_citations=True,
        test_semantic=True
    )
    
    # With baseline normalization
    detector.load_baseline()  # or create_baseline() first
    result = detector.detect_with_baseline("Complex question...")
"""

__version__ = "2.0.0"
__author__ = "Δt Research"

from .core import (
    DeltaTDetector,
    DetectionResult,
    quick_detect,
    batch_detect
)

from .config import (
    DetectorConfig,
    RiskProfile,
    RISK_PROFILES,
    get_config,
    set_profile
)

from .features import (
    FeatureExtractor,
    TemporalFeatures,
    GenerationTrace,
    compute_temporal_debt
)

from .baseline import (
    BaselineProfiler,
    ModelBaseline,
    FeatureStats
)

from .invariants import (
    TemporalCoherenceTest,
    SemanticConservationTest,
    EpistemicGroundingTest,
    IrreversibilityTest,
    MultiInvariantValidator,
    InvariantResult,
    MultiInvariantResult
)

from .reporting import (
    DetectionReport,
    ReportGenerator,
    format_console_report,
    format_json_report,
    format_markdown_report
)

# Optional API providers (may not be installed)
try:
    from .api_providers import (
        OpenAIProvider,
        AnthropicProvider,
        OllamaProvider,
        get_provider,
        detect_with_api,
        APIDetectionResult
    )
    _has_api_providers = True
except ImportError:
    _has_api_providers = False

__all__ = [
    # Core
    'DeltaTDetector',
    'DetectionResult',
    'quick_detect',
    'batch_detect',
    
    # Config
    'DetectorConfig',
    'RiskProfile',
    'RISK_PROFILES',
    'get_config',
    'set_profile',
    
    # Features
    'FeatureExtractor',
    'TemporalFeatures',
    'GenerationTrace',
    'compute_temporal_debt',
    
    # Baseline
    'BaselineProfiler',
    'ModelBaseline',
    'FeatureStats',
    
    # Invariants
    'TemporalCoherenceTest',
    'SemanticConservationTest',
    'EpistemicGroundingTest',
    'IrreversibilityTest',
    'MultiInvariantValidator',
    'InvariantResult',
    'MultiInvariantResult',
    
    # Reporting
    'DetectionReport',
    'ReportGenerator',
    'format_console_report',
    'format_json_report',
    'format_markdown_report',
    
    # API Providers (if available)
    'OpenAIProvider',
    'AnthropicProvider', 
    'OllamaProvider',
    'get_provider',
    'detect_with_api',
    'APIDetectionResult',
]
