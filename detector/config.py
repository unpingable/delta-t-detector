"""
Δt Hallucination Detector - Configuration Module
Risk profiles, thresholds, and detection parameters
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class RiskLevel(Enum):
    """Risk tolerance levels for different use cases"""
    CRITICAL = "critical"    # Medical, legal, financial
    HIGH = "high"            # Research, journalism
    MODERATE = "moderate"    # General Q&A
    LOW = "low"              # Creative, brainstorming
    PERMISSIVE = "permissive"  # Fiction, entertainment


@dataclass
class RiskProfile:
    """Configuration for a specific risk tolerance level"""
    name: str
    description: str
    
    # Temporal coherence thresholds
    temporal_threshold: float = 0.5
    confidence_acceleration_threshold: float = 0.15
    tokens_to_confidence_threshold: int = 8
    baseline_zscore_threshold: float = 2.0
    truthful_confidence_cap: float = 0.6
    temporal_debt_confidence_cap_threshold: float = 0.15
    low_evidence_entropy_threshold: float = 0.2
    low_evidence_tokens_threshold: int = 8
    low_evidence_confidence_cap: float = 0.5
    anomaly_score_uncertain_threshold: float = 1.0
    anomaly_confidence_cap: float = 0.5
    temporal_debt_weights: Dict[str, float] = field(default_factory=lambda: {
        'max_confidence_slope': 0.3,
        'confidence_acceleration': 0.2,
        'tokens_to_high_conf': 0.2,
        'entropy_variance': 0.1,
        'perturbation_sensitivity': 0.2,
        'answer_surfaced_early': 0.2,
        'entropy_recovery_detected': 0.1
    })
    
    # Semantic conservation
    semantic_similarity_threshold: float = 0.75
    require_semantic_stability: bool = True
    semantic_variants: int = 3
    
    # Epistemic grounding
    require_grounding: bool = True
    max_fabricated_citations: int = 0
    validate_urls: bool = True
    
    # Multi-invariant decision rules
    min_invariants_required: int = 2
    invariant_weights: Dict[str, float] = field(default_factory=lambda: {
        'temporal_coherence': 1.0,
        'semantic_conservation': 0.8,
        'epistemic_grounding': 1.0,
        'irreversibility': 0.5
    })
    
    # Feature extraction
    n_perturbations: int = 2
    max_new_tokens: int = 100
    
    # Ensemble detection
    use_ensemble: bool = False
    min_ensemble_agreement: float = 0.7


# Pre-defined risk profiles
RISK_PROFILES: Dict[str, RiskProfile] = {
    'medical': RiskProfile(
        name='medical',
        description='Medical/clinical applications - maximum caution',
        temporal_threshold=0.3,
        confidence_acceleration_threshold=0.10,
        tokens_to_confidence_threshold=12,
        truthful_confidence_cap=0.55,
        temporal_debt_confidence_cap_threshold=0.12,
        low_evidence_entropy_threshold=0.22,
        low_evidence_tokens_threshold=10,
        low_evidence_confidence_cap=0.45,
        anomaly_score_uncertain_threshold=0.9,
        anomaly_confidence_cap=0.45,
        semantic_similarity_threshold=0.85,
        require_semantic_stability=True,
        require_grounding=True,
        max_fabricated_citations=0,
        min_invariants_required=3,
        invariant_weights={
            'temporal_coherence': 1.0,
            'semantic_conservation': 1.0,
            'epistemic_grounding': 1.5,  # Extra weight for medical
            'irreversibility': 0.5
        }
    ),
    
    'legal': RiskProfile(
        name='legal',
        description='Legal/compliance applications - high caution',
        temporal_threshold=0.35,
        confidence_acceleration_threshold=0.12,
        tokens_to_confidence_threshold=10,
        truthful_confidence_cap=0.55,
        temporal_debt_confidence_cap_threshold=0.13,
        low_evidence_entropy_threshold=0.21,
        low_evidence_tokens_threshold=9,
        low_evidence_confidence_cap=0.45,
        anomaly_score_uncertain_threshold=0.95,
        anomaly_confidence_cap=0.45,
        semantic_similarity_threshold=0.80,
        require_grounding=True,
        max_fabricated_citations=0,
        min_invariants_required=3,
        invariant_weights={
            'temporal_coherence': 1.0,
            'semantic_conservation': 1.0,
            'epistemic_grounding': 1.2,
            'irreversibility': 0.5
        }
    ),
    
    'research': RiskProfile(
        name='research',
        description='Academic/research applications - balanced caution',
        temporal_threshold=0.45,
        confidence_acceleration_threshold=0.15,
        tokens_to_confidence_threshold=8,
        truthful_confidence_cap=0.6,
        temporal_debt_confidence_cap_threshold=0.15,
        low_evidence_entropy_threshold=0.2,
        low_evidence_tokens_threshold=8,
        low_evidence_confidence_cap=0.5,
        anomaly_score_uncertain_threshold=1.1,
        anomaly_confidence_cap=0.55,
        semantic_similarity_threshold=0.75,
        require_grounding=True,
        max_fabricated_citations=1,  # Allow one uncertain citation
        min_invariants_required=2,
    ),
    
    'general': RiskProfile(
        name='general',
        description='General Q&A - balanced approach',
        temporal_threshold=0.5,
        confidence_acceleration_threshold=0.18,
        tokens_to_confidence_threshold=6,
        truthful_confidence_cap=0.6,
        temporal_debt_confidence_cap_threshold=0.15,
        low_evidence_entropy_threshold=0.2,
        low_evidence_tokens_threshold=8,
        low_evidence_confidence_cap=0.5,
        anomaly_score_uncertain_threshold=1.0,
        anomaly_confidence_cap=0.5,
        semantic_similarity_threshold=0.70,
        require_grounding=True,
        max_fabricated_citations=1,
        min_invariants_required=2,
    ),
    
    'creative': RiskProfile(
        name='creative',
        description='Creative/brainstorming - permissive',
        temporal_threshold=0.7,
        confidence_acceleration_threshold=0.25,
        tokens_to_confidence_threshold=4,
        truthful_confidence_cap=0.8,
        temporal_debt_confidence_cap_threshold=0.25,
        low_evidence_entropy_threshold=0.1,
        low_evidence_tokens_threshold=4,
        low_evidence_confidence_cap=0.7,
        anomaly_score_uncertain_threshold=2.0,
        anomaly_confidence_cap=0.7,
        semantic_similarity_threshold=0.60,
        require_semantic_stability=False,
        require_grounding=False,
        max_fabricated_citations=3,
        min_invariants_required=1,
        invariant_weights={
            'temporal_coherence': 1.0,
            'semantic_conservation': 0.3,
            'epistemic_grounding': 0.2,
            'irreversibility': 0.5
        }
    ),
    
    'entertainment': RiskProfile(
        name='entertainment',
        description='Fiction/entertainment - very permissive',
        temporal_threshold=0.8,
        confidence_acceleration_threshold=0.30,
        tokens_to_confidence_threshold=3,
        truthful_confidence_cap=0.85,
        temporal_debt_confidence_cap_threshold=0.3,
        low_evidence_entropy_threshold=0.08,
        low_evidence_tokens_threshold=3,
        low_evidence_confidence_cap=0.75,
        anomaly_score_uncertain_threshold=2.5,
        anomaly_confidence_cap=0.75,
        semantic_similarity_threshold=0.50,
        require_semantic_stability=False,
        require_grounding=False,
        max_fabricated_citations=999,
        min_invariants_required=1,
    )
}


@dataclass 
class DetectorConfig:
    """Global detector configuration"""
    # Model settings
    default_model: str = "Qwen/Qwen2.5-3B-Instruct"
    device: str = "auto"  # auto, cuda, mps, cpu
    dtype: str = "auto"   # auto, float16, bfloat16, float32
    
    # Active risk profile
    active_profile: str = "general"
    
    # Baseline profiling
    baseline_calibration_size: int = 100
    baseline_path: str = "./baselines"
    
    # Feature extraction
    temperature_base: float = 0.7
    temperature_step: float = 0.1
    
    # Citation validation
    citation_timeout: int = 5  # seconds
    citation_user_agent: str = "DeltaT-Detector/1.0"
    
    # Caching
    cache_embeddings: bool = True
    embedding_cache_size: int = 1000
    
    # Reporting
    verbose: bool = True
    save_diagnostics: bool = True
    
    def get_profile(self) -> RiskProfile:
        """Get the active risk profile"""
        return RISK_PROFILES.get(self.active_profile, RISK_PROFILES['general'])
    
    def set_profile(self, profile_name: str) -> None:
        """Set the active risk profile"""
        if profile_name not in RISK_PROFILES:
            raise ValueError(f"Unknown profile: {profile_name}. Available: {list(RISK_PROFILES.keys())}")
        self.active_profile = profile_name


# Default global config
DEFAULT_CONFIG = DetectorConfig()


def get_config() -> DetectorConfig:
    """Get the current configuration"""
    return DEFAULT_CONFIG


def set_profile(profile_name: str) -> RiskProfile:
    """Convenience function to set and return risk profile"""
    DEFAULT_CONFIG.set_profile(profile_name)
    return DEFAULT_CONFIG.get_profile()
