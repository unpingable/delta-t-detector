"""
Δt Hallucination Detector - Feature Extraction Module
Enhanced feature engineering with phase-aware and trajectory analysis
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .utils import (
    smooth_series, confidence_from_entropy, phase_segment,
    detect_phase_transition, jaccard_similarity
)


@dataclass
class GenerationTrace:
    """Container for a single generation trace"""
    tokens: List[str]
    logprobs: List[float]
    entropies: List[float]
    temperature: float = 0.7
    margins: List[float] = field(default_factory=list)  # top-1 minus top-2 prob per step
    
    @property
    def confidence(self) -> List[float]:
        """Convert entropies to confidence scores"""
        return [confidence_from_entropy(e) for e in self.entropies]
    
    @property
    def text(self) -> str:
        """Join tokens into text"""
        return "".join(self.tokens)
    
    def __len__(self) -> int:
        return len(self.tokens)


@dataclass
class TemporalFeatures:
    """Container for extracted temporal features"""
    # Core confidence features
    max_confidence_slope: float = 0.0
    mean_confidence: float = 0.0
    confidence_acceleration: float = 0.0
    final_confidence: float = 0.0
    tail_confidence_mean: float = 0.0
    
    # Temporal thresholds
    tokens_to_high_conf: float = 0.0
    
    # Entropy features
    max_entropy_jump: float = 0.0
    entropy_variance: float = 0.0
    entropy_recovery_detected: bool = False
    
    # Perturbation features
    token_jaccard: float = 1.0
    perturbation_sensitivity: float = 0.0
    
    # Phase-aware features (new)
    early_phase_entropy: float = 0.0
    middle_phase_entropy: float = 0.0
    late_phase_entropy: float = 0.0
    phase_transition_index: Optional[int] = None
    answer_surfaced_early: bool = False
    
    # Trajectory features (new)
    confidence_second_derivative: float = 0.0  # d²C/dt²
    logprob_variance: float = 0.0
    entropy_trajectory_slope: float = 0.0
    confidence_monotonicity: float = 0.0

    # Governor integration features
    early_phase_confidence: float = 0.0
    middle_phase_confidence: float = 0.0
    late_phase_confidence: float = 0.0
    mean_logprob: float = 0.0
    unique_token_ratio: float = 0.0

    # Generation metadata
    generation: str = ""
    generation_length: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for ML pipeline"""
        return {
            'max_confidence_slope': self.max_confidence_slope,
            'mean_confidence': self.mean_confidence,
            'confidence_acceleration': self.confidence_acceleration,
            'final_confidence': self.final_confidence,
            'tail_confidence_mean': self.tail_confidence_mean,
            'tokens_to_high_conf': self.tokens_to_high_conf,
            'max_entropy_jump': self.max_entropy_jump,
            'entropy_variance': self.entropy_variance,
            'entropy_recovery_detected': float(self.entropy_recovery_detected),
            'token_jaccard': self.token_jaccard,
            'perturbation_sensitivity': self.perturbation_sensitivity,
            'early_phase_entropy': self.early_phase_entropy,
            'middle_phase_entropy': self.middle_phase_entropy,
            'late_phase_entropy': self.late_phase_entropy,
            'phase_transition_index': self.phase_transition_index,
            'answer_surfaced_early': float(self.answer_surfaced_early),
            'confidence_second_derivative': self.confidence_second_derivative,
            'logprob_variance': self.logprob_variance,
            'entropy_trajectory_slope': self.entropy_trajectory_slope,
            'confidence_monotonicity': self.confidence_monotonicity,
            'early_phase_confidence': self.early_phase_confidence,
            'middle_phase_confidence': self.middle_phase_confidence,
            'late_phase_confidence': self.late_phase_confidence,
            'mean_logprob': self.mean_logprob,
            'unique_token_ratio': self.unique_token_ratio,
            'generation': self.generation,
            'generation_length': self.generation_length
        }
    
    @classmethod
    def feature_columns(cls) -> List[str]:
        """Return list of numeric feature column names for ML"""
        return [
            'max_confidence_slope',
            'mean_confidence', 
            'confidence_acceleration',
            'final_confidence',
            'tail_confidence_mean',
            'tokens_to_high_conf',
            'max_entropy_jump',
            'entropy_variance',
            'entropy_recovery_detected',
            'token_jaccard',
            'perturbation_sensitivity',
            'early_phase_entropy',
            'middle_phase_entropy',
            'late_phase_entropy',
            'answer_surfaced_early',
            'confidence_second_derivative',
            'logprob_variance',
            'entropy_trajectory_slope',
            'confidence_monotonicity',
            'early_phase_confidence',
            'middle_phase_confidence',
            'late_phase_confidence',
            'mean_logprob',
            'unique_token_ratio'
        ]


class FeatureExtractor:
    """
    Extract temporal and trajectory features from generation traces
    """
    
    def __init__(
        self,
        high_conf_threshold: float = 0.8,
        smoothing_window: int = 3,
        early_answer_threshold: int = 5
    ):
        self.high_conf_threshold = high_conf_threshold
        self.smoothing_window = smoothing_window
        self.early_answer_threshold = early_answer_threshold
    
    def extract(
        self,
        traces: List[GenerationTrace],
        levenshtein_func: Optional[callable] = None,
        embed_func: Optional[callable] = None,
    ) -> Optional[TemporalFeatures]:
        """
        Extract all temporal features from generation traces

        Args:
            traces: List of GenerationTrace objects (from perturbation runs)
            levenshtein_func: Optional Levenshtein distance function (fallback)
            embed_func: Optional embedding function (str -> np.ndarray).
                        When provided, perturbation_sensitivity uses
                        1 - median_pairwise_cosine instead of Levenshtein.

        Returns:
            TemporalFeatures or None if extraction fails
        """
        if not traces or len(traces[0]) < 4:
            return None

        primary = traces[0]
        features = TemporalFeatures()

        # Core confidence features
        self._extract_confidence_features(primary, features)

        # Phase-aware features
        self._extract_phase_features(primary, features)

        # Trajectory features
        self._extract_trajectory_features(primary, features)

        # Perturbation features (requires multiple traces)
        if len(traces) > 1:
            self._extract_perturbation_features(
                traces, features, levenshtein_func, embed_func
            )

        # Generation metadata
        features.generation = primary.text
        features.generation_length = len(primary)

        # Unique token ratio
        if len(primary.tokens) > 0:
            features.unique_token_ratio = len(set(primary.tokens)) / len(primary.tokens)

        return features
    
    def _extract_confidence_features(
        self,
        trace: GenerationTrace,
        features: TemporalFeatures
    ) -> None:
        """Extract core confidence-based features"""
        confidence = trace.confidence
        entropies = trace.entropies
        
        # Smoothed confidence
        if len(confidence) >= self.smoothing_window:
            conf_smooth = smooth_series(confidence, self.smoothing_window)
        else:
            conf_smooth = np.array(confidence)
        
        # Confidence slopes (dC/dt)
        if len(conf_smooth) >= 2:
            slopes = np.diff(conf_smooth)
            features.max_confidence_slope = float(np.max(slopes))
            
            # Confidence acceleration (d²C/dt²)
            if len(slopes) > 1:
                accel = np.diff(slopes)
                features.confidence_acceleration = float(np.max(accel))
        
        features.mean_confidence = float(np.mean(confidence))
        features.final_confidence = float(confidence[-1])
        
        # Tail confidence
        tail_len = min(10, len(confidence))
        features.tail_confidence_mean = float(np.mean(confidence[-tail_len:]))
        
        # Tokens to high confidence (phase-reversal signature)
        above_thresh = np.where(np.array(conf_smooth) > self.high_conf_threshold)[0]
        if len(above_thresh) > 0:
            features.tokens_to_high_conf = float(above_thresh[0])
        else:
            features.tokens_to_high_conf = float(len(conf_smooth))
        
        # Entropy features
        entropy_diffs = np.diff(entropies)
        if len(entropy_diffs) > 0:
            features.max_entropy_jump = float(np.max(entropy_diffs))
        features.entropy_variance = float(np.var(entropies))
    
    def _extract_phase_features(
        self,
        trace: GenerationTrace,
        features: TemporalFeatures
    ) -> None:
        """Extract phase-aware features (context/reasoning/answer)"""
        entropies = trace.entropies
        confidence = trace.confidence
        
        # Segment into phases
        early, middle, late = phase_segment(entropies)

        if early:
            features.early_phase_entropy = float(np.mean(early))
        if middle:
            features.middle_phase_entropy = float(np.mean(middle))
        if late:
            features.late_phase_entropy = float(np.mean(late))

        # Phase confidence means
        early_c, middle_c, late_c = phase_segment(confidence)
        if early_c:
            features.early_phase_confidence = float(np.mean(early_c))
        if middle_c:
            features.middle_phase_confidence = float(np.mean(middle_c))
        if late_c:
            features.late_phase_confidence = float(np.mean(late_c))
        
        # Detect phase transitions
        features.phase_transition_index = detect_phase_transition(
            confidence, threshold=0.3
        )
        
        # Answer surfacing too early detection
        # If high confidence reached before expected reasoning phase
        if len(confidence) >= self.early_answer_threshold:
            early_conf = confidence[:self.early_answer_threshold]
            if any(c > self.high_conf_threshold for c in early_conf):
                features.answer_surfaced_early = True
        
        # Entropy recovery detection
        # Pattern: entropy decreases (exploration) then increases (collapse)
        if len(entropies) >= 6:
            mid_idx = len(entropies) // 2
            first_half = entropies[:mid_idx]
            second_half = entropies[mid_idx:]
            
            first_trend = np.mean(np.diff(first_half)) if len(first_half) > 1 else 0
            second_trend = np.mean(np.diff(second_half)) if len(second_half) > 1 else 0
            
            # Recovery = decrease then increase
            features.entropy_recovery_detected = (first_trend < -0.1 and second_trend > 0.1)
    
    def _extract_trajectory_features(
        self,
        trace: GenerationTrace,
        features: TemporalFeatures
    ) -> None:
        """Extract trajectory-based features"""
        confidence = trace.confidence
        logprobs = trace.logprobs
        entropies = trace.entropies
        
        # Second derivative of confidence (acceleration of belief change)
        if len(confidence) >= 4:
            first_deriv = np.diff(confidence)
            second_deriv = np.diff(first_deriv)
            features.confidence_second_derivative = float(np.max(np.abs(second_deriv)))
        
        # Logprob variance (smooth vs jumpy generation)
        features.logprob_variance = float(np.var(logprobs))

        # Mean logprob (average token likelihood)
        features.mean_logprob = float(np.mean(logprobs))
        
        # Entropy trajectory slope (overall trend)
        if len(entropies) >= 2:
            x = np.arange(len(entropies))
            slope, _ = np.polyfit(x, entropies, 1)
            features.entropy_trajectory_slope = float(slope)
        
        # Confidence monotonicity (how often does it increase vs decrease)
        if len(confidence) >= 2:
            diffs = np.diff(confidence)
            increases = np.sum(diffs > 0)
            total = len(diffs)
            features.confidence_monotonicity = float(increases / total) if total > 0 else 0.5
    
    def _extract_perturbation_features(
        self,
        traces: List[GenerationTrace],
        features: TemporalFeatures,
        levenshtein_func: Optional[callable] = None,
        embed_func: Optional[callable] = None,
    ) -> None:
        """Extract perturbation sensitivity features.

        When embed_func is provided, perturbation_sensitivity is computed as
        1 - median_pairwise_cosine across all trace embeddings.  This measures
        *semantic* instability rather than surface-level text variation.

        Falls back to Levenshtein distance when no embed_func is available.
        """
        # Token Jaccard: average pairwise across all traces
        jaccard_scores = []
        for i in range(len(traces)):
            for j in range(i + 1, len(traces)):
                si = set(traces[i].tokens)
                sj = set(traces[j].tokens)
                jaccard_scores.append(jaccard_similarity(si, sj))
        if jaccard_scores:
            features.token_jaccard = float(np.mean(jaccard_scores))

        # Semantic stability via embedding cosine (preferred)
        if embed_func is not None:
            texts = [t.text for t in traces]
            embeddings = [embed_func(t) for t in texts]
            cosines = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    a, b = embeddings[i], embeddings[j]
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a > 0 and norm_b > 0:
                        cosines.append(float(np.dot(a, b) / (norm_a * norm_b)))
            if cosines:
                features.perturbation_sensitivity = 1.0 - float(np.median(cosines))
            else:
                features.perturbation_sensitivity = 0.0
            return

        # Fallback: Levenshtein between first two traces
        if levenshtein_func is not None and len(traces) >= 2:
            text0 = traces[0].text
            text1 = traces[1].text
            max_len = max(len(text0), len(text1))
            if max_len > 0:
                dist = levenshtein_func(text0, text1)
                features.perturbation_sensitivity = dist / max_len
            else:
                features.perturbation_sensitivity = 0.0


def extract_features_from_raw(
    tokens: List[str],
    logprobs: List[float],
    entropies: List[float],
    temperature: float = 0.7,
    additional_runs: Optional[List[Tuple[List[str], List[float], List[float]]]] = None,
    levenshtein_func: Optional[callable] = None
) -> Optional[TemporalFeatures]:
    """
    Convenience function to extract features from raw generation outputs
    
    Args:
        tokens: List of generated tokens
        logprobs: List of log probabilities
        entropies: List of entropy values
        temperature: Generation temperature
        additional_runs: Optional list of (tokens, logprobs, entropies) for perturbation analysis
        levenshtein_func: Optional Levenshtein distance function
        
    Returns:
        TemporalFeatures or None
    """
    traces = [GenerationTrace(tokens, logprobs, entropies, temperature)]
    
    if additional_runs:
        for t, lp, e in additional_runs:
            traces.append(GenerationTrace(t, lp, e, temperature + 0.1))
    
    extractor = FeatureExtractor()
    return extractor.extract(traces, levenshtein_func)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


def _tokens_to_high_conf_penalty(
    tokens_to_high_conf: float,
    baseline_stats: Optional[Any] = None,
) -> float:
    """
    Compute penalty for how quickly the model reaches high confidence.

    Uses sigmoid((median_t - t) / k) so the penalty is relative to the
    model's calibrated behavior rather than an absolute constant.

    Without calibration, defaults to median=0, k=1 (conservative).
    """
    if baseline_stats is not None:
        median_t = baseline_stats.median
        iqr = baseline_stats.p75 - baseline_stats.p25
        k = max(iqr, 1.0)  # floor at 1.0 to avoid division by zero
    else:
        # Default: assume model typically hits confidence quickly.
        # sigmoid(0) = 0.5, a neutral penalty — honest about ignorance.
        median_t = 0.0
        k = 1.0

    return _sigmoid((median_t - tokens_to_high_conf) / k)


def compute_temporal_debt_components(
    features: TemporalFeatures,
    weights: Optional[Dict[str, float]] = None,
    baseline: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Compute temporal debt components for explainability and governance.

    Args:
        features: Extracted temporal features
        weights: Per-component weights (from risk profile)
        baseline: Optional ModelBaseline for calibrated normalization
    """
    if weights is None:
        weights = {
            'max_confidence_slope': 0.3,
            'confidence_acceleration': 0.2,
            'tokens_to_high_conf': 0.2,
            'entropy_variance': 0.1,
            'perturbation_sensitivity': 0.2,
            'answer_surfaced_early': 0.0,
            'entropy_recovery_detected': 0.1
        }

    # Get baseline stats for tokens_to_high_conf if available
    tthc_stats = None
    if baseline is not None and hasattr(baseline, 'feature_stats'):
        tthc_stats = baseline.feature_stats.get('tokens_to_high_conf')

    tthc_penalty = _tokens_to_high_conf_penalty(
        features.tokens_to_high_conf, tthc_stats
    )

    belief_change = (
        features.max_confidence_slope * weights.get('max_confidence_slope', 0.0) +
        features.confidence_acceleration * weights.get('confidence_acceleration', 0.0) +
        tthc_penalty * weights.get('tokens_to_high_conf', 0.0)
    )
    
    evidence = (
        features.entropy_variance * weights.get('entropy_variance', 0.0) +
        features.perturbation_sensitivity * weights.get('perturbation_sensitivity', 0.0)
    )
    
    phase_penalty = 0.0
    if features.answer_surfaced_early:
        phase_penalty += weights.get('answer_surfaced_early', 0.0)
    if features.entropy_recovery_detected:
        phase_penalty += weights.get('entropy_recovery_detected', 0.0)
    
    return {
        'belief_change': belief_change,
        'evidence': evidence,
        'phase_penalty': phase_penalty,
        'raw_total': belief_change + evidence + phase_penalty
    }


def compute_temporal_debt(
    features: TemporalFeatures,
    weights: Optional[Dict[str, float]] = None,
    baseline: Optional[Any] = None,
) -> float:
    """
    Compute temporal debt score based on features

    Temporal debt = rate of belief change - rate of evidence accumulation

    Higher score = more debt = more likely hallucination
    """
    components = compute_temporal_debt_components(features, weights, baseline)
    debt = components['raw_total']

    # Normalize to 0-1 range (approximately)
    return min(1.0, max(0.0, debt))
