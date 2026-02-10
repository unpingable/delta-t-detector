"""
Δt Hallucination Detector - Core Module
Main detector class integrating all components
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings

from .config import (
    DetectorConfig, RiskProfile, RISK_PROFILES, 
    get_config, set_profile
)
from .features import (
    FeatureExtractor, TemporalFeatures, GenerationTrace,
    extract_features_from_raw, compute_temporal_debt,
    compute_temporal_debt_components
)
from .baseline import (
    BaselineProfiler, ModelBaseline, 
    normalize_with_baseline, compute_anomaly_score
)
from .invariants import (
    TemporalCoherenceTest, SemanticConservationTest,
    EpistemicGroundingTest, IrreversibilityTest,
    MultiInvariantValidator, InvariantResult, MultiInvariantResult
)
from .reporting import (
    DetectionReport, ReportGenerator,
    format_console_report, format_json_report
)
from .utils import to_plain

# Suppress only specific noisy warnings, not everything
import warnings
warnings.filterwarnings('ignore', message='.*resume_download.*')  # HF deprecation noise


@dataclass
class DetectionResult:
    """Simple detection result container"""
    prediction: str  # Single-invariant: 'temporal_stable'/'temporal_unstable';
                     # Multi-invariant: 'FAIL'/'WARN'/'CLEAN';
                     # Either: 'uncertain'/'unknown'
    confidence: float
    temporal_debt: float
    features: Dict[str, Any]
    report: Optional[DetectionReport] = None
    fail_type: Optional[str] = None
    fail_subjects: Optional[list] = None
    warn_type: Optional[str] = None


class DeltaTDetector:
    """
    Main Δt Hallucination Detector
    
    Detects hallucinations by measuring temporal coherence violations -
    when belief change rate (dC/dt) exceeds evidence accumulation rate (dE/dt)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        config: Optional[DetectorConfig] = None,
        load_model: bool = True
    ):
        """
        Initialize the detector
        
        Args:
            model_name: HuggingFace model name/path
            config: Optional DetectorConfig, uses default if not provided
            load_model: Whether to load the LLM immediately
        """
        self.config = config or get_config()
        self.model_name = model_name
        
        # Core components
        self.model = None
        self.tokenizer = None
        self.device = None
        
        # Processors
        self.feature_extractor = FeatureExtractor()
        self.baseline_profiler = BaselineProfiler(self.config.baseline_path)
        self.report_generator = ReportGenerator(self.config.verbose)
        
        # Invariant testers
        self._init_invariant_testers()
        
        # Current baseline
        self._current_baseline: Optional[ModelBaseline] = None
        
        # Levenshtein function (lazy loaded)
        self._levenshtein_func = None

        # Sentence embedding model (lazy loaded)
        self._embed_model = None
        self._embed_func = None

        # Load model if requested
        if load_model:
            self._load_model()
    
    def _get_levenshtein(self):
        """Lazy load Levenshtein distance function"""
        if self._levenshtein_func is None:
            try:
                from Levenshtein import distance
                self._levenshtein_func = distance
            except ImportError:
                # Fallback to utils implementation
                from .utils import levenshtein_similarity
                def lev_dist(s1, s2):
                    sim = levenshtein_similarity(s1, s2)
                    return int((1 - sim) * max(len(s1), len(s2)))
                self._levenshtein_func = lev_dist
        return self._levenshtein_func

    def _get_embed_func(self):
        """Lazy load sentence-transformer embedding function.

        Returns a callable (str -> np.ndarray) or None if unavailable.
        """
        if self._embed_func is not None:
            return self._embed_func
        try:
            from sentence_transformers import SentenceTransformer
            self._embed_model = SentenceTransformer(
                'all-MiniLM-L6-v2', device=self.device or 'cpu'
            )
            self._embed_func = lambda text: self._embed_model.encode(
                text, convert_to_numpy=True, show_progress_bar=False
            )
            return self._embed_func
        except Exception:
            return None
    
    def _init_invariant_testers(self) -> None:
        """Initialize invariant test classes based on current profile"""
        profile = self.config.get_profile()
        
        self.temporal_coherence_test = TemporalCoherenceTest(
            threshold=profile.temporal_threshold,
            tokens_to_conf_threshold=profile.tokens_to_confidence_threshold,
            acceleration_threshold=profile.confidence_acceleration_threshold,
            zscore_threshold=profile.baseline_zscore_threshold
        )
        
        self.semantic_conservation_test = SemanticConservationTest(
            similarity_threshold=profile.semantic_similarity_threshold,
            n_variants=profile.semantic_variants
        )
        
        self.epistemic_grounding_test = EpistemicGroundingTest(
            max_fabricated=profile.max_fabricated_citations,
            validate_urls=profile.validate_urls
        )
        
        self.irreversibility_test = IrreversibilityTest(
            similarity_threshold=0.7,
            n_trials=2
        )
        
        self.multi_invariant_validator = MultiInvariantValidator(
            min_invariants_required=profile.min_invariants_required,
            weights=profile.invariant_weights
        )
    
    def _load_model(self) -> None:
        """Load the language model"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading model: {self.model_name}")
        
        # Determine device with CUDA > MPS > CPU priority
        self.device = self._select_device(self.config.device)
        print(f"Selected device: {self.device}")
        
        # Determine dtype based on device capabilities
        dtype = self._select_dtype(self.config.dtype, self.device)
        print(f"Selected dtype: {dtype}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model - CPU first then move to device (cleanest cross-platform approach)
        # This avoids MPS direct-load quirks and device_map complexity
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        # Move to target device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device} with dtype {dtype}")
    
    def _get_model_device(self) -> torch.device:
        """Get the device the model is actually on (handles edge cases)"""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")
    
    def _select_device(self, preference: str) -> str:
        """
        Select the best available compute device
        
        Priority: cuda > mps > cpu (unless preference is set)
        
        Args:
            preference: 'auto', 'cuda', 'mps', or 'cpu'
            
        Returns:
            Device string for PyTorch
        """
        if preference == "auto":
            # Check CUDA first (NVIDIA GPUs)
            if torch.cuda.is_available():
                return "cuda"
            
            # Check MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                try:
                    # Verify MPS is actually working
                    torch.zeros(1).to('mps')
                    return "mps"
                except Exception:
                    print("Warning: MPS available but not functional, falling back to CPU")
                    return "cpu"
            
            # Fallback to CPU
            return "cpu"
        
        elif preference == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available, falling back to CPU")
                return "cpu"
            return "cuda"
        
        elif preference == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available, falling back to CPU")
                return "cpu"
            return "mps"
        
        else:
            return "cpu"
    
    def _select_dtype(self, preference: str, device: str) -> torch.dtype:
        """
        Select appropriate dtype for the device
        
        Args:
            preference: 'auto', 'float16', 'bfloat16', or 'float32'
            device: Selected device string
            
        Returns:
            PyTorch dtype
        """
        if preference != "auto":
            dtype_map = {
                'float16': torch.float16,
                'bfloat16': torch.bfloat16,
                'float32': torch.float32
            }
            return dtype_map.get(preference, torch.float32)
        
        # Auto-select based on device
        if device == "cuda":
            # CUDA: prefer bfloat16 if supported, else float16
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        
        elif device == "mps":
            # MPS: float16 works well on Apple Silicon
            # bfloat16 has limited MPS support
            return torch.float16
        
        else:
            # CPU: float32 is most reliable
            return torch.float32
    
    # =========================================================================
    # Risk Profile Management
    # =========================================================================
    
    def set_risk_profile(self, profile_name: str) -> RiskProfile:
        """
        Set the active risk profile
        
        Args:
            profile_name: One of 'medical', 'legal', 'research', 'general', 
                         'creative', 'entertainment'
                         
        Returns:
            The activated RiskProfile
        """
        self.config.set_profile(profile_name)
        self._init_invariant_testers()  # Reinitialize with new thresholds
        return self.config.get_profile()
    
    def get_risk_profile(self) -> RiskProfile:
        """Get the current risk profile"""
        return self.config.get_profile()
    
    def list_risk_profiles(self) -> List[str]:
        """List available risk profiles"""
        return list(RISK_PROFILES.keys())
    
    # =========================================================================
    # Baseline Management
    # =========================================================================
    
    def load_baseline(self, model_name: Optional[str] = None) -> Optional[ModelBaseline]:
        """Load a baseline profile for the specified model"""
        name = model_name or self.model_name
        self._current_baseline = self.baseline_profiler.load_baseline(name)
        return self._current_baseline
    
    def create_baseline(
        self,
        calibration_prompts: List[str],
        dataset_name: str = "custom"
    ) -> ModelBaseline:
        """
        Create a baseline profile from calibration data
        
        Args:
            calibration_prompts: List of prompts to use for calibration
                                (should be prompts with known truthful answers)
            dataset_name: Name to identify this calibration set
            
        Returns:
            Created ModelBaseline
        """
        print(f"Creating baseline from {len(calibration_prompts)} prompts...")
        
        features_list = []
        for i, prompt in enumerate(calibration_prompts):
            if self.config.verbose and i % 10 == 0:
                print(f"  Calibrating: {i}/{len(calibration_prompts)}")
            
            features = self.compute_features(prompt)
            if features is not None:
                features_list.append(features)
        
        baseline = self.baseline_profiler.create_baseline(
            model_name=self.model_name,
            features_list=features_list,
            dataset_name=dataset_name
        )
        
        self.baseline_profiler.save_baseline(baseline)
        self._current_baseline = baseline
        
        print(f"Baseline created from {len(features_list)} samples")
        return baseline
    
    # =========================================================================
    # Generation with Metrics
    # =========================================================================
    
    def generate_with_metrics(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.7
    ) -> GenerationTrace:
        """
        Generate text while tracking logprobs and entropy per token
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            GenerationTrace with tokens, logprobs, entropies
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Initialize with load_model=True or call _load_model().")
        # Get actual model device (safer than assuming self.device)
        device = self._get_model_device()
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        logprobs = []
        entropies = []
        tokens = []
        
        curr_input_ids = inputs['input_ids']
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                try:
                    outputs = self.model(curr_input_ids)
                    next_token_logits = outputs.logits[:, -1, :]
                    
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                    
                    next_token_index = torch.multinomial(probs, num_samples=1)
                    token_id = next_token_index.item()
                    
                    token_logprob = torch.log(probs[0, token_id] + 1e-10)
                    
                    logprobs.append(float(token_logprob))
                    entropies.append(float(entropy))
                    tokens.append(self.tokenizer.decode([token_id]))
                    
                    if token_id == self.tokenizer.eos_token_id:
                        break
                        
                    curr_input_ids = torch.cat([curr_input_ids, next_token_index], dim=-1)
                except Exception as e:
                    print(f"Generation error: {e}")
                    break
        
        return GenerationTrace(tokens, logprobs, entropies, temperature)
    
    # =========================================================================
    # Feature Computation
    # =========================================================================
    
    def compute_features(
        self,
        prompt: str,
        n_perturbations: int = 2,
        max_new_tokens: int = 80
    ) -> Optional[TemporalFeatures]:
        """
        Compute Δt features for a prompt
        
        Args:
            prompt: Input prompt
            n_perturbations: Number of generation runs with different temperatures
            max_new_tokens: Maximum tokens per generation
            
        Returns:
            TemporalFeatures or None if extraction fails
        """
        profile = self.config.get_profile()
        n_perturbations = profile.n_perturbations
        
        traces = []
        
        for i in range(n_perturbations):
            temp = self.config.temperature_base + (i * self.config.temperature_step)
            try:
                trace = self.generate_with_metrics(
                    prompt, 
                    max_new_tokens=max_new_tokens,
                    temperature=temp
                )
                traces.append(trace)
            except Exception as e:
                print(f"Feature computation error: {e}")
                if i == 0:
                    return None
        
        if not traces:
            return None
        
        return self.feature_extractor.extract(
            traces,
            levenshtein_func=self._get_levenshtein(),
            embed_func=self._get_embed_func(),
        )
    
    # =========================================================================
    # Detection Methods
    # =========================================================================
    
    def detect(
        self,
        prompt: str,
        generate_report: bool = True
    ) -> DetectionResult:
        """
        Detect if a prompt's response is likely hallucinated
        
        This uses temporal coherence (Invariant 1) only.
        For comprehensive detection, use detect_multi_invariant()
        
        Args:
            prompt: Input prompt
            generate_report: Whether to generate full report
            
        Returns:
            DetectionResult
        """
        features = self.compute_features(prompt)
        
        if features is None:
            return DetectionResult(
                prediction='unknown',
                confidence=0.0,
                temporal_debt=0.0,
                features={},
                report=None
            )
        
        # Get threshold from profile
        profile = self.config.get_profile()

        # Compute temporal debt
        debt = compute_temporal_debt(features, weights=profile.temporal_debt_weights, baseline=self._current_baseline)
        
        # Normalize with baseline if available
        normalized = None
        anomaly_score = None
        if self._current_baseline:
            normalized = self._current_baseline.normalize_features(features)
            anomaly_score = compute_anomaly_score(features, self._current_baseline)
        
        # Test temporal coherence
        tc_result = self.temporal_coherence_test.test(
            features.to_dict(),
            normalized
        )
        
        # Decision (single-invariant: temporal labels only)
        if tc_result.violated or debt > profile.temporal_threshold:
            prediction = 'temporal_unstable'
            confidence = min(0.95, 0.5 + debt)
        else:
            prediction = 'temporal_stable'
            confidence = max(0.5, tc_result.score)

        # Precision guards: cap confident stability on low evidence or anomalies
        if prediction == 'temporal_stable' and debt > profile.temporal_debt_confidence_cap_threshold:
            confidence = min(confidence, profile.truthful_confidence_cap)
        
        if (features.entropy_variance < profile.low_evidence_entropy_threshold and
            features.tokens_to_high_conf < profile.low_evidence_tokens_threshold):
            prediction = 'uncertain'
            confidence = min(confidence, profile.low_evidence_confidence_cap)
        
        if anomaly_score is not None and anomaly_score > profile.anomaly_score_uncertain_threshold:
            prediction = 'uncertain'
            confidence = min(confidence, profile.anomaly_confidence_cap)
        
        # Generate report
        report = None
        if generate_report:
            report = self.report_generator.generate(
                prediction=prediction,
                confidence=confidence,
                features=features,
                invariant_result=MultiInvariantResult(
                    results={'temporal_coherence': tc_result},
                    n_violated=1 if tc_result.violated else 0,
                    aggregate_score=tc_result.score,
                    prediction=prediction,
                    confidence=confidence
                ),
                baseline=self._current_baseline,
                profile=profile,
                temporal_debt=debt,
                anomaly_score=anomaly_score,
                prompt=prompt
            )
        
        return DetectionResult(
            prediction=prediction,
            confidence=confidence,
            temporal_debt=debt,
            features=features.to_dict(),
            report=report
        )
    
    def detect_with_baseline(
        self,
        prompt: str,
        model_name: Optional[str] = None
    ) -> DetectionResult:
        """
        Detect with baseline normalization
        
        Args:
            prompt: Input prompt
            model_name: Optional model name for baseline lookup
            
        Returns:
            DetectionResult with baseline-normalized features
        """
        # Ensure baseline is loaded
        if self._current_baseline is None:
            self.load_baseline(model_name)
        
        return self.detect(prompt, generate_report=True)
    
    def detect_multi_invariant(
        self,
        prompt: str,
        validate_citations: bool = True,
        test_semantic: bool = True,
        test_irreversibility: bool = False,
        expected_min_anchors: Optional[int] = None,
    ) -> DetectionResult:
        """
        Full multi-invariant detection
        
        Tests all four invariants for comprehensive detection
        
        Args:
            prompt: Input prompt
            validate_citations: Whether to validate URLs/DOIs (requires network)
            test_semantic: Whether to test semantic conservation (slower)
            test_irreversibility: Whether to test irreversibility (requires extra generation)
            
        Returns:
            DetectionResult with multi-invariant analysis
        """
        profile = self.config.get_profile()
        results: Dict[str, InvariantResult] = {}
        
        # Generate primary response
        features = self.compute_features(prompt)
        
        if features is None:
            return DetectionResult(
                prediction='unknown',
                confidence=0.0,
                temporal_debt=0.0,
                features={},
                report=None
            )
        
        # Invariant 1: Temporal Coherence
        normalized = None
        anomaly_score = None
        if self._current_baseline:
            normalized = self._current_baseline.normalize_features(features)
            anomaly_score = compute_anomaly_score(features, self._current_baseline)
        
        results['temporal_coherence'] = self.temporal_coherence_test.test(
            features.to_dict(), normalized
        )
        
        # Invariant 2: Semantic Conservation
        if test_semantic and profile.require_semantic_stability:
            variants = self.semantic_conservation_test.generate_variants(prompt)
            responses = [features.generation]
            
            for variant in variants[1:]:
                trace = self.generate_with_metrics(variant, max_new_tokens=80)
                responses.append(trace.text)
            
            results['semantic_conservation'] = self.semantic_conservation_test.test(
                responses, variants
            )
        
        # Invariant 3: Epistemic Grounding
        if profile.require_grounding:
            results['epistemic_grounding'] = self.epistemic_grounding_test.test(
                features.generation,
                validate=validate_citations
            )
        
        # Invariant 4: Irreversibility
        if test_irreversibility:
            responses = [features.generation]
            trace2 = self.generate_with_metrics(prompt, max_new_tokens=80)
            responses.append(trace2.text)
            
            results['irreversibility'] = self.irreversibility_test.test(responses)
        
        # Aggregate results
        multi_result = self.multi_invariant_validator.aggregate(
            results, expected_min_anchors=expected_min_anchors)
        
        # Compute temporal debt
        debt = compute_temporal_debt(features, weights=profile.temporal_debt_weights, baseline=self._current_baseline)
        
        # Precision guards: cap confident truthfulness on low evidence or anomalies
        prediction = multi_result.prediction
        confidence = multi_result.confidence
        
        if prediction == 'CLEAN' and debt > profile.temporal_debt_confidence_cap_threshold:
            confidence = min(confidence, profile.truthful_confidence_cap)

        if (features.entropy_variance < profile.low_evidence_entropy_threshold and
            features.tokens_to_high_conf < profile.low_evidence_tokens_threshold):
            prediction = 'uncertain'
            confidence = min(confidence, profile.low_evidence_confidence_cap)

        if anomaly_score is not None and anomaly_score > profile.anomaly_score_uncertain_threshold:
            prediction = 'uncertain'
            confidence = min(confidence, profile.anomaly_confidence_cap)

        # Generate report
        report = self.report_generator.generate(
            prediction=prediction,
            confidence=confidence,
            features=features,
            invariant_result=multi_result,
            baseline=self._current_baseline,
            profile=profile,
            temporal_debt=debt,
            anomaly_score=anomaly_score,
            prompt=prompt
        )

        return DetectionResult(
            prediction=prediction,
            confidence=confidence,
            temporal_debt=debt,
            features=features.to_dict(),
            report=report,
            fail_type=multi_result.fail_type,
            fail_subjects=multi_result.fail_subjects,
            warn_type=multi_result.warn_type,
        )

    # =========================================================================
    # Ensemble Detection
    # =========================================================================
    
    def ensemble_detect(
        self,
        prompt: str,
        models: List[str],
        agreement_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Run detection across multiple models and aggregate
        
        Args:
            prompt: Input prompt
            models: List of model names to use
            agreement_threshold: Required agreement level for confident prediction
            
        Returns:
            Aggregated results with per-model breakdown
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        results = {}
        predictions = []
        
        original_model = self.model
        original_tokenizer = self.tokenizer
        original_name = self.model_name
        
        for model_name in models:
            try:
                # Load model
                self.model_name = model_name
                self._load_model()
                
                # Run detection
                result = self.detect(prompt, generate_report=False)
                results[model_name] = {
                    'prediction': result.prediction,
                    'confidence': result.confidence,
                    'temporal_debt': result.temporal_debt
                }
                predictions.append(result.prediction)
                
            except Exception as e:
                results[model_name] = {'error': str(e)}
        
        # Restore original model
        self.model = original_model
        self.tokenizer = original_tokenizer
        self.model_name = original_name
        
        # Aggregate predictions
        hallucination_count = predictions.count('hallucination')
        truthful_count = predictions.count('truthful')
        total = len(predictions)
        
        if hallucination_count / total >= agreement_threshold:
            ensemble_prediction = 'hallucination'
            ensemble_confidence = hallucination_count / total
        elif truthful_count / total >= agreement_threshold:
            ensemble_prediction = 'truthful'
            ensemble_confidence = truthful_count / total
        else:
            ensemble_prediction = 'uncertain'
            ensemble_confidence = max(hallucination_count, truthful_count) / total
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'ensemble_confidence': ensemble_confidence,
            'model_results': results,
            'agreement': max(hallucination_count, truthful_count) / total if total > 0 else 0
        }
    
    # =========================================================================
    # Explanation & Reporting
    # =========================================================================
    
    def explain_decision(self, result: DetectionResult) -> str:
        """
        Get human-readable explanation for a detection result
        
        Args:
            result: DetectionResult to explain
            
        Returns:
            Formatted explanation string
        """
        if result.report:
            return format_console_report(result.report)
        else:
            return f"Prediction: {result.prediction} (confidence: {result.confidence:.2f})"
    
    def get_json_report(self, result: DetectionResult) -> str:
        """Get JSON-formatted report"""
        if result.report:
            return format_json_report(result.report)
        return "{}"


# ============================================================================
# Convenience Functions
# ============================================================================

def quick_detect(
    prompt: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    risk_profile: str = "general"
) -> DetectionResult:
    """
    Quick single-shot detection
    
    Args:
        prompt: Input to detect
        model_name: Model to use
        risk_profile: Risk profile to use
        
    Returns:
        DetectionResult
    """
    detector = DeltaTDetector(model_name=model_name)
    detector.set_risk_profile(risk_profile)
    return detector.detect(prompt)


def batch_detect(
    prompts: List[str],
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    risk_profile: str = "general",
    progress: bool = True
) -> List[DetectionResult]:
    """
    Batch detection for multiple prompts
    
    Args:
        prompts: List of prompts to detect
        model_name: Model to use
        risk_profile: Risk profile to use
        progress: Show progress bar
        
    Returns:
        List of DetectionResults
    """
    detector = DeltaTDetector(model_name=model_name)
    detector.set_risk_profile(risk_profile)
    
    results = []
    
    if progress:
        try:
            from tqdm import tqdm
            prompts = tqdm(prompts, desc="Detecting")
        except ImportError:
            pass
    
    for prompt in prompts:
        results.append(detector.detect(prompt, generate_report=False))
    
    return results
