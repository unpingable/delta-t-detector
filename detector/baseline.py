"""
Δt Hallucination Detector - Baseline Profiling Module
Model-specific calibration and normalization
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .features import TemporalFeatures


@dataclass
class FeatureStats:
    """Statistics for a single feature"""
    mean: float
    std: float
    median: float
    p5: float   # 5th percentile
    p25: float  # 25th percentile  
    p75: float  # 75th percentile
    p95: float  # 95th percentile
    min_val: float
    max_val: float
    n_samples: int
    
    def z_score(self, value: float) -> float:
        """Compute z-score for a value"""
        if self.std <= 0:
            return 0.0
        return (value - self.mean) / self.std
    
    def percentile_score(self, value: float) -> float:
        """Estimate percentile position of a value (0-1)"""
        if value <= self.p5:
            return 0.05
        elif value <= self.p25:
            return 0.05 + 0.20 * (value - self.p5) / (self.p25 - self.p5 + 1e-10)
        elif value <= self.median:
            return 0.25 + 0.25 * (value - self.p25) / (self.median - self.p25 + 1e-10)
        elif value <= self.p75:
            return 0.50 + 0.25 * (value - self.median) / (self.p75 - self.median + 1e-10)
        elif value <= self.p95:
            return 0.75 + 0.20 * (value - self.p75) / (self.p95 - self.p75 + 1e-10)
        else:
            return 0.95


@dataclass
class ModelBaseline:
    """Baseline profile for a specific model"""
    model_name: str
    created_at: str
    n_calibration_samples: int
    calibration_dataset: str
    feature_stats: Dict[str, FeatureStats]
    metadata: Dict[str, Any]
    
    def normalize_features(self, features: TemporalFeatures) -> Dict[str, float]:
        """
        Normalize features relative to this baseline
        
        Returns z-scores for each feature
        """
        normalized = {}
        feature_dict = features.to_dict()
        
        for name, value in feature_dict.items():
            if name in self.feature_stats and isinstance(value, (int, float)):
                stats = self.feature_stats[name]
                normalized[name] = value
                normalized[f"{name}_zscore"] = stats.z_score(value)
                normalized[f"{name}_percentile"] = stats.percentile_score(value)
            elif isinstance(value, (int, float)):
                normalized[name] = value
        
        return normalized
    
    def is_anomalous(
        self, 
        features: TemporalFeatures,
        z_threshold: float = 2.0
    ) -> Tuple[bool, List[str]]:
        """
        Check if features are anomalous relative to baseline
        
        Returns (is_anomalous, list of anomalous features)
        """
        anomalies = []
        feature_dict = features.to_dict()
        
        for name, value in feature_dict.items():
            if name in self.feature_stats and isinstance(value, (int, float)):
                z = self.feature_stats[name].z_score(value)
                if abs(z) > z_threshold:
                    anomalies.append(name)
        
        return len(anomalies) > 0, anomalies
    
    def save(self, path: str) -> None:
        """Save baseline to JSON file"""
        data = {
            'model_name': self.model_name,
            'created_at': self.created_at,
            'n_calibration_samples': self.n_calibration_samples,
            'calibration_dataset': self.calibration_dataset,
            'feature_stats': {
                k: asdict(v) for k, v in self.feature_stats.items()
            },
            'metadata': self.metadata
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelBaseline':
        """Load baseline from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        feature_stats = {
            k: FeatureStats(**v) for k, v in data['feature_stats'].items()
        }
        
        return cls(
            model_name=data['model_name'],
            created_at=data['created_at'],
            n_calibration_samples=data['n_calibration_samples'],
            calibration_dataset=data['calibration_dataset'],
            feature_stats=feature_stats,
            metadata=data.get('metadata', {})
        )


class BaselineProfiler:
    """
    Create and manage baseline profiles for different models
    """
    
    def __init__(self, baselines_dir: str = "./baselines"):
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_baselines: Dict[str, ModelBaseline] = {}
    
    def create_baseline(
        self,
        model_name: str,
        features_list: List[TemporalFeatures],
        dataset_name: str = "calibration",
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelBaseline:
        """
        Create a baseline profile from calibration data
        
        Args:
            model_name: Name/identifier of the model
            features_list: List of TemporalFeatures from truthful responses
            dataset_name: Name of the calibration dataset
            metadata: Optional additional metadata
            
        Returns:
            ModelBaseline object
        """
        if len(features_list) < 10:
            raise ValueError(f"Need at least 10 samples for calibration, got {len(features_list)}")
        
        # Collect all feature values
        feature_columns = TemporalFeatures.feature_columns()
        feature_values: Dict[str, List[float]] = {col: [] for col in feature_columns}
        
        for features in features_list:
            feature_dict = features.to_dict()
            for col in feature_columns:
                if col in feature_dict and isinstance(feature_dict[col], (int, float)):
                    feature_values[col].append(float(feature_dict[col]))
        
        # Compute statistics for each feature
        feature_stats = {}
        for col, values in feature_values.items():
            if len(values) >= 5:  # Need minimum samples
                arr = np.array(values)
                feature_stats[col] = FeatureStats(
                    mean=float(np.mean(arr)),
                    std=float(np.std(arr)),
                    median=float(np.median(arr)),
                    p5=float(np.percentile(arr, 5)),
                    p25=float(np.percentile(arr, 25)),
                    p75=float(np.percentile(arr, 75)),
                    p95=float(np.percentile(arr, 95)),
                    min_val=float(np.min(arr)),
                    max_val=float(np.max(arr)),
                    n_samples=len(values)
                )
        
        baseline = ModelBaseline(
            model_name=model_name,
            created_at=datetime.now().isoformat(),
            n_calibration_samples=len(features_list),
            calibration_dataset=dataset_name,
            feature_stats=feature_stats,
            metadata=metadata or {}
        )
        
        # Cache it
        self._loaded_baselines[model_name] = baseline
        
        return baseline
    
    def save_baseline(self, baseline: ModelBaseline) -> str:
        """Save a baseline to disk"""
        # Sanitize model name for filename
        safe_name = baseline.model_name.replace('/', '_').replace('\\', '_')
        filename = f"{safe_name}_baseline.json"
        path = self.baselines_dir / filename
        baseline.save(str(path))
        return str(path)
    
    def load_baseline(self, model_name: str) -> Optional[ModelBaseline]:
        """Load a baseline from disk"""
        # Check cache first
        if model_name in self._loaded_baselines:
            return self._loaded_baselines[model_name]
        
        # Try to load from file
        safe_name = model_name.replace('/', '_').replace('\\', '_')
        filename = f"{safe_name}_baseline.json"
        path = self.baselines_dir / filename
        
        if path.exists():
            baseline = ModelBaseline.load(str(path))
            self._loaded_baselines[model_name] = baseline
            return baseline
        
        return None
    
    def list_baselines(self) -> List[str]:
        """List available baseline profiles"""
        baselines = []
        for path in self.baselines_dir.glob("*_baseline.json"):
            try:
                baseline = ModelBaseline.load(str(path))
                baselines.append(baseline.model_name)
            except Exception:
                continue
        return baselines
    
    def get_or_create_baseline(
        self,
        model_name: str,
        calibration_func: callable,
        n_samples: int = 100,
        force_recreate: bool = False
    ) -> ModelBaseline:
        """
        Get existing baseline or create new one
        
        Args:
            model_name: Name of the model
            calibration_func: Function that returns List[TemporalFeatures] for calibration
            n_samples: Number of samples for calibration
            force_recreate: If True, always recreate baseline
            
        Returns:
            ModelBaseline
        """
        if not force_recreate:
            existing = self.load_baseline(model_name)
            if existing is not None:
                return existing
        
        # Run calibration
        features_list = calibration_func(n_samples)
        baseline = self.create_baseline(model_name, features_list)
        self.save_baseline(baseline)
        
        return baseline


def normalize_with_baseline(
    features: TemporalFeatures,
    baseline: Optional[ModelBaseline]
) -> Dict[str, float]:
    """
    Normalize features using baseline if available
    
    Returns either normalized features or raw features
    """
    if baseline is None:
        return features.to_dict()
    
    return baseline.normalize_features(features)


def compute_anomaly_score(
    features: TemporalFeatures,
    baseline: ModelBaseline,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute overall anomaly score relative to baseline
    
    Higher score = more anomalous = more likely hallucination
    """
    if weights is None:
        weights = {
            'max_confidence_slope': 1.0,
            'tokens_to_high_conf': 1.0,
            'entropy_variance': 0.8,
            'perturbation_sensitivity': 0.9,
            'confidence_acceleration': 0.7
        }
    
    normalized = baseline.normalize_features(features)
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for feature, weight in weights.items():
        zscore_key = f"{feature}_zscore"
        if zscore_key in normalized:
            # Use absolute z-score (anomalies can be in either direction)
            weighted_sum += abs(normalized[zscore_key]) * weight
            total_weight += weight
    
    if total_weight > 0:
        return weighted_sum / total_weight
    
    return 0.0
