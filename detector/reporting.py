"""
Δt Hallucination Detector - Reporting Module
Comprehensive diagnostic output and explanations
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .features import TemporalFeatures
from .invariants import MultiInvariantResult, InvariantResult
from .baseline import ModelBaseline
from .config import RiskProfile
from .utils import to_plain


@dataclass
class DetectionReport:
    """Comprehensive detection report"""
    
    # Core prediction
    prediction: str  # 'hallucination' or 'truthful'
    confidence: float
    
    # Context
    model_baseline: Optional[str]
    risk_profile: str
    timestamp: str
    
    # Invariant results
    invariant_results: Dict[str, Dict[str, Any]]
    n_invariants_violated: int
    
    # Temporal features
    temporal_features: Dict[str, Any]
    
    # Analysis
    temporal_debt_score: float
    anomaly_score: Optional[float]
    
    # Human-readable explanation
    explanation: str
    key_findings: List[str]
    recommendations: List[str]
    
    # Raw data
    generation: str
    prompt: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return to_plain(asdict(self))
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, default=to_plain)
    
    def summary(self) -> str:
        """Generate a brief summary"""
        emoji = "⚠️" if self.prediction == 'hallucination' else "✓"
        return (
            f"{emoji} Prediction: {self.prediction.upper()} "
            f"(confidence: {self.confidence:.1%})\n"
            f"   Invariants violated: {self.n_invariants_violated}/4\n"
            f"   Risk profile: {self.risk_profile}\n"
            f"   Key issue: {self.key_findings[0] if self.key_findings else 'None'}"
        )


class ReportGenerator:
    """
    Generates comprehensive detection reports
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
    
    def generate(
        self,
        prediction: str,
        confidence: float,
        features: TemporalFeatures,
        invariant_result: Optional[MultiInvariantResult] = None,
        baseline: Optional[ModelBaseline] = None,
        profile: Optional[RiskProfile] = None,
        temporal_debt: float = 0.0,
        anomaly_score: Optional[float] = None,
        prompt: Optional[str] = None
    ) -> DetectionReport:
        """
        Generate a comprehensive detection report
        
        Args:
            prediction: 'hallucination' or 'truthful'
            confidence: Prediction confidence
            features: Extracted temporal features
            invariant_result: Optional multi-invariant test results
            baseline: Optional model baseline used
            profile: Risk profile used
            temporal_debt: Computed temporal debt score
            anomaly_score: Optional anomaly score from baseline comparison
            prompt: Original prompt (optional)
            
        Returns:
            DetectionReport
        """
        # Prepare invariant results
        if invariant_result:
            inv_dict = {
                name: result.to_dict() 
                for name, result in invariant_result.results.items()
            }
            n_violated = invariant_result.n_violated
        else:
            inv_dict = {}
            n_violated = 0
        
        # Generate explanation
        explanation, key_findings = self._generate_explanation(
            prediction, features, invariant_result, temporal_debt
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            prediction, features, invariant_result, profile
        )
        
        return DetectionReport(
            prediction=prediction,
            confidence=confidence,
            model_baseline=baseline.model_name if baseline else None,
            risk_profile=profile.name if profile else 'default',
            timestamp=datetime.now().isoformat(),
            invariant_results=inv_dict,
            n_invariants_violated=n_violated,
            temporal_features=features.to_dict(),
            temporal_debt_score=temporal_debt,
            anomaly_score=anomaly_score,
            explanation=explanation,
            key_findings=key_findings,
            recommendations=recommendations,
            generation=features.generation,
            prompt=prompt
        )
    
    def _generate_explanation(
        self,
        prediction: str,
        features: TemporalFeatures,
        invariant_result: Optional[MultiInvariantResult],
        temporal_debt: float
    ) -> tuple[str, List[str]]:
        """Generate human-readable explanation"""
        key_findings = []
        
        if prediction == 'hallucination':
            # Identify primary issues
            if features.tokens_to_high_conf < 6:
                key_findings.append(
                    f"High confidence reached in only {features.tokens_to_high_conf:.0f} tokens "
                    "(expected: 10+ for well-grounded responses)"
                )
            
            if features.max_confidence_slope > 0.5:
                key_findings.append(
                    f"Rapid confidence increase detected (slope: {features.max_confidence_slope:.2f})"
                )
            
            if features.answer_surfaced_early:
                key_findings.append(
                    "Answer appeared before adequate reasoning/context phase"
                )
            
            if features.entropy_recovery_detected:
                key_findings.append(
                    "Entropy recovery pattern suggests exploration followed by forced collapse"
                )
            
            if invariant_result:
                for name, result in invariant_result.results.items():
                    if result.violated:
                        if name == 'epistemic_grounding':
                            inv = result.details.get('invalid_count', 0)
                            key_findings.append(f"Citation validation failed ({inv} invalid)")
                        elif name == 'semantic_conservation':
                            sim = result.details.get('min_similarity', 0)
                            key_findings.append(f"Semantic drift detected (min similarity: {sim:.2f})")
            
            if not key_findings:
                key_findings.append(f"Elevated temporal debt score ({temporal_debt:.2f})")
            
            explanation = (
                f"High temporal debt ({temporal_debt:.2f}) indicates the model reached "
                f"high confidence ({features.final_confidence:.2f}) faster than evidence "
                f"accumulation rate justifies. "
            )
            
            if key_findings:
                explanation += f"Primary concern: {key_findings[0].lower()}"
        
        else:  # truthful
            key_findings.append("Temporal coherence maintained throughout generation")
            
            if features.tokens_to_high_conf >= 10:
                key_findings.append(
                    f"Appropriate reasoning time ({features.tokens_to_high_conf:.0f} tokens "
                    "before high confidence)"
                )
            
            if features.perturbation_sensitivity < 0.3:
                key_findings.append("Response stable under perturbation")
            
            explanation = (
                f"Low temporal debt ({temporal_debt:.2f}) suggests confidence "
                f"accumulated at a rate consistent with evidence gathering. "
                f"Generation shows stable temporal trajectory."
            )
        
        return explanation, key_findings
    
    def _generate_recommendations(
        self,
        prediction: str,
        features: TemporalFeatures,
        invariant_result: Optional[MultiInvariantResult],
        profile: Optional[RiskProfile]
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if prediction == 'hallucination':
            # Universal recommendations for hallucinations
            recommendations.append("Request explicit sources or citations")
            recommendations.append("Cross-validate with retrieval-augmented system")
            
            # Specific recommendations based on issues
            if features.perturbation_sensitivity > 0.5:
                recommendations.append(
                    "Consider regenerating with lower temperature for more consistent output"
                )
            
            if invariant_result:
                grounding = invariant_result.results.get('epistemic_grounding')
                if grounding and grounding.violated:
                    recommendations.append(
                        "Verify all citations independently before using"
                    )
                
                semantic = invariant_result.results.get('semantic_conservation')
                if semantic and semantic.violated:
                    recommendations.append(
                        "Rephrase question and compare responses for consistency"
                    )
            
            if profile and profile.name in ['medical', 'legal']:
                recommendations.append(
                    f"Given {profile.name} context: consult authoritative sources before acting"
                )
        
        else:  # truthful
            if profile and profile.name in ['medical', 'legal', 'research']:
                recommendations.append(
                    "Standard verification still recommended for high-stakes applications"
                )
            
            if features.entropy_variance > 0.3:
                recommendations.append(
                    "Consider requesting more detailed explanation if topic is complex"
                )
        
        return recommendations


def format_console_report(report: DetectionReport) -> str:
    """Format report for console output"""
    lines = []
    
    # Header
    emoji = "🚨" if report.prediction == 'hallucination' else "✅"
    lines.append(f"\n{'='*70}")
    lines.append(f"{emoji} DETECTION REPORT")
    lines.append(f"{'='*70}")
    
    # Prediction
    lines.append(f"\nPrediction: {report.prediction.upper()}")
    lines.append(f"Confidence: {report.confidence:.1%}")
    lines.append(f"Risk Profile: {report.risk_profile}")
    if report.model_baseline:
        lines.append(f"Baseline: {report.model_baseline}")
    
    # Invariant results
    lines.append(f"\n{'-'*70}")
    lines.append("INVARIANT ANALYSIS")
    lines.append(f"{'-'*70}")
    lines.append(f"Invariants violated: {report.n_invariants_violated}/4")
    
    for name, result in report.invariant_results.items():
        status = "❌ VIOLATED" if result['violated'] else "✓ OK"
        lines.append(f"  {name}: {result['score']:.2f} {status}")
    
    # Temporal features
    lines.append(f"\n{'-'*70}")
    lines.append("KEY TEMPORAL FEATURES")
    lines.append(f"{'-'*70}")
    
    tf = report.temporal_features
    lines.append(f"  Tokens to high confidence: {tf.get('tokens_to_high_conf', 'N/A'):.1f}")
    lines.append(f"  Max confidence slope: {tf.get('max_confidence_slope', 'N/A'):.3f}")
    lines.append(f"  Final confidence: {tf.get('final_confidence', 'N/A'):.2f}")
    lines.append(f"  Entropy variance: {tf.get('entropy_variance', 'N/A'):.3f}")
    lines.append(f"  Perturbation sensitivity: {tf.get('perturbation_sensitivity', 'N/A'):.3f}")
    
    # Scores
    lines.append(f"\n  Temporal debt score: {report.temporal_debt_score:.3f}")
    if report.anomaly_score is not None:
        lines.append(f"  Baseline anomaly score: {report.anomaly_score:.3f}")
    
    # Explanation
    lines.append(f"\n{'-'*70}")
    lines.append("EXPLANATION")
    lines.append(f"{'-'*70}")
    lines.append(f"  {report.explanation}")
    
    # Key findings
    if report.key_findings:
        lines.append(f"\n  Key findings:")
        for finding in report.key_findings:
            lines.append(f"    • {finding}")
    
    # Recommendations
    if report.recommendations:
        lines.append(f"\n{'-'*70}")
        lines.append("RECOMMENDATIONS")
        lines.append(f"{'-'*70}")
        for rec in report.recommendations:
            lines.append(f"  → {rec}")
    
    # Footer
    lines.append(f"\n{'='*70}")
    lines.append(f"Report generated: {report.timestamp}")
    lines.append(f"{'='*70}\n")
    
    return '\n'.join(lines)


def format_json_report(report: DetectionReport, indent: int = 2) -> str:
    """Format report as JSON"""
    return report.to_json(indent=indent)


def format_markdown_report(report: DetectionReport) -> str:
    """Format report as Markdown"""
    lines = []
    
    emoji = "🚨" if report.prediction == 'hallucination' else "✅"
    
    lines.append(f"# {emoji} Hallucination Detection Report\n")
    lines.append(f"**Prediction:** {report.prediction.upper()}")
    lines.append(f"**Confidence:** {report.confidence:.1%}")
    lines.append(f"**Risk Profile:** {report.risk_profile}")
    lines.append(f"**Timestamp:** {report.timestamp}\n")
    
    lines.append("## Invariant Analysis\n")
    lines.append(f"Invariants violated: {report.n_invariants_violated}/4\n")
    lines.append("| Invariant | Score | Status |")
    lines.append("|-----------|-------|--------|")
    
    for name, result in report.invariant_results.items():
        status = "❌ Violated" if result['violated'] else "✓ OK"
        lines.append(f"| {name} | {result['score']:.2f} | {status} |")
    
    lines.append("\n## Explanation\n")
    lines.append(report.explanation + "\n")
    
    if report.key_findings:
        lines.append("### Key Findings\n")
        for finding in report.key_findings:
            lines.append(f"- {finding}")
    
    if report.recommendations:
        lines.append("\n### Recommendations\n")
        for rec in report.recommendations:
            lines.append(f"- {rec}")
    
    return '\n'.join(lines)
