"""
Δt Hallucination Detector - Invariant Tests Module
Implementations of all four hallucination invariants
"""

import re
import asyncio
try:
    import aiohttp
except ImportError:
    aiohttp = None
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from .utils import (
    extract_citations, count_citations, levenshtein_similarity,
    jaccard_similarity, tokenize_simple
)


@dataclass
class InvariantResult:
    """Result from a single invariant test"""
    name: str
    score: float           # 0.0 = strong violation, 1.0 = fully satisfied
    violated: bool
    details: Dict[str, Any]
    confidence: float = 1.0  # Confidence in this result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'score': self.score,
            'violated': self.violated,
            'details': self.details,
            'confidence': self.confidence
        }


@dataclass
class MultiInvariantResult:
    """Aggregated result from all invariant tests"""
    results: Dict[str, InvariantResult]
    n_violated: int
    aggregate_score: float
    prediction: str  # 'FAIL', 'WARN', or 'CLEAN'
    confidence: float
    fail_type: Optional[str] = None       # 'FABRICATED_IDENTIFIER' or None
    fail_subjects: Optional[List[str]] = None  # e.g. ['doi:10.xxxx', 'arxiv:2501.00000']
    warn_type: Optional[str] = None      # 'NEED_EVIDENCE' or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'results': {k: v.to_dict() for k, v in self.results.items()},
            'n_violated': self.n_violated,
            'aggregate_score': self.aggregate_score,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'fail_type': self.fail_type,
            'fail_subjects': self.fail_subjects,
        }


# ============================================================================
# Invariant 1: Temporal Coherence
# ============================================================================

class TemporalCoherenceTest:
    """
    Tests temporal coherence (Invariant 1)
    
    Checks if dC/dt (belief change rate) exceeds dE/dt (evidence accumulation rate)
    Uses phase-reversal signatures as the primary detection method
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        tokens_to_conf_threshold: int = 8,
        acceleration_threshold: float = 0.15,
        zscore_threshold: float = 2.0
    ):
        self.threshold = threshold
        self.tokens_to_conf_threshold = tokens_to_conf_threshold
        self.acceleration_threshold = acceleration_threshold
        self.zscore_threshold = zscore_threshold
    
    def test(
        self,
        features: Dict[str, Any],
        baseline_normalized: Optional[Dict[str, Any]] = None
    ) -> InvariantResult:
        """
        Test temporal coherence from extracted features
        
        Args:
            features: Dictionary of temporal features
            baseline_normalized: Optional baseline-normalized features
            
        Returns:
            InvariantResult
        """
        # Use normalized features if available
        feat = baseline_normalized or features
        
        # Phase-reversal detection
        tokens_to_conf = feat.get('tokens_to_high_conf', 10)
        max_slope = feat.get('max_confidence_slope', 0)
        acceleration = feat.get('confidence_acceleration', 0)
        
        # Compute violation indicators
        violations = []

        if self.tokens_to_conf_threshold > 0 and tokens_to_conf < self.tokens_to_conf_threshold:
            violations.append(f"High confidence reached too quickly ({tokens_to_conf:.1f} tokens)")

        if max_slope > self.threshold:
            violations.append(f"Confidence slope too steep ({max_slope:.3f})")

        if acceleration > self.acceleration_threshold:
            violations.append(f"Confidence acceleration too high ({acceleration:.3f})")

        # Baseline anomaly checks (z-scores)
        if baseline_normalized:
            for key, label in [
                ('tokens_to_high_conf_zscore', 'tokens_to_high_conf'),
                ('max_confidence_slope_zscore', 'max_confidence_slope'),
                ('confidence_acceleration_zscore', 'confidence_acceleration')
            ]:
                z = baseline_normalized.get(key)
                if isinstance(z, (int, float)) and abs(z) > self.zscore_threshold:
                    violations.append(
                        f"{label} anomalous vs baseline (z={z:.2f})"
                    )

        # Compute score (higher = more coherent = good)
        # Slope score: lower slope is better
        slope_score = max(0.0, 1.0 - max_slope / 1.0)

        # Acceleration score: lower is better (normalize against threshold)
        accel_norm = self.acceleration_threshold if self.acceleration_threshold > 0 else 0.5
        accel_score = max(0.0, 1.0 - acceleration / (2.0 * accel_norm))

        # Combine scores
        score = 0.6 * slope_score + 0.4 * accel_score
        score = max(0.0, min(1.0, score))

        violated = len(violations) > 0
        
        return InvariantResult(
            name='temporal_coherence',
            score=score,
            violated=violated,
            details={
                'tokens_to_high_conf': tokens_to_conf,
                'max_confidence_slope': max_slope,
                'confidence_acceleration': acceleration,
                'violations': violations,
                'thresholds': {
                    'tokens': self.tokens_to_conf_threshold,
                    'slope': self.threshold,
                    'acceleration': self.acceleration_threshold
                }
            }
        )


# ============================================================================
# Invariant 2: Semantic Conservation
# ============================================================================

class SemanticConservationTest:
    """
    Tests semantic conservation (Invariant 2)
    
    Rephrases the question and checks if responses remain semantically similar
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        n_variants: int = 3,
        use_embeddings: bool = True
    ):
        self.similarity_threshold = similarity_threshold
        self.n_variants = n_variants
        self.use_embeddings = use_embeddings
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load sentence transformer model"""
        if self._embedding_model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                self.use_embeddings = False
                print("Warning: sentence-transformers not installed, falling back to token similarity")
        return self._embedding_model
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if self.use_embeddings:
            model = self._get_embedding_model()
            if model is not None:
                embeddings = model.encode([text1, text2])
                # Cosine similarity
                sim = np.dot(embeddings[0], embeddings[1])
                sim /= (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-10)
                return float(sim)
        
        # Fallback to token-based similarity
        tokens1 = set(tokenize_simple(text1))
        tokens2 = set(tokenize_simple(text2))
        return jaccard_similarity(tokens1, tokens2)
    
    def test(
        self,
        responses: List[str],
        query_variants: Optional[List[str]] = None
    ) -> InvariantResult:
        """
        Test semantic conservation across response variants
        
        Args:
            responses: List of responses to different phrasings
            query_variants: Optional list of query variants (for logging)
            
        Returns:
            InvariantResult
        """
        if len(responses) < 2:
            return InvariantResult(
                name='semantic_conservation',
                score=1.0,
                violated=False,
                details={'error': 'Insufficient responses for comparison'},
                confidence=0.5
            )
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                sim = self.compute_similarity(responses[i], responses[j])
                similarities.append({
                    'pair': (i, j),
                    'similarity': sim
                })
        
        # Average similarity
        avg_similarity = np.mean([s['similarity'] for s in similarities])
        min_similarity = np.min([s['similarity'] for s in similarities])
        
        # Semantic drift detection
        drifting_pairs = [s for s in similarities if s['similarity'] < self.similarity_threshold]
        
        violated = len(drifting_pairs) > 0
        
        return InvariantResult(
            name='semantic_conservation',
            score=float(avg_similarity),
            violated=violated,
            details={
                'avg_similarity': float(avg_similarity),
                'min_similarity': float(min_similarity),
                'n_comparisons': len(similarities),
                'drifting_pairs': drifting_pairs,
                'threshold': self.similarity_threshold,
                'query_variants': query_variants
            }
        )
    
    def generate_variants(self, query: str) -> List[str]:
        """
        Generate query variants for testing
        
        Note: In production, this would use the LLM to rephrase
        For now, uses simple transformations
        """
        variants = [query]
        
        # Simple transformations
        # 1. Question form
        if not query.endswith('?'):
            variants.append(query + '?')
        
        # 2. Add "please"
        if 'please' not in query.lower():
            variants.append(f"Please tell me: {query}")
        
        # 3. Rephrase as request
        variants.append(f"I would like to know: {query}")
        
        return variants[:self.n_variants]


# ============================================================================
# Invariant 3: Epistemic Grounding
# ============================================================================

class EpistemicGroundingTest:
    """
    Tests epistemic grounding (Invariant 3)

    Validates citations and sources mentioned in responses via existence
    checks: HEAD requests for URLs, doi.org for DOIs, rfc-editor.org
    for RFCs, arxiv.org for arXiv IDs.  No semantic fetching — just
    "does the cited anchor exist?"
    """

    _USER_AGENT = 'DeltaT-Detector/1.0'

    def __init__(
        self,
        max_fabricated: int = 0,
        validate_urls: bool = True,
        timeout: int = 5
    ):
        self.max_fabricated = max_fabricated
        self.validate_urls_enabled = validate_urls
        self.timeout = timeout

    # ------------------------------------------------------------------
    # URL validation (aiohttp fast-path, requests fallback)
    # ------------------------------------------------------------------

    def _head_requests(self, url: str) -> Tuple[bool, str]:
        """Validate a single URL via requests (sync, HEAD only)."""
        try:
            import requests
            resp = requests.head(
                url, allow_redirects=True, timeout=self.timeout,
                headers={'User-Agent': self._USER_AGENT},
            )
            valid = resp.status_code in (200, 301, 302, 403, 401, 405)
            return valid, f"HTTP {resp.status_code}"
        except Exception as e:
            return False, str(e)[:80]

    async def _validate_url_async(self, url: str, session: Any) -> Tuple[str, bool, str]:
        """Validate a single URL asynchronously"""
        try:
            async with session.head(
                url,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                allow_redirects=True
            ) as response:
                valid = response.status in [200, 301, 302, 403, 401, 405]
                return url, valid, f"HTTP {response.status}"
        except asyncio.TimeoutError:
            return url, False, "timeout"
        except Exception as e:
            return url, False, str(e)[:50]

    async def _validate_urls_async(self, urls: List[str]) -> Dict[str, Tuple[bool, str]]:
        """Validate multiple URLs concurrently"""
        results = {}
        async with aiohttp.ClientSession(
            headers={'User-Agent': self._USER_AGENT}
        ) as session:
            tasks = [self._validate_url_async(url, session) for url in urls]
            completed = await asyncio.gather(*tasks, return_exceptions=True)
            for result in completed:
                if isinstance(result, Exception):
                    continue
                url, valid, reason = result
                results[url] = (valid, reason)
        return results

    def validate_urls(self, urls: List[str]) -> Dict[str, Tuple[bool, str]]:
        """Validate URLs — async if aiohttp available, sync fallback."""
        if not urls:
            return {}

        # Try aiohttp fast-path
        if aiohttp is not None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._validate_urls_async(urls))

        # Fallback: synchronous requests
        results = {}
        for url in urls:
            valid, reason = self._head_requests(url)
            results[url] = (valid, reason)
        return results

    # ------------------------------------------------------------------
    # Format validators (local, no network)
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_doi_format(doi: str) -> bool:
        """Check DOI format locally before network lookup."""
        # Must start with 10., have a registrant code, and a suffix
        # Reject repeated segments, too-short suffixes, control chars
        if not re.match(r'^10\.\d{4,}/.{2,}$', doi):
            return False
        # Reject obvious repetition (e.g. 10.5592/10.5592/10.5592...)
        parts = doi.split('/')
        if len(parts) > 3 and len(set(parts[1:])) == 1:
            return False
        return True

    @staticmethod
    def _valid_arxiv_format(arxiv_id: str) -> bool:
        """Check arXiv ID format: YYMM.NNNNN(vN)."""
        return bool(re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id))

    # ------------------------------------------------------------------
    # Specific anchor validators
    # ------------------------------------------------------------------

    def validate_doi(self, doi: str) -> Tuple[bool, str]:
        """Validate a DOI — format check first, then network."""
        if not self._valid_doi_format(doi):
            return False, "invalid DOI format"
        url = f"https://doi.org/{doi}"
        results = self.validate_urls([url])
        return results.get(url, (False, "not checked"))

    def validate_rfc(self, rfc_number: str) -> Tuple[bool, str]:
        """Validate an RFC by checking rfc-editor.org"""
        url = f"https://www.rfc-editor.org/rfc/rfc{rfc_number}"
        valid, reason = self._head_requests(url)
        return valid, reason

    def validate_arxiv(self, arxiv_id: str) -> Tuple[bool, str]:
        """Validate an arXiv paper — format check first, then network."""
        if not self._valid_arxiv_format(arxiv_id):
            return False, "invalid arXiv format"
        url = f"https://arxiv.org/abs/{arxiv_id}"
        valid, reason = self._head_requests(url)
        return valid, reason

    # ------------------------------------------------------------------
    # Main test
    # ------------------------------------------------------------------

    def test(
        self,
        text: str,
        validate: bool = True
    ) -> InvariantResult:
        """
        Test epistemic grounding of a response

        Args:
            text: Response text to analyze
            validate: Whether to actually validate citations

        Returns:
            InvariantResult
        """
        citations = extract_citations(text)
        total_citations = count_citations(citations)

        if total_citations == 0:
            return InvariantResult(
                name='epistemic_grounding',
                score=0.5,
                violated=False,
                details={
                    'total_citations': 0,
                    'note': 'No citations found to validate'
                },
                confidence=0.5
            )

        valid_count = 0
        invalid_count = 0
        format_invalid = 0
        resolve_invalid = 0
        validation_results = {}

        _FORMAT_REASONS = {"invalid DOI format", "invalid arXiv format"}

        def _response_class(reason: str) -> str:
            if reason in _FORMAT_REASONS:
                return "format_rejected"
            if "HTTP 200" in reason or "HTTP 301" in reason or "HTTP 302" in reason:
                return "found"
            if "HTTP 404" in reason:
                return "not_found"
            if "HTTP 403" in reason or "HTTP 401" in reason:
                return "found"  # exists but access restricted
            if "timeout" in reason.lower():
                return "timeout"
            if "HTTP" in reason:
                return "not_found"
            return "error"

        def _record(key, valid, reason, resolver):
            nonlocal valid_count, invalid_count, format_invalid, resolve_invalid
            from datetime import datetime, timezone
            validation_results[key] = {
                'valid': valid,
                'reason': reason,
                'resolver': resolver,
                'response_class': _response_class(reason),
                'timestamp': datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            }
            if valid:
                valid_count += 1
            else:
                invalid_count += 1
                if reason in _FORMAT_REASONS:
                    format_invalid += 1
                else:
                    resolve_invalid += 1

        if validate and self.validate_urls_enabled:
            # URLs (no format check — always resolve)
            if citations['urls']:
                url_results = self.validate_urls(citations['urls'])
                for url, (valid, reason) in url_results.items():
                    _record(url, valid, reason, "HEAD")

            # DOIs
            for doi in citations['dois']:
                valid, reason = self.validate_doi(doi)
                _record(f"doi:{doi}", valid, reason, "doi.org")

            # RFCs
            for rfc in citations.get('rfcs', []):
                valid, reason = self.validate_rfc(rfc)
                _record(f"rfc:{rfc}", valid, reason, "rfc-editor.org")

            # arXiv
            for arxiv_id in citations.get('arxiv', []):
                valid, reason = self.validate_arxiv(arxiv_id)
                _record(f"arxiv:{arxiv_id}", valid, reason, "arxiv.org")
        else:
            valid_count = total_citations

        # Remaining unvalidated (PMIDs, ISBNs — harder to check)
        unvalidated = len(citations['pmids']) + len(citations['isbns'])

        # Grounding score
        if valid_count + invalid_count > 0:
            grounding_score = valid_count / (valid_count + invalid_count)
        else:
            grounding_score = 1.0 if unvalidated == 0 else 0.5

        violated = invalid_count > self.max_fabricated

        return InvariantResult(
            name='epistemic_grounding',
            score=grounding_score,
            violated=violated,
            details={
                'total_citations': total_citations,
                'valid_count': valid_count,
                'invalid_count': invalid_count,
                'format_invalid': format_invalid,
                'resolve_invalid': resolve_invalid,
                'unvalidated_count': unvalidated,
                'citations_found': {k: len(v) for k, v in citations.items()},
                'validation_results': validation_results,
                'max_fabricated_allowed': self.max_fabricated
            }
        )


# ============================================================================
# Invariant 4: Irreversibility
# ============================================================================

class IrreversibilityTest:
    """
    Tests irreversibility (Invariant 4)
    
    Checks that the same prompt produces similar responses (no learning residue)
    For current LLMs, high similarity is expected (they don't learn between calls)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.7,
        n_trials: int = 2
    ):
        self.similarity_threshold = similarity_threshold
        self.n_trials = n_trials
    
    def test(
        self,
        responses: List[str]
    ) -> InvariantResult:
        """
        Test irreversibility across repeated generations
        
        Args:
            responses: List of responses to the same prompt
            
        Returns:
            InvariantResult
        """
        if len(responses) < 2:
            return InvariantResult(
                name='irreversibility',
                score=1.0,
                violated=False,
                details={'error': 'Insufficient responses'},
                confidence=0.5
            )
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                # Use Levenshtein similarity
                sim = levenshtein_similarity(responses[i], responses[j])
                
                # Also compute token-level
                tokens_i = set(tokenize_simple(responses[i]))
                tokens_j = set(tokenize_simple(responses[j]))
                token_sim = jaccard_similarity(tokens_i, tokens_j)
                
                similarities.append({
                    'pair': (i, j),
                    'levenshtein': sim,
                    'token_jaccard': token_sim
                })
        
        avg_lev = np.mean([s['levenshtein'] for s in similarities])
        avg_token = np.mean([s['token_jaccard'] for s in similarities])
        
        # Combined similarity score
        combined_score = 0.6 * avg_lev + 0.4 * avg_token
        
        # For LLMs, we expect high similarity (no memory between calls)
        # Violation would be very LOW similarity (suggesting something weird)
        # or very HIGH similarity with suspicious patterns
        
        violated = combined_score < self.similarity_threshold
        
        return InvariantResult(
            name='irreversibility',
            score=float(combined_score),
            violated=violated,
            details={
                'avg_levenshtein_similarity': float(avg_lev),
                'avg_token_similarity': float(avg_token),
                'n_trials': len(responses),
                'pairwise_similarities': similarities,
                'threshold': self.similarity_threshold,
                'note': 'High similarity expected for stateless LLMs'
            }
        )


# ============================================================================
# Multi-Invariant Aggregation
# ============================================================================

class MultiInvariantValidator:
    """
    Aggregates results from all invariant tests
    """
    
    def __init__(
        self,
        min_invariants_required: int = 2,
        weights: Optional[Dict[str, float]] = None
    ):
        self.min_invariants_required = min_invariants_required
        self.weights = weights or {
            'temporal_coherence': 1.0,
            'semantic_conservation': 0.8,
            'epistemic_grounding': 1.0,
            'irreversibility': 0.5
        }
    
    def aggregate(
        self,
        results: Dict[str, InvariantResult],
        expected_min_anchors: Optional[int] = None,
    ) -> MultiInvariantResult:
        """
        Aggregate invariant results into final prediction.

        Decision rule:
          - EG (epistemic_grounding) violation with invalid anchors → FAIL
            (fail_type=FABRICATED_IDENTIFIER)
          - anchors_found < expected_min_anchors → WARN (warn_type=NEED_EVIDENCE)
          - SC/TC violations without EG → WARN only (demoted from gating)
          - No violations → CLEAN

        SC and TC are recorded as telemetry but cannot produce FAIL alone.
        """
        n_violated = sum(1 for r in results.values() if r.violated)

        # Weighted score (telemetry, not gating)
        weighted_sum = 0.0
        total_weight = 0.0
        for name, result in results.items():
            weight = self.weights.get(name, 1.0)
            weighted_sum += result.score * weight * result.confidence
            total_weight += weight * result.confidence
        aggregate_score = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Check EG for fabricated identifiers
        eg = results.get('epistemic_grounding')
        eg_violated = eg is not None and eg.violated
        fail_type = None
        fail_subjects: List[str] = []

        if eg_violated:
            # Extract the invalid anchors from EG details
            vr = eg.details.get('validation_results', {})
            for anchor, detail in vr.items():
                if isinstance(detail, dict) and not detail.get('valid', True):
                    fail_subjects.append(anchor)
            if fail_subjects:
                fail_type = 'FABRICATED_IDENTIFIER'

        # Check evasion: anchors below expected minimum
        warn_type = None
        anchors_found = 0
        if eg is not None:
            anchors_found = eg.details.get('total_citations', 0)

        # Decision rule
        if eg_violated:
            prediction = 'FAIL'
            confidence = min(0.95, 0.6 + 0.1 * len(fail_subjects))
        elif expected_min_anchors is not None and anchors_found < expected_min_anchors:
            prediction = 'WARN'
            warn_type = 'NEED_EVIDENCE'
            confidence = 0.6
        elif n_violated > 0:
            prediction = 'WARN'
            confidence = min(0.85, 0.5 + 0.1 * n_violated)
        else:
            prediction = 'CLEAN'
            confidence = aggregate_score

        return MultiInvariantResult(
            results=results,
            n_violated=n_violated,
            aggregate_score=aggregate_score,
            prediction=prediction,
            confidence=confidence,
            fail_type=fail_type,
            fail_subjects=fail_subjects if fail_subjects else None,
            warn_type=warn_type,
        )
