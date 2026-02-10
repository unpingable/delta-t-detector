"""
Δt Hallucination Detector - Utility Functions
Helper functions for serialization, text processing, and validation
"""

import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache
import hashlib


def to_plain(o: Any) -> Any:
    """Convert numpy types to native Python for JSON serialization"""
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: to_plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_plain(item) for item in o]
    return o


def smooth_series(data: List[float], window: int = 3) -> np.ndarray:
    """Apply moving average smoothing to a series"""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window)/window, mode='valid')


def compute_entropy(probs: np.ndarray) -> float:
    """Compute Shannon entropy of a probability distribution"""
    probs = np.clip(probs, 1e-10, 1.0)
    return -np.sum(probs * np.log(probs))


def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text"""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def compute_text_hash(text: str) -> str:
    """Compute a hash for text (for caching)"""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# Citation extraction patterns
DOI_PATTERN = re.compile(
    r'\b(10\.\d{4,}/[^\s<>|]+)',
    re.IGNORECASE
)

URL_PATTERN = re.compile(
    r'https?://[^\s<>"{}|\\^`\[\]]+',
    re.IGNORECASE
)

ARXIV_PATTERN = re.compile(
    r'\barxiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)\b',
    re.IGNORECASE
)

PMID_PATTERN = re.compile(
    r'\bPMID:\s*(\d+)\b',
    re.IGNORECASE
)

ISBN_PATTERN = re.compile(
    r'\bISBN[-:]?\s*((?:\d[-\s]?){9,13}[\dXx])\b',
    re.IGNORECASE
)

RFC_PATTERN = re.compile(
    r'\bRFC\s*(\d{1,5})\b',
    re.IGNORECASE
)

PYPI_PATTERN = re.compile(
    r'\bpypi:\s*([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?==[0-9A-Za-z.+!_-]+)',
    re.IGNORECASE
)

PYPI_URL_PATTERN = re.compile(
    r'https?://pypi\.org/project/([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?)',
    re.IGNORECASE
)

CVE_PATTERN = re.compile(
    r'\b(CVE-\d{4}-\d{4,})\b',
    re.IGNORECASE
)


def extract_citations(text: str) -> Dict[str, List[str]]:
    """
    Extract citations from text

    Returns dict with keys: 'dois', 'urls', 'arxiv', 'pmids', 'isbns', 'rfcs',
    'pypi', 'cves'. Also includes 'pypi_urls' (metadata only, excluded from count).
    """
    # Strip trailing punctuation from DOIs and URLs
    dois = [d.rstrip('.,;:)]}') for d in DOI_PATTERN.findall(text)]
    urls = [u.rstrip('.,;:)]}') for u in URL_PATTERN.findall(text)]
    # Normalize CVE IDs to uppercase
    cves = [c.upper() for c in CVE_PATTERN.findall(text)]
    return {
        'dois': dois,
        'urls': urls,
        'arxiv': ARXIV_PATTERN.findall(text),
        'pmids': PMID_PATTERN.findall(text),
        'isbns': ISBN_PATTERN.findall(text),
        'rfcs': RFC_PATTERN.findall(text),
        'pypi': PYPI_PATTERN.findall(text),
        'cves': cves,
        'pypi_urls': PYPI_URL_PATTERN.findall(text),
    }


# Keys excluded from count_citations (metadata, not independent anchors)
_COUNT_EXCLUDE = frozenset({'pypi_urls'})


def count_citations(citations: Dict[str, List[str]]) -> int:
    """Count total citations extracted (excludes metadata-only keys)"""
    return sum(len(v) for k, v in citations.items() if k not in _COUNT_EXCLUDE)


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets"""
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity (1 - normalized distance)
    Uses dynamic programming for efficiency
    """
    if s1 == s2:
        return 1.0
    
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    
    # Create distance matrix
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost  # substitution
            )
    
    max_len = max(len1, len2)
    return 1.0 - (d[len1][len2] / max_len)


def tokenize_simple(text: str) -> List[str]:
    """Simple whitespace tokenization with normalization"""
    text = normalize_text(text)
    return text.split()


# Stopwords for keyword extraction
STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
    'could', 'may', 'might', 'must', 'can', 'of', 'at', 'by', 'for', 'with',
    'about', 'against', 'between', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
    'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    'there', 'when', 'where', 'why', 'how', 'all', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 'that', 'this', 'these', 
    'those', 'what', 'which', 'who', 'whom', 'it', 'its', 'i', 'you', 'he',
    'she', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
    'his', 'our', 'their', 'if', 'but', 'or', 'and', 'as', 'just', 'also'
})


def extract_keywords(text: str, min_length: int = 3) -> set:
    """Extract keywords from text (excluding stopwords)"""
    tokens = tokenize_simple(text)
    return {t for t in tokens if t not in STOPWORDS and len(t) >= min_length}


def phase_segment(
    sequence: List[float], 
    n_phases: int = 3
) -> Tuple[List[float], List[float], List[float]]:
    """
    Segment a sequence into phases (context, reasoning, answer)
    Returns (early, middle, late) phase values
    """
    if len(sequence) < n_phases:
        return list(sequence), [], []
    
    phase_size = len(sequence) // n_phases
    
    early = sequence[:phase_size]
    middle = sequence[phase_size:2*phase_size]
    late = sequence[2*phase_size:]
    
    return early, middle, late


def detect_phase_transition(
    sequence: List[float],
    threshold: float = 0.5
) -> Optional[int]:
    """
    Detect significant phase transitions in a sequence
    Returns index of first significant transition, or None
    """
    if len(sequence) < 3:
        return None
    
    diffs = np.diff(sequence)
    significant = np.where(np.abs(diffs) > threshold)[0]
    
    return int(significant[0]) if len(significant) > 0 else None


def confidence_from_entropy(entropy: float) -> float:
    """Convert entropy to confidence score: C(t) = 1 / (1 + H(t))"""
    return 1.0 / (1.0 + entropy)


def entropy_from_confidence(confidence: float) -> float:
    """Convert confidence to entropy: H(t) = (1 / C(t)) - 1"""
    if confidence <= 0:
        return float('inf')
    return (1.0 / confidence) - 1.0


class RollingStats:
    """Compute rolling statistics efficiently"""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> Dict[str, float]:
        """Add value and return current stats"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        
        arr = np.array(self.values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'variance': float(np.var(arr))
        }
    
    def reset(self):
        """Reset the rolling window"""
        self.values = []


def validate_features(features: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a features dict has all required fields
    Returns (is_valid, list of missing fields)
    """
    required = [
        'max_confidence_slope',
        'mean_confidence',
        'entropy_variance',
        'final_confidence'
    ]
    
    missing = [f for f in required if f not in features]
    return len(missing) == 0, missing
