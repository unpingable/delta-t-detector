#!/usr/bin/env python3
"""
3-Way Margin-Based Runtime Controller
======================================
Uses top-2 logit margin at identifier tokens to make per-prompt decisions:

  LOW_MARGIN_RETRY   — fork_risk < τ: model uncertain, retry at greedy
  KNOWLEDGE_BOUNDARY — LOW_MARGIN_RETRY where retry also FAILs: stop, can't fix ignorance
  CONFIDENT_WRONG    — fork_risk >= τ AND oracle FAIL: don't waste tokens
  FAST_PATH          — fork_risk >= τ AND oracle CLEAN/WARN: proceed

fork_risk = m_min from identifier windows (min margin across namespace triggers).
τ default = 0.05 (conservative; Phi-3 CVE M_median=0.0, Qwen-3B CVE=0.05).
Oracle = EG validator (authoritative API checks per namespace).
"""

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Module-level defaults (set from CLI args)
# ---------------------------------------------------------------------------

MAX_NEW_TOKENS = 250
DEFAULT_TEMPERATURE = 0.7
RETRY_TEMPERATURE = 0.0  # greedy retry
SEED_BASE = 42
TAU = 0.05

ROLE_SINGLE = """\
You are a factual research assistant. Answer the user's question accurately.
You MUST structure your response EXACTLY as follows:

ANSWER: <your answer in 1-3 sentences>
CITATIONS:
- <citation 1>
- <citation 2>
...
CLAIMS:
- <claim 1 supported by citation 1>
- <claim 2 supported by citation 2>
..."""

# ---------------------------------------------------------------------------
# Retry templates (per namespace) — from retry_enforcement.py
# ---------------------------------------------------------------------------

RETRY_TEMPLATES = {
    "pypi": """\
Your previous answer did not include references in the required format.

Re-output ONLY the following, exactly in this format:

ANSWER: <2-4 sentences answering the original question>

REFS:
1. pypi:name==version
2. pypi:name==version

CLAIMS:
1. <claim supported by REFS #1>
2. <claim supported by REFS #2>

Rules:
- REFS MUST contain exactly 2 anchors in pypi:name==version format.
- No URLs. No prose references. No "latest version". No extra items.
- If you cannot provide a valid package version, write "pypi:UNKNOWN==0.0.0" exactly.""",

    "cves": """\
Your previous answer did not include references in the required format.

Re-output ONLY the following, exactly in this format:

ANSWER: <2-4 sentences answering the original question>

REFS:
1. CVE-YYYY-NNNNN
2. CVE-YYYY-NNNNN

CLAIMS:
1. <claim supported by REFS #1>
2. <claim supported by REFS #2>

Rules:
- REFS MUST contain exactly 2 CVE identifiers in CVE-YYYY-NNNNN format.
- No URLs. No descriptions. No advisory links. No extra items.
- If you cannot provide a valid CVE, write "CVE-0000-0000" exactly.""",

    "dois": """\
Your previous answer did not include references in the required format.

Re-output ONLY the following, exactly in this format:

ANSWER: <2-4 sentences answering the original question>

REFS:
1. doi:10.XXXX/identifier
2. doi:10.XXXX/identifier

CLAIMS:
1. <claim supported by REFS #1>
2. <claim supported by REFS #2>

Rules:
- REFS MUST contain exactly 2 DOI identifiers.
- No URLs. No prose references. No extra items.
- If you cannot provide a valid DOI, write "doi:10.0000/unknown" exactly.""",

    "rfcs": """\
Your previous answer did not include references in the required format.

Re-output ONLY the following, exactly in this format:

ANSWER: <2-4 sentences answering the original question>

REFS:
1. RFC NNNN
2. RFC NNNN

CLAIMS:
1. <claim supported by REFS #1>
2. <claim supported by REFS #2>

Rules:
- REFS MUST contain exactly 2 RFC numbers in RFC NNNN format.
- No URLs. No prose references. No extra items.
- If you cannot provide a valid RFC, write "RFC 0000" exactly.""",
}

SENTINEL_CHECKS = {
    "pypi": lambda spec: "unknown" in spec.lower().split("==")[0],
    "cves": lambda cve_id: cve_id.upper() in ("CVE-0000-0000",),
    "dois": lambda doi: doi.strip().lower() in ("10.0000/unknown",),
    "rfcs": lambda rfc_num: rfc_num.strip() == "0000",
    "arxiv": lambda arxiv_id: arxiv_id.strip().startswith("0000."),
}

# ---------------------------------------------------------------------------
# Identifier window detection + margin analysis — from coupling.py
# ---------------------------------------------------------------------------

_ID_TRIGGERS = [
    re.compile(r'CVE-', re.IGNORECASE),
    re.compile(r'RFC\s*\d', re.IGNORECASE),
    re.compile(r'pypi:', re.IGNORECASE),
    re.compile(r'==\d'),                     # version pin
    re.compile(r'10\.\d{4,}/', re.IGNORECASE),  # DOI prefix
    re.compile(r'arxiv:', re.IGNORECASE),
    re.compile(r'\d{4}\.\d{4,}'),            # arXiv ID pattern
]

_ID_WINDOW = 16


def find_identifier_windows(tokens: List[str], window_size: int = _ID_WINDOW) -> List[Tuple[int, int]]:
    """Find (start, end) index pairs for identifier emission windows."""
    if not tokens:
        return []

    char_offsets = []
    for tok_idx, tok in enumerate(tokens):
        char_offsets.extend([tok_idx] * len(tok))

    full_text = "".join(tokens)
    windows = []
    seen_starts = set()

    for pat in _ID_TRIGGERS:
        for m in pat.finditer(full_text):
            char_start = m.start()
            if char_start >= len(char_offsets):
                continue
            tok_start = char_offsets[char_start]
            if tok_start in seen_starts:
                continue
            seen_starts.add(tok_start)
            tok_end = min(tok_start + window_size, len(tokens))
            windows.append((tok_start, tok_end))

    return sorted(windows)


def compute_margin_stats(margins: List[float], tokens: List[str]) -> dict:
    """Compute margin statistics over identifier emission windows."""
    if not margins:
        return {"m_min": None, "m_mean": None, "m_min_global": None,
                "n_id_windows": 0, "n_id_tokens": 0}

    m_min_global = min(margins)

    windows = find_identifier_windows(tokens)
    if not windows:
        return {"m_min": None, "m_mean": None, "m_min_global": round(m_min_global, 4),
                "n_id_windows": 0, "n_id_tokens": 0}

    id_margins = []
    for start, end in windows:
        for i in range(start, min(end, len(margins))):
            id_margins.append(margins[i])

    if not id_margins:
        return {"m_min": None, "m_mean": None, "m_min_global": round(m_min_global, 4),
                "n_id_windows": len(windows), "n_id_tokens": 0}

    return {
        "m_min": round(min(id_margins), 4),
        "m_mean": round(sum(id_margins) / len(id_margins), 4),
        "m_min_global": round(m_min_global, 4),
        "n_id_windows": len(windows),
        "n_id_tokens": len(id_margins),
    }


# ---------------------------------------------------------------------------
# Chat formatting + scoring — from retry_enforcement.py
# ---------------------------------------------------------------------------

def format_chat(tokenizer, system_prompt: str, user_content: str) -> str:
    """Format messages using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def format_chat_retry(
    tokenizer, system_prompt: str,
    user_content: str, assistant_response: str,
    retry_instruction: str,
) -> str:
    """Format multi-turn: system + user + assistant + retry user."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_response},
        {"role": "user", "content": retry_instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def score_attempt(
    detector,
    text: str,
    expected_min_anchors: Optional[int],
    expected_anchor_type: Optional[str] = None,
) -> dict:
    """Score a response using EG + aggregator. Returns full result dict."""
    eg_result = detector.epistemic_grounding_test.test(
        text, validate=True, check_relevance=True
    )
    multi_result = detector.multi_invariant_validator.aggregate(
        {"epistemic_grounding": eg_result},
        expected_min_anchors=expected_min_anchors,
        expected_anchor_type=expected_anchor_type,
    )
    return {
        "eg": eg_result.to_dict(),
        "verdict": multi_result.prediction,
        "confidence": multi_result.confidence,
        "fail_type": multi_result.fail_type,
        "fail_subjects": multi_result.fail_subjects,
        "warn_type": multi_result.warn_type,
    }


def count_sentinels(text: str, anchor_type: str) -> int:
    """Count how many extracted anchors of the given type are sentinels."""
    from detector.utils import extract_citations
    citations = extract_citations(text)
    check_fn = SENTINEL_CHECKS.get(anchor_type)
    if not check_fn:
        return 0
    anchors = citations.get(anchor_type, [])
    return sum(1 for a in anchors if check_fn(a))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _derive_seed(prompt_id: str, attempt: str) -> int:
    """Deterministic seed from prompt_id + attempt."""
    payload = f"{SEED_BASE}|{prompt_id}|controller|{attempt}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)


def _generate(detector, formatted_prompt: str, seed: int,
              temp_override: Optional[float] = None):
    """Generate with fixed seed and return (text, infer_ms, trace).

    Always returns the full trace (with margins) for controller decisions.
    temp_override: if provided, use this instead of DEFAULT_TEMPERATURE.
    """
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    temp = temp_override if temp_override is not None else DEFAULT_TEMPERATURE
    effective_temp = temp if temp > 0 else 0.01  # pseudo-greedy

    t0 = time.monotonic()
    trace = detector.generate_with_metrics(
        formatted_prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=effective_temp,
    )
    infer_ms = (time.monotonic() - t0) * 1000
    return trace.text, infer_ms, trace


# ---------------------------------------------------------------------------
# Verdict comparison
# ---------------------------------------------------------------------------

def _verdict_improved(v1: str, v2: str) -> bool:
    """Did the verdict improve from v1 to v2?

    FAIL → CLEAN/WARN = improved
    WARN → CLEAN = improved
    """
    if v1 == "FAIL" and v2 in ("CLEAN", "WARN"):
        return True
    if v1 == "WARN" and v2 == "CLEAN":
        return True
    return False


def _verdict_regressed(v1: str, v2: str) -> bool:
    """Did the verdict get worse from v1 to v2?"""
    if v1 == "CLEAN" and v2 in ("FAIL", "WARN"):
        return True
    if v1 == "WARN" and v2 == "FAIL":
        return True
    return False


# ---------------------------------------------------------------------------
# RFC section integrity: parse section headings, check section-level claims
# ---------------------------------------------------------------------------

_RFC_SECTION_PAT = re.compile(r'^(\d+(?:\.\d+)*)\.  (.+)', re.MULTILINE)
_RFC_SECTION_REF_PAT = re.compile(
    r'RFC\s*(\d{3,5})\s*(?:,?\s*)?(?:Section|§)\s*([\d.]+)',
    re.IGNORECASE,
)

# Two-layer cache: in-memory (session) + persistent file (.cache/rfc_sections/)
_rfc_section_cache: Dict[str, Optional[Dict[str, str]]] = {}
_RFC_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "rfc_sections"


def _parse_rfc_sections(text: str) -> Dict[str, str]:
    """Parse RFC plain text into {section_number: title} map.

    Only matches headings at column 0 (not indented list items).
    """
    return {m.group(1): m.group(2) for m in _RFC_SECTION_PAT.finditer(text)}


def _fetch_rfc_sections(rfc_number: str) -> Optional[Dict[str, str]]:
    """Fetch RFC text and return parsed section map.

    Two-layer cache:
    1. In-memory dict (fast, per-session)
    2. Persistent JSON file in .cache/rfc_sections/ (amortized across runs)
    Falls back to HTTP fetch if both miss.
    """
    # Layer 1: in-memory
    if rfc_number in _rfc_section_cache:
        return _rfc_section_cache[rfc_number]

    # Layer 2: persistent file cache
    cache_file = _RFC_CACHE_DIR / f"{rfc_number}.json"
    if cache_file.exists():
        try:
            sections = json.loads(cache_file.read_text())
            _rfc_section_cache[rfc_number] = sections
            return sections
        except Exception:
            pass  # corrupt cache file, re-fetch

    # Layer 3: HTTP fetch
    import requests as _requests
    try:
        resp = _requests.get(
            f"https://www.rfc-editor.org/rfc/rfc{rfc_number}.txt",
            headers={"User-Agent": _FETCH_UA},
            timeout=_FETCH_TIMEOUT,
        )
        if resp.status_code != 200:
            _rfc_section_cache[rfc_number] = None
            return None
        sections = _parse_rfc_sections(resp.text)
        _rfc_section_cache[rfc_number] = sections
        # Persist to file cache
        try:
            _RFC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(sections))
        except Exception:
            pass  # non-fatal: cache write failure doesn't block
        return sections
    except Exception:
        return None


def _extract_section_refs(text: str) -> List[Tuple[str, str]]:
    """Extract unique (rfc_number, section_number) pairs from model output.

    Matches patterns like:
        RFC 7540 Section 3.2
        RFC 7540, Section 3.2
        RFC7540 §3.2

    Deduplicates: model may repeat the same ref in REFS/CLAIMS blocks.
    """
    seen: set = set()
    result: list = []
    for m in _RFC_SECTION_REF_PAT.finditer(text):
        pair = (m.group(1), m.group(2).rstrip('.'))
        if pair not in seen:
            seen.add(pair)
            result.append(pair)
    return result


def check_rfc_section_integrity(
    text: str,
    prompt_text: str,
    relevance_fn,
    relevance_threshold: float = 0.15,
) -> Optional[dict]:
    """Check if RFC section references in model output match actual section content.

    Returns None if no section references found, otherwise:
        {
            "n_refs": int,
            "n_verified": int,       # section exists and title is relevant
            "n_wrong_section": int,   # section exists but title doesn't match claim
            "n_no_such_section": int, # section number doesn't exist in that RFC
            "n_fetch_failed": int,
            "refs": [...]
        }
    """
    section_refs = _extract_section_refs(text)
    if not section_refs:
        return None

    refs = []
    n_verified = 0
    n_wrong_section = 0
    n_no_such_section = 0
    n_fetch_failed = 0

    for rfc_num, sec_num in section_refs:
        sections = _fetch_rfc_sections(rfc_num)
        if sections is None:
            n_fetch_failed += 1
            refs.append({
                "rfc": rfc_num, "section": sec_num,
                "status": "fetch_failed", "actual_title": None,
                "relevance": None,
            })
            continue

        actual_title = sections.get(sec_num)
        if actual_title is None:
            n_no_such_section += 1
            refs.append({
                "rfc": rfc_num, "section": sec_num,
                "status": "no_such_section", "actual_title": None,
                "relevance": None,
                "available_sections": len(sections),
            })
            continue

        # Check if the model's surrounding text is relevant to the actual section title
        rel_response = relevance_fn(text, actual_title)
        rel_prompt = relevance_fn(prompt_text, actual_title)
        relevance = max(rel_response, rel_prompt)

        if relevance >= relevance_threshold:
            n_verified += 1
            status = "verified"
        else:
            n_wrong_section += 1
            status = "wrong_section"

        refs.append({
            "rfc": rfc_num, "section": sec_num,
            "status": status, "actual_title": actual_title,
            "relevance": round(relevance, 3),
        })

    return {
        "n_refs": len(section_refs),
        "n_verified": n_verified,
        "n_wrong_section": n_wrong_section,
        "n_no_such_section": n_no_such_section,
        "n_fetch_failed": n_fetch_failed,
        "refs": refs,
    }


# ---------------------------------------------------------------------------
# Grounding: fetch authoritative metadata and check relevance
# ---------------------------------------------------------------------------

# Authority tiers for fetch failure semantics.
#   "authoritative"  — canonical registry; 404 = identifier does not exist (MITRE, IETF, doi.org)
#   "conventional"   — de facto registry; 404 = very likely doesn't exist (PyPI JSON API)
#   "non_authoritative" — may fail for reasons unrelated to existence (arXiv, web URLs)
# 404 from authoritative/conventional → not_found (negative evidence).
# 404 from non_authoritative → fetch_failed (genuinely inconclusive).
_AUTHORITY_TIERS = {
    "cves": "authoritative",     # MITRE is the CVE Numbering Authority
    "rfcs": "authoritative",     # rfc-editor.org is the official IETF registry
    "dois": "authoritative",     # doi.org is the official DOI resolver
    "pypi": "conventional",      # PyPI JSON API is the de facto package registry
    "arxiv": "non_authoritative",  # arXiv API can fail transiently
}

# Namespaces where 404 is definitive (not_found, counted as negative evidence)
_DEFINITIVE_NAMESPACES = {
    ns for ns, tier in _AUTHORITY_TIERS.items()
    if tier in ("authoritative", "conventional")
}

# Groundable namespaces (have a fetch_evidence implementation)
_GROUNDABLE_NAMESPACES = {"cves", "rfcs", "dois", "arxiv"}

_FETCH_TIMEOUT = 10
_FETCH_UA = "DeltaT-Detector/1.0"


def _fetch_evidence(anchor_type: str, identifier: str) -> dict:
    """Fetch metadata from authoritative source with full HTTP evidence.

    Returns a dict with:
        metadata:       extracted text (title/description), or None
        http_status:    actual HTTP status code (int), or "timeout"/"error:ExcType"
        content_type:   response Content-Type header
        body_sniff:     sha256 of first 256 response bytes (detect soft 404s)
        source_url:     exact URL fetched
    """
    import requests as _requests

    urls_and_parsers = {
        "cves": (
            f"https://cveawg.mitre.org/api/cve/{identifier.upper()}",
            lambda data: _parse_cve_json(data),
        ),
        "rfcs": (
            f"https://www.rfc-editor.org/rfc/rfc{identifier}.json",
            lambda data: data.get("title") if isinstance(data, dict) else None,
        ),
        "dois": (
            f"https://api.crossref.org/works/{identifier}",
            lambda data: (data.get("message", {}).get("title", [None]) or [None])[0]
            if isinstance(data, dict) else None,
        ),
        "arxiv": (
            f"https://export.arxiv.org/api/query?id_list={identifier}",
            None,  # XML, handled separately
        ),
    }

    if anchor_type not in urls_and_parsers:
        return {"metadata": None, "http_status": "unsupported", "content_type": "",
                "body_sniff": "", "source_url": None}

    url, parser = urls_and_parsers[anchor_type]
    try:
        resp = _requests.get(url, headers={"User-Agent": _FETCH_UA},
                             timeout=_FETCH_TIMEOUT)
    except _requests.exceptions.Timeout:
        return {"metadata": None, "http_status": "timeout", "content_type": "",
                "body_sniff": "", "source_url": url}
    except Exception as exc:
        return {"metadata": None, "http_status": f"error:{type(exc).__name__}",
                "content_type": "", "body_sniff": "", "source_url": url}

    ct = resp.headers.get("Content-Type", "")
    sniff = hashlib.sha256(resp.content[:256]).hexdigest()[:16]
    base = {"http_status": resp.status_code, "content_type": ct,
            "body_sniff": sniff, "source_url": url}

    if resp.status_code != 200:
        return {**base, "metadata": None}

    # Parse metadata from successful response
    metadata = None
    if anchor_type == "arxiv":
        # XML: extract second <title> (first is feed title)
        matches = re.findall(r"<title[^>]*>(.+?)</title>", resp.text, re.DOTALL)
        if len(matches) >= 2:
            metadata = matches[1].strip()
    else:
        # JSON namespaces
        try:
            data = resp.json()
            metadata = parser(data)
        except Exception:
            pass  # 200 but unparseable → metadata stays None

    return {**base, "metadata": metadata}


def _parse_cve_json(data: dict) -> Optional[str]:
    """Extract English description from MITRE CVE JSON."""
    cna = data.get("containers", {}).get("cna", {})
    for desc in cna.get("descriptions", []):
        if desc.get("lang", "en").startswith("en"):
            return desc.get("value")
    descriptions = cna.get("descriptions", [])
    if descriptions:
        return descriptions[0].get("value")
    return None


def ground_anchors(
    detector,
    text: str,
    prompt_text: str,
    expected_anchor_type: Optional[str],
    relevance_threshold: float = 0.15,
) -> dict:
    """Attempt to ground extracted anchors against authoritative sources.

    Returns:
        {
            "groundable": bool,       # whether any groundable anchors found
            "n_grounded": int,        # anchors with fetched metadata
            "n_confirmed": int,       # metadata relevant to prompt
            "n_refuted": int,         # anchor exists but metadata irrelevant
            "n_not_found": int,       # authoritative API says identifier doesn't exist
            "n_failed_fetch": int,    # non-authoritative fetch failed (genuinely inconclusive)
            "verdict": str,           # "confirmed" | "refuted" | "inconclusive"
            "details": [...]          # per-anchor grounding results
        }
    """
    from detector.utils import extract_citations

    citations = extract_citations(text)
    eg_test = detector.epistemic_grounding_test

    details = []
    n_grounded = 0
    n_confirmed = 0
    n_refuted = 0
    n_not_found = 0
    n_failed = 0

    # Determine which anchor types to ground
    types_to_check = []
    if expected_anchor_type and expected_anchor_type in _GROUNDABLE_NAMESPACES:
        types_to_check = [expected_anchor_type]
    else:
        types_to_check = [t for t in _GROUNDABLE_NAMESPACES if citations.get(t)]

    if not types_to_check:
        return {"groundable": False, "n_grounded": 0, "n_confirmed": 0,
                "n_refuted": 0, "n_not_found": 0, "n_failed_fetch": 0,
                "verdict": "inconclusive", "details": []}

    for anchor_type in types_to_check:
        is_definitive = anchor_type in _DEFINITIVE_NAMESPACES
        authority_tier = _AUTHORITY_TIERS.get(anchor_type, "non_authoritative")
        anchors = list(dict.fromkeys(citations.get(anchor_type, [])))  # dedup
        for anchor in anchors:
            fetch_ts = datetime.now(timezone.utc).isoformat()
            ev = _fetch_evidence(anchor_type, anchor)
            metadata = ev["metadata"]
            http_status = ev["http_status"]

            # Classify the fetch result
            if metadata is None:
                # Distinguish definitive absence from transient failure.
                # For definitive namespaces: non-200 (including real 404, or 200
                # with unparseable body) = identifier doesn't exist.
                # For non-authoritative: any failure is genuinely inconclusive.
                # Also catch: 301/302 redirects, 429 rate-limit, 403/451 blocks.
                if isinstance(http_status, int) and http_status in (301, 302, 429, 403, 451):
                    # Ambiguous regardless of tier — redirect, rate-limit, or blocked
                    n_failed += 1
                    status = "fetch_failed"
                elif is_definitive:
                    n_not_found += 1
                    status = "not_found"
                else:
                    n_failed += 1
                    status = "fetch_failed"

                details.append({
                    "anchor": f"{anchor_type}:{anchor}",
                    "namespace": anchor_type,
                    "identifier": anchor,
                    "status": status,
                    "authority_tier": authority_tier,
                    "source_url": ev["source_url"],
                    "http_status": http_status,
                    "content_type": ev["content_type"],
                    "body_sniff": ev["body_sniff"],
                    "timestamp": fetch_ts,
                    "metadata": None,
                    "relevance": None,
                    "grounding": status,
                })
                continue

            n_grounded += 1
            # Check relevance: does the model's response + prompt match the metadata?
            # Use max of (response vs metadata, prompt vs metadata) since either
            # can carry the topical signal.
            rel_response = eg_test.check_citation_relevance(text, metadata)
            rel_prompt = eg_test.check_citation_relevance(prompt_text, metadata)
            relevance = max(rel_response, rel_prompt)
            if relevance >= relevance_threshold:
                n_confirmed += 1
                grounding = "confirmed"
            else:
                n_refuted += 1
                grounding = "refuted"

            details.append({
                "anchor": f"{anchor_type}:{anchor}",
                "namespace": anchor_type,
                "identifier": anchor,
                "status": grounding,
                "authority_tier": authority_tier,
                "source_url": ev["source_url"],
                "http_status": http_status,
                "content_type": ev["content_type"],
                "body_sniff": ev["body_sniff"],
                "timestamp": fetch_ts,
                "metadata": metadata[:200] if metadata else None,
                "relevance": round(relevance, 3),
                "grounding": grounding,
            })

    # Verdict logic: not_found from authoritative APIs counts as refuted
    n_negative = n_refuted + n_not_found  # both are evidence against
    n_evidence = n_grounded + n_not_found  # total anchors with a definitive answer

    if n_evidence == 0:
        verdict = "inconclusive"
    elif n_confirmed > 0 and n_negative == 0:
        verdict = "confirmed"
    elif n_negative > 0 and n_confirmed == 0:
        verdict = "refuted"
    elif n_confirmed >= n_negative:
        verdict = "mixed_lean_confirmed"
    else:
        verdict = "mixed_lean_refuted"

    # Section integrity check (RFC only — first namespace with section-level data)
    # If section integrity finds failures, demote the verdict.
    # "right book, wrong page" is not "confirmed".
    section_integrity = None
    if any(t == "rfcs" for t in types_to_check):
        section_integrity = check_rfc_section_integrity(
            text, prompt_text,
            relevance_fn=eg_test.check_citation_relevance,
            relevance_threshold=relevance_threshold,
        )
        if section_integrity is not None:
            n_bad = (section_integrity["n_wrong_section"]
                     + section_integrity["n_no_such_section"])
            if n_bad > 0 and verdict == "confirmed":
                # Existence confirmed but address fabricated → refuted
                verdict = "refuted"
            elif n_bad > 0 and verdict.startswith("mixed"):
                # Mixed existence + bad sections → lean refuted
                verdict = "mixed_lean_refuted"

    result = {
        "groundable": n_evidence > 0 or n_failed > 0,
        "n_grounded": n_grounded,
        "n_confirmed": n_confirmed,
        "n_refuted": n_refuted,
        "n_not_found": n_not_found,
        "n_failed_fetch": n_failed,
        "verdict": verdict,
        "details": details,
    }
    if section_integrity is not None:
        result["section_integrity"] = section_integrity
    return result


# ---------------------------------------------------------------------------
# Core: run one prompt through the 3-way controller
# ---------------------------------------------------------------------------

def run_prompt(
    detector,
    item,
    tau: float,
    retry_temp: float,
    run_id: str,
) -> tuple:
    """Run 3-way controller on a single prompt.

    Returns (record, final_result, final_text).
    """
    tok = detector.tokenizer
    prompt_id = item.id
    user_prompt = item.prompt
    expected_min = item.expected_min_anchors
    expected_type = item.expected_anchor_type

    # --- Step 1: Generate with trace at operational temp ---
    formatted = format_chat(tok, ROLE_SINGLE, user_prompt)
    seed1 = _derive_seed(prompt_id, "attempt1")
    text1, ms1, trace1 = _generate(detector, formatted, seed1)

    # --- Step 2: Compute fork risk ---
    margin_stats1 = compute_margin_stats(trace1.margins, trace1.tokens)
    fork_risk = margin_stats1["m_min"]
    n_id_windows = margin_stats1["n_id_windows"]

    # --- Step 3: Validate via oracle ---
    result1 = score_attempt(detector, text1, expected_min, expected_type)

    # --- Step 4: Apply policy ---
    policy = None
    result2 = None
    text2 = None
    ms2 = None
    margin_stats2 = None
    retry_gain = None
    retry_regression = None
    grounding = None

    # Edge case: no identifier windows → nothing to be uncertain about
    if n_id_windows == 0 or fork_risk is None:
        policy = "FAST_PATH"
        final_result = result1
        final_text = text1

    elif fork_risk < tau:
        # LOW_MARGIN: model uncertain at identifier tokens
        # Try GROUND first (fetch authoritative metadata, check relevance)
        # Fall back to RETRY if grounding is inconclusive

        grounding = ground_anchors(
            detector, text1, user_prompt, expected_type,
        )

        if grounding["groundable"] and grounding["verdict"] == "confirmed":
            if result1["verdict"] == "FAIL":
                # Safety: oracle found fabrication despite grounding confirmation.
                # Grounding may have only validated a subset of anchors.
                # Fall through to retry.
                pass
            else:
                # Grounding confirms: all anchors exist AND are relevant to prompt
                policy = "GROUNDED"
                final_result = result1
                final_text = text1

        elif grounding["groundable"] and grounding["verdict"] == "refuted":
            # Grounding refutes: anchors exist but none match the prompt
            # The model cited real identifiers for wrong claims
            # Don't retry (grounding already proved it wrong)
            policy = "GROUNDED_REFUTED"
            final_result = result1
            final_text = text1

        elif grounding["groundable"] and grounding["verdict"].startswith("mixed"):
            # Mixed: some anchors match, some don't — fall through to retry
            pass  # handled by the else branch below

        if policy is None:
            # Grounding inconclusive (fetch failed, no groundable anchors, etc.)
            # Fall back to greedy retry
            retry_template = RETRY_TEMPLATES.get(expected_type)
            if not retry_template:
                policy = "FAST_PATH"
                final_result = result1
                final_text = text1
            else:
                policy = "LOW_MARGIN_RETRY"
                formatted_retry = format_chat_retry(
                    tok, ROLE_SINGLE, user_prompt, text1, retry_template
                )
                seed2 = _derive_seed(prompt_id, "attempt2")
                text2, ms2, trace2 = _generate(
                    detector, formatted_retry, seed2, temp_override=retry_temp
                )
                margin_stats2 = compute_margin_stats(trace2.margins, trace2.tokens)
                result2 = score_attempt(detector, text2, expected_min, expected_type)

                retry_gain = _verdict_improved(result1["verdict"], result2["verdict"])
                retry_regression = _verdict_regressed(result1["verdict"], result2["verdict"])

                # Promote to KNOWLEDGE_BOUNDARY if both attempts FAIL
                if result1["verdict"] == "FAIL" and result2["verdict"] == "FAIL":
                    policy = "KNOWLEDGE_BOUNDARY"

                final_result = result2
                final_text = text2

    elif result1["verdict"] == "FAIL":
        # CONFIDENT_WRONG: high margin but oracle says fabricated
        policy = "CONFIDENT_WRONG"
        final_result = result1
        final_text = text1

    else:
        # FAST_PATH: high margin, oracle passes (CLEAN or WARN-evasion)
        policy = "FAST_PATH"
        final_result = result1
        final_text = text1

    # --- Post-policy: standalone section integrity gate ---
    # If RFC section refs are in the output and grounding didn't already run
    # section integrity, check now. Cached section maps make this cheap.
    section_integrity = None
    if grounding is None or "section_integrity" not in grounding:
        section_refs = _extract_section_refs(final_text)
        if section_refs:
            section_integrity = check_rfc_section_integrity(
                final_text, user_prompt,
                relevance_fn=detector.epistemic_grounding_test.check_citation_relevance,
            )
            if section_integrity is not None:
                n_bad = (section_integrity["n_wrong_section"]
                         + section_integrity["n_no_such_section"])
                if n_bad > 0 and final_result["verdict"] == "CLEAN":
                    # Address fabrication: existence passed, section is wrong.
                    # Override verdict to WARN — don't fast-path fabricated addresses.
                    final_result = dict(final_result)
                    final_result["verdict"] = "WARN"
                    final_result["warn_type"] = "SECTION_INTEGRITY_FAILURE"

    # --- Build record ---
    record = {
        "prompt_id": prompt_id,
        "fork_risk": fork_risk,
        "tau": tau,
        "n_id_windows": n_id_windows,
        "policy": policy,
        "attempt1": {
            "verdict": result1["verdict"],
            "fail_type": result1.get("fail_type"),
            "fail_subjects": result1.get("fail_subjects"),
            "warn_type": result1.get("warn_type"),
            "margin_stats": margin_stats1,
            "infer_ms": round(ms1, 1),
        },
        "final_verdict": final_result["verdict"],
    }

    if grounding is not None:
        record["grounding"] = {
            "verdict": grounding["verdict"],
            "n_grounded": grounding["n_grounded"],
            "n_confirmed": grounding["n_confirmed"],
            "n_refuted": grounding["n_refuted"],
            "n_not_found": grounding.get("n_not_found", 0),
            "details": grounding["details"],
        }
        if "section_integrity" in grounding:
            record["grounding"]["section_integrity"] = grounding["section_integrity"]

    if result2 is not None:
        record["attempt2"] = {
            "verdict": result2["verdict"],
            "fail_type": result2.get("fail_type"),
            "fail_subjects": result2.get("fail_subjects"),
            "warn_type": result2.get("warn_type"),
            "margin_stats": margin_stats2,
            "infer_ms": round(ms2, 1),
        }
        record["retry_gain"] = retry_gain
        record["retry_regression"] = retry_regression

    # Standalone section integrity (ran outside grounding)
    if section_integrity is not None:
        record["section_integrity"] = section_integrity

    return record, final_result, final_text


# ---------------------------------------------------------------------------
# Summary + printing
# ---------------------------------------------------------------------------

def print_summary(summary: dict):
    """Print human-readable summary."""
    print(f"\n{'='*60}")
    print("3-WAY CONTROLLER SUMMARY")
    print(f"{'='*60}")
    print(f"  Model:    {summary['model']}")
    print(f"  Corpus:   {summary['corpus']}")
    print(f"  Tau:      {summary['tau']}")
    print(f"  Temp:     {summary['temperature']} | Retry: {summary['retry_temperature']}")
    print(f"  Seed:     {summary['seed']}")
    print(f"  Prompts:  {summary['n_prompts']}")
    print()

    # Policy triggers
    print("Policy triggers:")
    n = summary["n_prompts"]
    for policy in ("GROUNDED", "GROUNDED_REFUTED", "LOW_MARGIN_RETRY",
                    "CONFIDENT_WRONG", "KNOWLEDGE_BOUNDARY", "FAST_PATH"):
        count = summary["policy_counts"].get(policy, 0)
        pct = count / n * 100 if n > 0 else 0
        print(f"  {policy:22s} {count:2d} ({pct:5.1f}%)")
    print()

    # Retry stats
    n_retries = summary["policy_counts"].get("LOW_MARGIN_RETRY", 0)
    if n_retries > 0:
        print("Retry outcomes:")
        print(f"  Retry gain (improved):    {summary['retry_gain_count']}/{n_retries}")
        print(f"  Retry regression (worse): {summary['retry_regression_count']}/{n_retries}")
        print(f"  Retry neutral (same):     {n_retries - summary['retry_gain_count'] - summary['retry_regression_count']}/{n_retries}")
        print()

    # Verdicts
    print("Final verdicts:")
    for v in ("CLEAN", "WARN", "FAIL"):
        count = summary["verdicts"].get(v, 0)
        print(f"  {v:5s} {count}")
    print()

    # Per-prompt detail
    print("Per-prompt breakdown:")
    for p in summary["per_prompt"]:
        fork_str = f"m_min={p['fork_risk']:.4f}" if p['fork_risk'] is not None else "m_min=N/A"
        line = f"  {p['prompt_id']:22s} {fork_str:16s} → {p['policy']:22s}"
        if p.get("retry_gain") is not None:
            v1 = p["attempt1_verdict"]
            v2 = p["final_verdict"]
            gain_str = "GAIN" if p["retry_gain"] else ("REGRESS" if p.get("retry_regression") else "SAME")
            line += f" [{v1}→{v2} {gain_str}]"
        else:
            line += f" [{p['final_verdict']}]"
        print(line)

    print(f"\n  Stored: {summary['run_dir']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="3-way margin-based runtime controller"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--corpus", required=True,
                        help="Path to locked corpus JSONL")
    parser.add_argument("--tau", type=float, default=0.05,
                        help="Fork risk threshold (default 0.05)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Operational temperature (default 0.7)")
    parser.add_argument("--retry-temperature", type=float, default=0.0,
                        help="Retry temperature (default 0.0 for greedy)")
    parser.add_argument("--quantize", choices=["4bit", "8bit"], default=None,
                        help="Quantization mode (requires bitsandbytes)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed (default 42)")
    args = parser.parse_args()

    # Set module-level defaults
    global DEFAULT_TEMPERATURE, RETRY_TEMPERATURE, SEED_BASE, TAU
    DEFAULT_TEMPERATURE = args.temperature if args.temperature > 0 else 0.01
    RETRY_TEMPERATURE = args.retry_temperature
    SEED_BASE = args.seed
    TAU = args.tau

    # Load model
    from detector.core import DeltaTDetector
    from detector.config import DetectorConfig
    from detector.eval import load_jsonl
    from detector.run_store import store_run, PredictionRecord, _utcnow_iso

    print(f"3-Way Controller | tau={TAU} | temp={args.temperature} | retry_temp={RETRY_TEMPERATURE}")
    print(f"Loading model: {args.model}")
    if args.quantize:
        print(f"Quantization: {args.quantize}")

    config = DetectorConfig()
    config.device = args.device
    if args.quantize:
        config.quantize = args.quantize
    detector = DeltaTDetector(model_name=args.model, config=config)
    detector.set_risk_profile("general")

    prompts = list(load_jsonl(args.corpus))
    print(f"Loaded {len(prompts)} prompts from {args.corpus}\n")

    timestamp = _utcnow_iso()
    run_id = f"controller_tau{TAU}_{timestamp}"
    started_at = _utcnow_iso()

    # Aggregates
    policy_counts = {"GROUNDED": 0, "GROUNDED_REFUTED": 0,
                      "LOW_MARGIN_RETRY": 0, "CONFIDENT_WRONG": 0,
                      "KNOWLEDGE_BOUNDARY": 0, "FAST_PATH": 0}
    verdict_counts: Dict[str, int] = {}
    retry_gain_count = 0
    retry_regression_count = 0

    records = []
    prediction_records = []

    for i, item in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {item.id}", end=" ", flush=True)

        record, final_result, final_text = run_prompt(
            detector, item, TAU, RETRY_TEMPERATURE, run_id
        )

        records.append(record)
        policy_counts[record["policy"]] += 1
        v = record["final_verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

        if record.get("retry_gain"):
            retry_gain_count += 1
        if record.get("retry_regression"):
            retry_regression_count += 1

        # Print inline status
        fork_str = f"m_min={record['fork_risk']:.4f}" if record['fork_risk'] is not None else "no_windows"
        print(f"→ {record['policy']} ({fork_str}) → {record['final_verdict']}")

        # Build PredictionRecord for store_run
        eg = final_result["eg"]
        prediction_records.append(PredictionRecord(
            id=item.id,
            prompt=item.prompt,
            expected_risk=item.expected_risk,
            prediction=final_result["verdict"],
            confidence=final_result["confidence"],
            temporal_debt=0.0,
            anomaly_score=None,
            features={},
            guard_flags={"low_evidence": False, "anomaly": False, "debt_cap": False},
            notes=f"controller|policy={record['policy']}|tau={TAU}|fork_risk={record['fork_risk']}",
            invariant_results={"epistemic_grounding": eg},
            n_violated=1 if eg.get("violated") else 0,
            aggregate_score=eg.get("score"),
            fail_type=final_result.get("fail_type"),
            fail_subjects=final_result.get("fail_subjects"),
            warn_type=final_result.get("warn_type"),
            expected_min_anchors=item.expected_min_anchors,
        ))

    # Store run bundle
    run_dir = store_run(
        predictions=prediction_records,
        corpus_path=args.corpus,
        model=args.model,
        profile=f"general+controller_tau{TAU}",
        device=args.device,
        baseline_used=False,
        started_at=started_at,
        multi_invariant=True,
    )

    # Write detailed records
    details_path = run_dir / "controller_details.jsonl"
    with open(details_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    print(f"\n  Stored: {run_dir}")
    print(f"  Details: {details_path}")

    # Build summary
    n = len(prompts)
    per_prompt = []
    for r in records:
        pp = {
            "prompt_id": r["prompt_id"],
            "fork_risk": r["fork_risk"],
            "policy": r["policy"],
            "attempt1_verdict": r["attempt1"]["verdict"],
            "final_verdict": r["final_verdict"],
        }
        if "retry_gain" in r:
            pp["retry_gain"] = r["retry_gain"]
            pp["retry_regression"] = r.get("retry_regression", False)
        per_prompt.append(pp)

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "corpus": args.corpus,
        "model": args.model,
        "tau": TAU,
        "temperature": args.temperature,
        "retry_temperature": RETRY_TEMPERATURE,
        "seed": SEED_BASE,
        "seed_derivation": "sha256(SEED_BASE|prompt_id|controller|attempt)",
        "n_prompts": n,
        "policy_counts": policy_counts,
        "retry_gain_count": retry_gain_count,
        "retry_regression_count": retry_regression_count,
        "verdicts": verdict_counts,
        "per_prompt": per_prompt,
    }

    print_summary(summary)

    # Write summary JSON
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / f"controller_tau{TAU}_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Report: {report_path}")


if __name__ == "__main__":
    main()
