# Δt Hallucination Detector

Detect hallucinations by measuring **temporal coherence violations** — when belief change rate (dC/dt) exceeds evidence accumulation rate (dE/dt).

## Installation

```bash
git clone https://github.com/unpingable/delta-t-detector.git
cd delta-t-detector
pip install -e .
```

For all features:
```bash
pip install -e ".[full]"
```

Optional dependencies:
- `torch` (required for local model runs)
- `aiohttp` (required for URL validation in grounding checks)
- `sentence-transformers` (semantic conservation embeddings)

## Quick Start

### Python

```python
from detector import DeltaTDetector

detector = DeltaTDetector()  # Auto-selects CUDA > MPS > CPU
result = detector.detect("What is the capital of France?")

print(result.prediction)    # 'truthful' or 'hallucination'
print(result.confidence)    # 0.0 - 1.0
print(result.temporal_debt) # Higher = more suspicious
```

### Command Line

```bash
# Check devices
python -m detector.run devices

# Detect
python -m detector.run detect --prompt "What causes earthquakes?"

# With risk profile
python -m detector.run detect --prompt "What treats hypertension?" --profile medical

# Run tests
python -m detector.run test
```

## Risk Profiles

```python
detector.set_risk_profile('medical')      # Very conservative
detector.set_risk_profile('legal')        # Conservative  
detector.set_risk_profile('general')      # Balanced (default)
detector.set_risk_profile('creative')     # Permissive
```

## The Four Invariants

This detector is a reference implementation of **Invariant 1 (Temporal Coherence)** from the companion paper:

> Beck, J. (2025). "You Need More Than Just Attention: Invariant Requirements for Temporal Coherence in AI Systems."

| Invariant | What It Tests | This Implementation |
|-----------|---------------|---------------------|
| **Temporal Coherence** | Does confidence outpace evidence? | ✓ Full |
| **Semantic Conservation** | Stable meaning across rephrasings? | ⚠ Approximated |
| **Epistemic Grounding** | Do sources constrain claims? | ⚠ Approximated |
| **Irreversibility** | Do errors leave residue? | ⚠ Approximated |

All detector features are computed from per-query traces; no statistics persist across queries; each detect() call resets all rolling state.

Evidence accumulation is operationalized via proxy signals (token progression, entropy stabilization, and confidence trajectory), not assumed to correspond directly to explicit reasoning tokens.

Epistemic Grounding tests detect citation fabrication and topical drift relative to known reference material; they do not certify factual correctness.

All temporal statistics are computed per-query. No detector state persists across calls unless explicitly configured.

## Multi-Invariant Detection

```python
result = detector.detect_multi_invariant(
    "Explain quantum entanglement",
    validate_citations=True,
    test_semantic=True
)
```

## Structured Signal Schema

For governor-style integration, a stable JSON signal payload is available:

```python
from detector.reporting import build_signal, format_signal_json

result = detector.detect("What is the capital of France?")
signal = build_signal(result.report)
signal_json = format_signal_json(result.report)
```

The signal schema includes:
- `schema_version`
- `prediction`, `confidence`, `temporal_debt`
- `temporal_debt_components` and `temporal_debt_weights`
- `signals` (temporal feature vector)
- `invariants` (per-invariant scores/violations)
- `provenance` (generation hash, length, phase stats)
- JSON Schema: `schema/signal.schema.json`

## Project Structure

```
delta-t-detector/
├── detector/
│   ├── __init__.py      # Package exports
│   ├── core.py          # DeltaTDetector main class
│   ├── config.py        # Risk profiles
│   ├── features.py      # Temporal feature extraction
│   ├── invariants.py    # Four invariant tests
│   ├── baseline.py      # Model calibration
│   ├── reporting.py     # Output formatting
│   ├── api_providers.py # OpenAI/Anthropic/Ollama
│   ├── run.py           # CLI
│   ├── utils.py         # Helpers
│   └── tests.py         # Test suite
├── pyproject.toml
├── LICENSE
└── README.md
```

## License

Apache 2.0

## Links

- [Paper: You Need More Than Just Attention (Zenodo)](https://zenodo.org/records/18039927)
- [Substack: The Neutral Zone](https://neutralzone.substack.com/)
