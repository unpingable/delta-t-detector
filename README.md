# Delta-T Detector

**An evaluation harness for LLM reliability under constraint.**

It focuses on failure modes that can be *falsified mechanically*: fabricated identifiers (DOI/arXiv/etc.), mismatched citations (real IDs pointing to the wrong work), and evasion when anchors are required.

Each run produces an auditable bundle (flight recorder + per-step artifacts + resolver provenance) so results can be diffed over time and across platforms.

**Origin:** this repo began as a Δt/commitment-dynamics exploration (see [Zenodo paper](https://zenodo.org/records/18039927)). The current codebase generalizes that work into practical detection, gating, and multi-agent topology experiments.

## What it detects

- **`FABRICATED_IDENTIFIER`** — plausible-looking DOI/arXiv/RFC/PMID that doesn't resolve
- **`MISMATCHED_CITATION`** — real identifier, wrong paper / doesn't support the claim
- **`NEED_EVIDENCE`** — evasion / insufficient anchors under expected constraints

## Why this exists

LLMs can produce *plausible lies* that are syntactically valid and only falsifiable via external verification. This harness **manufactures checkability** (e.g., citation-forcing lanes) and uses resolvers + receipts to distinguish compliance, evasion, and fabrication. It does not certify truth — it measures specific, falsifiable failure classes.

The detector is also a leverage-mode correlator (Paper 16): it extracts discriminating power from the temporal baseline between fast LLM inference and slow evidence verification. The mismatch the detector measures is the same mismatch it exploits — confidence arrives in milliseconds, resolver evidence in seconds, and the gap between them is the instrument's aperture.

## How it works

1. **Lanes / corpora** define prompts with expected constraints (canary, ladder, N-pressure, coupling)
2. **Extract anchors** from LLM output (DOI, arXiv, URL, RFC, PMID, ISBN)
3. **Resolve / validate** each anchor against external registries (doi.org, arxiv.org, etc.)
4. **Score** via Epistemic Grounding (EG) → FAIL / WARN / CLEAN verdict
5. **Store** an immutable run bundle: `predictions.jsonl`, `summary.json`, `manifest.json` (hash-pinned, crash-safe)

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

## Quick start

### Run the canonical regression lane (15 prompts, ~5 min on GPU)

```bash
python -m detector.run eval \
  --file data/canonical_15.jsonl \
  --profile general \
  --multi-invariant
```

### Run one prompt interactively

```bash
python -m detector.run detect \
  --prompt "Provide 2 DOIs for papers on vision transformers." \
  --profile general
```

### Check devices

```bash
python -m detector.run devices
```

### Run tests

```bash
python3 -m pytest detector/tests.py -v
```

## Experiments included

### Citation ladder (L1-L6)

60 prompts with escalating citation pressure. L1 (bare ask) → L6 (abstention allowed). Measures the crossover from evasion to fabrication.

### N-pressure curve

20 prompts varying N=1,2,3,5 requested citations. N=2 is the "perfect lie" zone — maximal plausible fabrication, zero evasion.

### Coupling topology (single vs chain vs hub)

Tests whether chaining or hub-aggregating multiple LLM steps shifts the lie mix from fabrication to mismatch. Same 15 canonical prompts, same model, same resolver.

```bash
python3 scripts/coupling.py --device cuda --corpus data/canonical_15.jsonl
```

## Risk profiles

```python
detector.set_risk_profile('medical')   # Very conservative
detector.set_risk_profile('legal')     # Conservative
detector.set_risk_profile('general')   # Balanced (default)
detector.set_risk_profile('creative')  # Permissive
```

## The four invariants

The detector implements four invariant tests from the companion paper:

| Invariant | What it tests | Status |
|-----------|---------------|--------|
| **Temporal Coherence (TC)** | Does confidence outpace evidence? | WARN-only (telemetry) |
| **Semantic Conservation (SC)** | Stable meaning across rephrasings? | WARN-only (telemetry) |
| **Epistemic Grounding (EG)** | Do sources actually exist and support claims? | **FAIL-gating** |
| **Irreversibility** | Do errors leave residue? | Experimental |

Only EG violations produce FAIL. TC/SC violations are demoted to WARN (they measure model variability, not fabrication).

## Run bundles

Every eval run produces an immutable bundle in `runs/`:

```
runs/<run_id>_<timestamp>/
  manifest.json       # hash-pinned metadata (written last = crash-safety gate)
  predictions.jsonl   # per-prompt results
  summary.json        # aggregate metrics + flight recorder
  steps.jsonl         # (coupling only) per-step intermediates
```

## Structured signals

For governor-style integration:

```python
from detector.reporting import build_signal
from detector.governor_signal import build_governor_signal

result = detector.detect("What is the capital of France?")
signal = build_signal(result.report)          # reporting signal
gov_signal = build_governor_signal(result)    # 19-field governor vector
```

JSON schemas: `schema/signal.schema.json`, `schema/governor_signal.schema.json`

## Project structure

```
delta-t-detector/
  detector/
    core.py            # DeltaTDetector (requires torch)
    invariants.py      # TC, SC, EG, Irreversibility tests
    eval.py            # JSONL corpus runner
    run_store.py       # Append-only run storage
    governor_signal.py # 19-field signal for downstream governor
    config.py          # Risk profiles
    features.py        # Temporal feature extraction
    baseline.py        # Model calibration
    reporting.py       # Output formatting + structured signals
    run.py             # CLI entry point
    tests.py           # Test suite
  scripts/
    coupling.py        # Coupling topology experiment
    overnight.py       # Lane-based overnight harness
    replay.py          # CPU-only threshold scan
    flappers.py        # Flapper report (tier flips across runs)
  data/
    canonical_15.jsonl # N=2 + L4 regression lane
    eval_seed_v3.jsonl # 90-prompt full corpus
    canary_10.jsonl    # Fast canary lane
    ...
  bin/
    overnight.sh       # Non-interactive overnight wrapper
  schema/
    signal.schema.json
    governor_signal.schema.json
```

## Non-goals

- We don't solve alignment.
- We don't certify truth.
- We focus on **falsifiable anchors** and **governance artifacts** — failure modes you can check mechanically.

## Research lineage

This repo is the companion code for three papers:

> Beck, J. (2025). "You Need More Than Just Attention: Invariant Requirements for Temporal Coherence in AI Systems." [DOI: 10.5281/zenodo.18039926](https://zenodo.org/records/18039927)

> Beck, J. (2025). "Detecting Temporal Debt in Language Models and Software Systems: Applications of Δt-Constrained Inference." [DOI: 10.5281/zenodo.17859323](https://zenodo.org/records/17859324)

> Beck, J. (2026). "Cybernetic Fault Domains: When Commitment Outruns Verification." Section 3.2. [DOI: 10.5281/zenodo.18518894](https://zenodo.org/records/18518895)

> Beck, J. (2026). "The Gain Geometry of Temporal Mismatch: Shear, Leverage, and Capture in Multi-Timescale Systems." Preprint, Δt Framework Paper 16.

Paper #10 (Invariant Requirements) defines the four invariants the detector tests. Paper #08 (Detecting Temporal Debt) covers the hallucination detection domain and Δt diagnostic. Paper #15 (Cybernetic Fault Domains) provides the CFDD framework where §3.2 instantiates this detector. Paper #16 (Gain Geometry) identifies the detector as a leverage-mode correlator that extracts resolution from the temporal baseline between inference speed and evidence speed.

## License

Apache 2.0

## Links

- [Paper: Invariant Requirements for Temporal Coherence (Zenodo)](https://zenodo.org/records/18039927)
- [Paper: Detecting Temporal Debt (Zenodo)](https://zenodo.org/records/17859324)
- [Paper: Cybernetic Fault Domains (Zenodo)](https://zenodo.org/records/18518895)
- [Companion repo: scalar-reward-collapse](https://github.com/unpingable/scalar-reward-collapse)
- [Substack: The Neutral Zone](https://neutralzone.substack.com/)
