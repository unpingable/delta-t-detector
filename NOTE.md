# Namespace-Dependent Fabrication in Small Language Models

**Setup.** We probe a 3B-parameter instruction-tuned model (Qwen 2.5 3B-Instruct) for citation fabrication across five identifier namespaces: RFC, CVE, PyPI package versions, DOI, and arXiv. Each namespace has an authoritative existence oracle (rfc-editor.org, MITRE CVE API, PyPI JSON API, doi.org). We use N=2 citation pressure (ask for exactly 2 identifiers) at temperature 0.7 with deterministic seeding. Format-locked variants forbid URLs and enforce identifier-only output.

## Findings

### 1. Fabrication rate is a function of namespace memorization

| Namespace | Fabrication (locked) | Memorization | Failure Mode |
|-----------|---------------------|--------------|--------------|
| RFC       | 0%                  | Full         | None |
| CVE       | 9%                  | High         | Recent/obscure IDs |
| PyPI      | 25%                 | Partial      | Version fabrication |
| DOI/arXiv | ~45%                | Low          | Wholesale fabrication |

The same model, at the same temperature, with the same prompt structure, produces 0% fabrication for one namespace and 45% for another. Citation integrity is per-namespace, not per-model.

### 2. Format locking reveals or removes fabrication depending on namespace

When we constrain output format (forbid URLs, require only typed identifiers):

- **PyPI**: fabrication *increases* (13% to 25%). URLs were hiding version ignorance.
- **CVE**: fabrication *decreases* (15% to 9%). URLs were the fabrication source; the CVE IDs were real.

General principle: locking increases fabrication when it forces unmemorized fields; locking decreases fabrication when it removes escape hatches.

### 3. Format shift is the dominant evasion behavior

Under soft formatting (suggest `pypi:name==version` but don't forbid alternatives), 80% of prompts substituted URLs for the requested format. The model satisfies the user's intent while avoiding the exact representation that enables version-level validation.

Three-way behavioral split under soft vs locked prompting:

| Behavior | Soft | Locked |
|----------|------|--------|
| Comply (correct identifier) | 7% | 75% |
| Fabricate (wrong identifier) | 13% | 25% |
| Format shift (URLs instead) | 80% | 0% |

Fabrication was always latent. Format shift hid it.

### 4. Multi-agent hub topology increases fabrication

When two agents generate independently and a third merges their outputs:

| Topology | Fabrication | Lie Rate |
|----------|-------------|----------|
| Single   | 17.8%       | 44.6%    |
| Chain    | 12.5%       | 38.5%    |
| Hub      | 36.7%       | 62.2%    |

Causal decomposition of hub's +19pp fabrication uplift: selection pressure (+13pp) > synthesis (+5pp) > role framing (+1pp). The merge operator optimizes for coherence over accuracy. Non-generative selection and code-enforced provenance tracking reduce fabrication by only 2-5pp. At 3B scale, single-agent is strictly better for citation integrity.

### 5. URL HEAD checks are not identifier validation

`cve.mitre.org` returns HTTP 200 for nonexistent CVE IDs (search results page). Wikipedia returns HTTP 403 on some platforms and 404 on others for the same fabricated page. Generic URL reachability is sensitive to platform, TLS stack, DNS resolution, and server-side behavior. Type-specific validators using authoritative APIs (MITRE CVE API, PyPI JSON, doi.org) are platform-invariant (164/165 anchors agreed across Linux and macOS).

### 6. N=2 is the danger zone

| N (citations requested) | Fabrication | Evasion |
|--------------------------|-------------|---------|
| 1                        | 14%         | 20%     |
| 2                        | 50%         | 0%      |
| 3                        | 47%         | 0%      |
| 5                        | 26%         | 60%     |

N=2 maximizes plausible fabrication with zero evasion. N=1 is too easy (model complies honestly). N=5 overwhelms the model into evasion. The attractor for "lies that pass casual inspection" is N=2.

## Design Implications

1. **Force checkable channels.** Require typed identifiers (`pypi:name==version`, `CVE-YYYY-NNNNN`) rather than accepting URLs. Format locking exposes fabrication that URL substitution conceals.

2. **Use authoritative existence oracles.** Every namespace needs its own validator. HEAD/200 is necessary but not sufficient. HTTP 401/403 is ambiguous (platform-dependent). Only 200 and 404 from authoritative APIs are definitive.

3. **Don't aggregate with LLMs.** Hub/merge topologies amplify fabrication. If you must aggregate, use non-generative selection or code-level passthrough, not LLM rewriting. The merge operator creates new lies.

4. **Citation integrity claims must specify namespace.** A model that aces RFC citations but fabricates 45% of DOIs is not "honest about citations." Governance policies should test against the namespaces that matter for their domain.

5. **Treat timing/confidence signals as telemetry, not gates.** Token-level confidence saturates too fast in small models to discriminate truth from fabrication. Anchors with external oracles are the primary sensor. Temporal coherence is a secondary triage signal, never a standalone detector.

## Replication

All measurements were replicated across Linux (x86_64, NVIDIA RTX 5060 Ti) and macOS (ARM64, Mac mini M4). Resolver behavior is platform-invariant for authoritative APIs. One URL-level discrepancy (Wikipedia 403 vs 404 for a fabricated page) confirmed the fragility of generic HEAD checks.

Code, corpora, and run artifacts: [repository link]
