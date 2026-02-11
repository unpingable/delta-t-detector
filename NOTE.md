# Namespace-Dependent Fabrication in Small Language Models

## Key Claims

1. Fabrication rate is namespace-dependent (0% RFC → 45% DOI), not model-global
2. The namespace ordering is model-specific (Qwen and Phi-3 invert on CVE vs PyPI)
3. Format locking exposes latent fabrication or removes noise, depending on memorization
4. Models actively avoid checkable channels (format-shift evasion)
5. Hub/merge topologies amplify fabrication (+19pp); single-agent is strictly better
6. Retry at temp=0.7 is net-dangerous (67% harm rate); retry at temp=0 is safe (0% harm)
7. Temperature is a lie amplifier: ~50% of measured fabrication is sampling noise
8. Models show distinct policy fingerprints: "lies to comply" (Qwen) vs "evades; lies when trapped" (Phi-3)
9. Scale (3B→7B) halves fabrication and closes knowledge boundaries; 4-bit quantization preserves the effect

---

**Setup.** We probe three small instruction-tuned models — Qwen 2.5 3B-Instruct (3B), Qwen 2.5 7B-Instruct (7B, NF4 4-bit), and Phi-3 Mini 3.8B-Instruct (3.8B) — for citation fabrication across four identifier namespaces: RFC, CVE, PyPI package versions, and DOI/arXiv. Each namespace has an authoritative existence oracle (rfc-editor.org, MITRE CVE API, PyPI JSON API, doi.org). We use N=2 citation pressure (ask for exactly 2 identifiers) at temperature 0.7 with deterministic seeding. Format-locked variants forbid URLs and enforce identifier-only output. Findings 1-6 report Qwen 3B results; Findings 7-9 compare across models and temperatures; Finding 10 tests the scale gradient.

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

### 7. The namespace spectrum is model-specific

Running the same four-lane suite on Phi-3 Mini 3.8B-Instruct:

| Lane (locked) | Qwen 2.5 3B | Phi-3 Mini 3.8B |
|----------------|-------------|-----------------|
| RFC            | 0%          | 0%              |
| CVE            | 9%          | **41%**         |
| PyPI           | 25%         | **0%**          |
| DOI/arXiv      | ~18%        | ~37%            |

The existence of namespace-dependent fabrication is universal, but the ordering is model-specific. CVE and PyPI are inverted: Qwen memorized CVEs but not PyPI versions; Phi-3 evades PyPI claims and fabricates CVEs. Phi-3's 0% PyPI fabrication comes from evasion (7/10 prompts triggered format-shift or missing-anchor warnings), not honest compliance. When it cannot evade (CVE format is harder to dodge), fabrication jumps to 41%. A governance policy calibrated on one model family will be exactly wrong for another.

### 8. Retry enforcement reveals a three-way behavioral split

When evasion is detected, a single strict retry with an UNKNOWN abstention escape hatch (e.g. `pypi:UNKNOWN==0.0.0`) reveals three distinct outcomes:

| Model | Corpus | Retries | → Clean | → Fail | → Abstain |
|---|---|---|---|---|---|
| Phi-3 | PyPI locked | 3 | 1 | 1 | 1 |
| Phi-3 | CVE locked | 2 | 0 | 2 | 0 |
| Qwen | PyPI locked | 0 | — | — | — |
| Qwen | CVE locked | 1 | 0 | 1 | 0 |

Qwen never evades on PyPI (nothing to retry). Phi-3 shows all three outcomes on PyPI: comply, abstain, fabricate. On CVE (unmemorized for Phi-3), retry uniformly converts to fabrication — the model lies when forced on things it doesn't know. UNKNOWN sentinels were used only by Phi-3 on PyPI (1/3 retries), never on CVE — the abstention escape hatch is model- and namespace-dependent.

### 9. Temperature separates sampling noise from knowledge boundaries

At temperature=0 (greedy), retry-induced fabrication disappears completely:

| Metric | Temp=0.7 | Temp=0 |
|---|---|---|
| Retry WARN→FAIL (harm) | 4/6 (67%) | **0/7 (0%)** |
| Retry WARN→CLEAN (benefit) | 2/6 (33%) | **6/7 (86%)** |
| 1st-attempt FAIL on PyPI | 3/20 | **0/20** |
| 1st-attempt FAIL on CVE | 3/20 | **3/20** |

PyPI fabrication at temp=0.7 was ~100% sampling noise (drops to zero at temp=0). CVE fabrication persists — it's a genuine knowledge boundary. Persistent evasion (WARN→WARN) appears only at temp=0, showing that temperature masks the model's true refusal rate by converting deterministic refusals into probabilistic fabrications.

Fabrication has two sources: **sampling-accessible fabrication** (goes away at greedy — the model's mode is correct but temperature pushes it into plausible-looking errors) and **knowledge-boundary failures** (persist at greedy — the mode itself is wrong).

### 10. Scale halves fabrication and closes knowledge boundaries

Running Qwen 2.5 7B-Instruct (NF4 4-bit) on the same locked corpora:

| Lane (locked) | Qwen 3B t=0.7 | Qwen 7B 4-bit t=0.7 | Qwen 7B 4-bit t=0 |
|---|---|---|---|
| RFC | 0% | 0% | 0% |
| PyPI | 25% | 10% | 5% |
| CVE | 9% | 6.2% | 5% |

At the prompt level, CVE locked at 7B produces 0 FAILs at greedy — the knowledge boundary that persisted at 3B (1 FAIL at temp=0) is closed. Scale shifts namespaces from "knowledge boundary" to "sampling noise only." PyPI crossed this threshold at 3B; CVE crosses at 7B. The implication: fabrication ∝ 1/(memorization × scale), and each model size has a namespace-specific crossover point where greedy decoding eliminates fabrication entirely. 4-bit quantization preserves this effect — NF4 does not reintroduce knowledge-boundary fabrication.

## Design Rules

1. **Force checkable channels.** Require typed identifiers (`pypi:name==version`, `CVE-YYYY-NNNNN`), not URLs. Use authoritative existence oracles per namespace (MITRE CVE API, PyPI JSON, doi.org). HEAD/200 is not validation — `cve.mitre.org` returns 200 for nonexistent CVEs; Wikipedia returns 403 or 404 depending on platform. Only 200 and 404 from type-specific APIs are definitive.

2. **Retry is an intervention, not a better sample.** At temp=0.7, retry converts evasion to fabrication 67% of the time. Never act on a retry without verification. Pair with stronger validation, not blind acceptance. Abstention (`UNKNOWN` sentinels) is a first-class outcome — a model that refuses is safer than one that fabricates.

3. **Gate steps run greedy.** Generate the first answer at operational temperature. Any enforcement retry, "produce anchors" step, or checkable-output step must be greedy (temp=0). Temperature is risk budget; don't spend it on falsifiable identifiers.

4. **The namespace spectrum is model-indexed.** A model that aces RFC citations but fabricates 45% of DOIs is not "honest about citations." The ordering itself varies across model families (Qwen: CVE easy, PyPI hard; Phi-3: reversed). Governance policies must test the specific model against the specific namespaces that matter for their domain. Don't transfer results across model families.

5. **Don't aggregate citations with LLMs.** Hub/merge topologies amplify fabrication (+19pp). The merge operator creates new lies. Single-agent is strictly better for citation integrity at 3-4B scale. If you must aggregate, use code-level passthrough, not LLM rewriting.

## Replication

Qwen 3B measurements were replicated across Linux (x86_64, NVIDIA RTX 5060 Ti 16GB) and macOS (ARM64, Mac mini M4). Resolver behavior is platform-invariant for authoritative APIs. One URL-level discrepancy (Wikipedia 403 vs 404 for a fabricated page) confirmed the fragility of generic HEAD checks. Phi-3 Mini and Qwen 7B (NF4 4-bit) were tested on the same Linux hardware with identical corpora and decoding parameters.

Code, corpora, and run artifacts: [repository link]
