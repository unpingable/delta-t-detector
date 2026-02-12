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
10. Two-stage decoding (primary warm, retry greedy) eliminates retry harm on memorized namespaces; scale makes it redundant
11. Seed sensitivity is a model fingerprint: Qwen is stable, Phi-3 is chaotic even at greedy
12. Top-2 logit margin at identifier tokens predicts drift_class: Phi-3 CVE has median margin 0.0 (half tokens at tie)
13. Margin-based controller with authoritative grounding yields six terminal policies; authoritative 404 is evidence, not absence of evidence
14. Section integrity catches fabrication invisible to existence oracles: real RFCs cited with wrong or nonexistent section numbers

---

**Setup.** We probe three small instruction-tuned models — Qwen 2.5 3B-Instruct (3B), Qwen 2.5 7B-Instruct (7B, NF4 4-bit), and Phi-3 Mini 3.8B-Instruct (3.8B) — for citation fabrication across four identifier namespaces: RFC, CVE, PyPI package versions, and DOI/arXiv. Each namespace has an authoritative existence oracle (rfc-editor.org, MITRE CVE API, PyPI JSON API, doi.org). We use N=2 citation pressure (ask for exactly 2 identifiers) at temperature 0.7 with deterministic seeding. Format-locked variants forbid URLs and enforce identifier-only output. Findings 1-6 report Qwen 3B results; Findings 7-9 compare across models and temperatures; Finding 10 tests the scale gradient; Findings 11-13 quantify seed sensitivity and its mechanistic cause (top-2 margin); Finding 14 builds a runtime controller with authoritative grounding; Finding 15 extends grounding from existence to section-level integrity.

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

### 11. Two-stage decoding eliminates retry harm on memorized namespaces

Comparing three retry policies (no retry, same-temp retry at 0.7, two-stage with retry at 0.0):

| Model | Lane | Same-temp harm | Two-stage harm | Two-stage benefit |
|---|---|---|---|---|
| Phi-3 | PyPI (memorized) | 33% | **0%** | **100%** |
| Phi-3 | CVE (unmemorized) | 100% | 50% | 50% |
| Qwen 3B | CVE | 100% | **0%** (→ persistent evasion) | 0% |
| Qwen 7B | both | — (0 retries) | — (0 retries) | — |

Two-stage eliminates all retry harm when the model knows the answer (memorized namespace + behavioral evasion). On unmemorized namespaces, harm drops but persists — greedy can't fix ignorance. For Qwen, two-stage converts WARN→FAIL to persistent evasion (WARN→WARN), which is strictly safer. At 7B scale, the controller becomes redundant — no evasion means nothing to retry.

### 12. Seed sensitivity is a model fingerprint

3-seed drift check (seeds 42, 137, 271) across all models and lanes:

| Model | Drift class at t=0.7 | Drift class at t=0 |
|---|---|---|
| Qwen-7B-4bit | STABLE (max 5pp) | STABLE (max 0.1pp) |
| Qwen-3B | SEED_SENSITIVE on PyPI (10pp) | STABLE (max 1pp) |
| Phi-3-Mini | CHAOTIC (CVE 27pp, PyPI 15pp) | SEED_SENSITIVE (CVE 11pp) |

Seed sensitivity is inversely correlated with model quality. Qwen-7B is rock-stable everywhere. Qwen-3B drifts only at sampling temperature on its weakest namespace. Phi-3 is chaotic at t=0.7 and still seed-sensitive at pseudo-greedy — the model's logit surface is flatter, so even minimal temperature creates large behavioral variation. The two-stage controller is perfectly safe across all 3 seeds for Phi-3 on PyPI (12/12 retries → clean, 0 harm).

### 13. Top-2 logit margin predicts drift class

Computing the margin between top-1 and top-2 token probability (M = p1 - p2) at each generation step within identifier emission windows:

| Model | PyPI M_median | CVE M_median | Drift Class |
|---|---|---|---|
| Qwen-7B-4bit | 0.35 | 0.15 | STABLE |
| Qwen-3B | 0.24 | 0.05 | SEED_SENSITIVE |
| Phi-3-Mini | 0.22 | **0.00** | CHAOTIC |

M_median is the median per-prompt minimum margin across identifier windows. Phi-3 on CVE has median margin 0.0 — half of all prompts contain at least one identifier token where the model is at a complete tie between two next-token candidates. This is the mechanistic cause of the CHAOTIC classification: flat logit peaks at identifier tokens mean temperature trivially flips the output.

The causal chain: low margin → flat logits → temperature amplifies fork probability → seed sensitivity → fabrication variance. The margin is the cause; drift_class is the symptom. This can be computed online during generation to trigger controller mode switches.

### 14. Margin-based runtime controller with grounding validates six terminal regimes

Turning the margin metric into a live per-prompt policy (fork_risk = m_min at identifier windows, τ=0.05). When fork_risk < τ, the controller first attempts **grounding** (fetch authoritative metadata, check relevance) before falling back to greedy retry.

| Policy | Trigger | Qwen-3B CVE | Phi-3 CVE |
|---|---|---|---|
| GROUNDED | Low margin, anchors confirmed | 0% | 0% |
| GROUNDED_REFUTED | Low margin, anchors refuted/not_found | 30% | 50% |
| LOW_MARGIN_RETRY | Low margin, grounding inconclusive | 10% | 20% (2 GAIN) |
| CONFIDENT_WRONG | High margin, oracle FAIL | 0% | 10% |
| KNOWLEDGE_BOUNDARY | Retry also FAILs | 0% | 0% |
| FAST_PATH | High margin, oracle pass | 60% | 20% |

Key design insight: for authoritative APIs (MITRE CVE, PyPI JSON, rfc-editor.org), a 404 is `not_found` — evidence of non-existence, not inconclusive. Treating it as neutral created a false positive (grounding "confirmed" a prompt with one real + one fabricated CVE). Counting `not_found` as negative evidence fixed the classification and unlocked a retry gain (FAIL→CLEAN on Phi-3).

CONFIDENT_WRONG fires on Phi-3 CVE (m_min=0.18, oracle FAIL) but not Qwen-3B CVE (all fabrication at m_min=0.0). Grounding saves retry tokens when anchors can be resolved. Zero regressions at τ=0.05 across all runs.

### 15. Section integrity catches fabrication invisible to existence oracles

Extending grounding from "does this RFC exist?" to "did the model cite the right section?" RFC plain text is fetched from rfc-editor.org, section headings are parsed at column 0, and each "RFC NNNNN Section X.Y" reference is verified against the actual section map.

On a 5-prompt section-specific RFC corpus (Qwen 3B, t=0.7):
- **RFC 6749 §4** (OAuth): verified — actual title "Obtaining Authorization" matches prompt about authorization grants (relevance 0.50)
- **RFC 6455 §5.1** (WebSocket): wrong_section — actual title "Overview," not the opening handshake (which is §4). Relevance 0.0.
- **RFC 7589 §5.1.1** (WebSocket): no_such_section — this 13-section NETCONF document doesn't have §5.1.1

The existence oracle says CLEAN for all three (the RFCs are real). Section integrity reveals the model fabricated section-level claims while citing correct documents — the citation analog of "correct name, wrong address." This failure mode is invisible to existence-only checking and represents the next layer of the grounding hierarchy.

## Design Rules

1. **Force checkable channels.** Require typed identifiers (`pypi:name==version`, `CVE-YYYY-NNNNN`), not URLs. Use authoritative existence oracles per namespace (MITRE CVE API, PyPI JSON, doi.org). HEAD/200 is not validation — `cve.mitre.org` returns 200 for nonexistent CVEs; Wikipedia returns 403 or 404 depending on platform. Only 200 and 404 from type-specific APIs are definitive.

2. **Retry is an intervention, not a better sample.** At temp=0.7, retry converts evasion to fabrication 67% of the time. Never act on a retry without verification. Pair with stronger validation, not blind acceptance. Abstention (`UNKNOWN` sentinels) is a first-class outcome — a model that refuses is safer than one that fabricates.

3. **Gate steps run greedy.** Generate the first answer at operational temperature. Any enforcement retry, "produce anchors" step, or checkable-output step must be greedy (temp=0). Temperature is risk budget; don't spend it on falsifiable identifiers.

4. **The namespace spectrum is model-indexed.** A model that aces RFC citations but fabricates 45% of DOIs is not "honest about citations." The ordering itself varies across model families (Qwen: CVE easy, PyPI hard; Phi-3: reversed). Governance policies must test the specific model against the specific namespaces that matter for their domain. Don't transfer results across model families.

5. **Don't aggregate citations with LLMs.** Hub/merge topologies amplify fabrication (+19pp). The merge operator creates new lies. Single-agent is strictly better for citation integrity at 3-4B scale. If you must aggregate, use code-level passthrough, not LLM rewriting.

6. **Measure spread, not just rate.** A single fabrication rate hides two failure modes: "stable-but-wrong" (high mean, low spread — knowledge boundary) vs "variable-but-sometimes-right" (low mean, high spread — sampling noise). Multi-seed drift checks separate them. Retry only helps the high-spread case.

## Replication

Qwen 3B measurements were replicated across Linux (x86_64, NVIDIA RTX 5060 Ti 16GB) and macOS (ARM64, Mac mini M4). Resolver behavior is platform-invariant for authoritative APIs. One URL-level discrepancy (Wikipedia 403 vs 404 for a fabricated page) confirmed the fragility of generic HEAD checks. Phi-3 Mini and Qwen 7B (NF4 4-bit) were tested on the same Linux hardware with identical corpora and decoding parameters.

Code, corpora, and run artifacts: [repository link]
