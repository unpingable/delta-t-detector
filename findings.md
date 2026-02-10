# Empirical Findings — Δt Hallucination Detector

## Behavioral Phase Map (Qwen 3B under citation pressure)

Under increasing citation pressure, the model chooses one of four behavioral regimes:
1. **Evasion** — refuses to produce anchors, talks around the question
2. **Fabrication** — produces syntactically valid but nonexistent identifiers
3. **Compliance** — produces real, resolvable identifiers
4. **Abstention** — explicitly says "I don't know" (never observed for Qwen 3B)

## N-Pressure Curve (2026-02-10, post-cache/error-classification fix)

| Level | Fab Rate | Evasion | Notes |
|-------|----------|---------|-------|
| N=1   | 14%      | 1/5     | Low pressure, mostly compliant |
| N=2   | **50%**  | 0/5     | "Perfect lie" zone — maximal plausible fabrication, zero evasion |
| N=3   | **47%**  | 0/5     | Still deep in fabrication territory |
| N=5   | 26%      | 3/5     | Compliance-to-evasion transition — can't satisfy, switches strategy |

**Key insight**: N=2 is the *danger zone* — the model is confident enough to fabricate convincingly but not overwhelmed enough to evade. This is the attractor for "lies that pass casual inspection."

## Citation Ladder Results (2026-02-10)

| Level | Fab Rate | Evasion | Zero-Anchor | Notes |
|-------|----------|---------|-------------|-------|
| L1 (bare ask)       | 0%  | 6/10 | 6/10 | Pure evasion |
| L2 (format hint)    | 25% | 2/10 | 4/10 | Starting to try |
| L3 (explicit template) | 44% | 0/10 | 0/10 | Compliance kicks in |
| L4 (quantity constraint) | 47% | 1/10 | 0/10 | Peak fabrication |
| L5 (verification pressure) | 80% | 0/10 | 0/10 | Adversarial, very high fab |
| L6 (abstention allowed) | 80% | 0/10 | 0/10 | Abstention option ignored |

**L4 is the sweet spot** for detection: high fabrication, low evasion, non-adversarial.

## Measurement Engineering: "Tooling Got Better → Reality Got Worse" (2026-02-10)

Added resolver cache + explicit error classification (don't count timeouts/rate-limits as invalid).

Before fix vs after:
| Level | Fab Rate (before) | Fab Rate (after) | Delta |
|-------|-------------------|------------------|-------|
| N=1   | 33%               | 14%              | -19pp (timeouts were inflating) |
| N=2   | 42%               | **50%**          | +8pp (real fab hidden by rate-limit noise) |
| N=3   | 37%               | **47%**          | +10pp (same) |
| N=5   | 29%               | 26%              | -3pp (evasion still dominates) |

The measurement surface changed. N=1 was overstated (transient errors counted as fabrication). N=2/N=3 were *understated* — resolver failures were masking real fabrication. This is exactly the "stopped lying to yourself" pattern.

**Lesson**: Any eval metric that touches network I/O must distinguish definitive results from transient failures, or you're measuring your resolver's uptime, not the model's honesty.

## FAIL Split: SC/TC Demotion (2026-02-10)

Demoting SC/TC from FAIL-gating to WARN-only eliminated false-positive FAILs on canary (was 2, now 0). Only EG violations (fabricated identifiers) can produce FAIL. This matches the empirical reality: SC/TC violations are "model variability" (background radiation), not fabrication.

## Format vs Resolve Invalid

All observed invalidations are resolve-invalid (HTTP 404), zero format-invalid. The model generates syntactically valid DOIs and arXiv IDs that simply don't exist. Regex/format checks are past their useful phase — the model has learned the format.

## Canonical Regression Lane (2026-02-10)

Locked N=2 + L4 as the canonical regression lane (15 prompts). Rationale:
- Zero evasion (model always tries to comply)
- Maximal plausible fabrication (~50%)
- Highly sensitive to real changes in model behavior or detection logic
- Non-adversarial (unlike L5/L6)

This is the lane to watch for coupling topology experiments and cross-platform comparisons.

## Coupling Topology Hypotheses (pre-registered 2026-02-10)

**Question**: Does coupling multiple LLM steps change the *lie mix*? Specifically, does chain or hub topology shift failures from FABRICATED_IDENTIFIER → MISMATCHED_CITATION (lies get more plausible)?

- **H1**: Coupling reduces fabrication but increases mismatch (lies get polished). Mechanism: later steps "fix" obviously fake identifiers by substituting real-but-wrong ones.
- **H2**: Hub reduces both via independence, or increases mismatch via merge-confabulation. Mechanism: two independent answers disagree on citations, merger picks plausible-looking ones.
- **H3**: Chain increases confident storytelling → more mismatch. Mechanism: each step builds on prior narrative, increasing coherence at the expense of accuracy.

**Pass/fail for "interesting"**: Any topology causes ≥10pp change in mismatch_rate, fabrication_rate, or lie_rate vs single.

**Design**: Same 15 canonical prompts (N=2 + L4), same model (Qwen 3B), same temperature (0.7), deterministic seeding (SHA-256 of prompt_id+topology+step), same resolver + cache. Only the final output scored by EG. Intermediates logged to steps.jsonl. Hub-A provenance tracked (novel citations not in B or C).

## Coupling Topology Results (2026-02-10)

| Topology | Anchors | Valid | Fab Rate | Mismatch | Lie Rate | Evasion | Verdicts |
|----------|---------|-------|----------|----------|----------|---------|----------|
| single   | 101     | 52    | 17.8%    | 26.7%    | 44.6%    | 0.0%    | 3C / 12F |
| chain    | 104     | 62    | 12.5%    | 26.0%    | 38.5%    | 6.7%    | 3C / 12F |
| hub      | 98      | 37    | **36.7%** | 25.5%   | **62.2%** | 0.0%   | 1C / 14F |

**Pass/fail for "interesting"** (≥10pp delta vs single):
- hub fabrication_rate: +18.9pp **[INTERESTING]**
- hub lie_rate: +17.7pp **[INTERESTING]**
- chain fabrication_rate: -5.3pp (not significant)
- chain lie_rate: -6.1pp (not significant)
- mismatch_rate: topology-invariant (~26% everywhere)

**Hypothesis outcomes**:
- **H1 (coupling polishes lies)**: **Rejected.** Neither chain nor hub shifts failures toward mismatch. Mismatch rate is ~26% regardless of topology — it appears to be a property of the model, not the pipeline.
- **H2 (hub reduces both or increases mismatch)**: **Rejected in both forms.** Hub *increases* fabrication dramatically (+18.9pp) without touching mismatch. The merge step is a liar generator, not a polisher.
- **H3 (chain increases mismatch)**: **Rejected.** Chain is a statistical wash — small fabrication decrease, small evasion increase, no mismatch shift.

**Key finding: Merge pressure is a liar generator.** Hub-A must synthesize two independent answers into one, and when B and C disagree on citations, the merger fabricates new ones rather than choosing between existing ones. Hub provenance tracking confirmed this: 2/15 prompts had hub-A introducing novel citations not present in either B or C response.

**Implication for multi-agent architectures**: Hub/merge topologies are actively toxic for citation integrity. The merge step doesn't polish lies (H1/H2) — it *creates new ones*. Chain is neutral. Any governance pipeline should treat hub outputs with higher suspicion, or enforce strict merge (code-level rejection of novel anchors, not prompt-level instruction).

**Open question**: Is hub bad because it cheats (invents new citations despite instructions), or bad even when it can't cheat? Next step: run hub with code-enforced no-novel-anchors and compare.
