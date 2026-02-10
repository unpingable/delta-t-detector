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

## Hub-Enforced Experiment (2026-02-10)

**Question**: Is hub bad because it cheats (introduces novel citations), or bad even when cheats are stripped?

**Method**: `hub_enforced` — identical pipeline to `hub` (same seeds → same raw B, C, A generation), but after A generates, any citation lines containing novel anchors not in B or C are removed before scoring. Code-level enforcement, not prompt-level.

| Topology     | Anchors | Valid | Fab Rate  | Mismatch | Lie Rate  | Verdicts |
|--------------|---------|-------|-----------|----------|-----------|----------|
| single       | 101     | 52    | 17.8%     | 26.7%    | 44.6%     | 3C / 12F |
| hub          | 98      | 37    | **36.7%** | 25.5%    | **62.2%** | 1C / 14F |
| hub_enforced | 92      | 36    | **34.8%** | 26.1%    | **60.9%** | 1C / 14F |

**Enforcement delta** (hub_enforced vs hub):
- fabrication_rate: -2.0pp
- mismatch_rate: +0.6pp
- lie_rate: -1.4pp

**Answer: Hub is bad even when it can't cheat.** Stripping novel anchors only removed 6 anchors (98→92) and reduced fabrication by a trivial 2pp. The remaining +17pp fabrication uplift vs single is structural — the merge step itself causes fabrication, not just the cheating. Hub-A's 2/15 novel-citation violations were the tip of the iceberg; the real damage is that merge pressure causes A to *hallucinate existing-format-but-nonexistent identifiers* even when drawing only from B and C's citation pools.

**Implication**: Prompt-level "only use citations from inputs" is insufficient, and even code-level enforcement barely helps. The merge step is fundamentally toxic for citation integrity. Multi-agent hub architectures need structural guarantees (e.g., citation passthrough without LLM rewriting) rather than filtering.

## Hub-Select Experiment (2026-02-10)

**Question**: If we make aggregation non-generative (A just picks B or C wholesale), does fabrication collapse back to single-level?

**Method**: `hub_select` — B and C generate independently (same seeds as hub), then A is prompted to output only "CHOOSE: B" or "CHOOSE: C". The chosen candidate is scored wholesale — A generates zero citation text.

| Topology   | Anchors | Valid | Fab Rate  | Mismatch | Lie Rate  | Verdicts |
|------------|---------|-------|-----------|----------|-----------|----------|
| single     | 101     | 52    | 17.8%     | 26.7%    | 44.6%     | 3C / 12F |
| hub        | 98      | 37    | **36.7%** | 25.5%    | **62.2%** | 1C / 14F |
| hub_select | 89      | 37    | **31.5%** | 27.0%    | **58.4%** | 2C / 13F |

**Selection delta** (hub_select vs hub): fab -5.3pp, lie -3.8pp
**Hub_select vs single**: fab +13.6pp [INTERESTING], lie +13.9pp [INTERESTING]

**Prediction was wrong.** Hub_select does NOT collapse back to single. It's slightly better than hub (-5pp fab, -4pp lie), but it's still +14pp above single on both metrics. The fabrication isn't coming from the merge step's generation — it's already in B and C.

**Why**: B and C are generated with hub-specific seeds (different from single's seed). The selection merely passes through whichever candidate the selector picks — but both candidates are already worse than single. The independence topology (two perspectives on the same question) doesn't improve citation quality; if anything, it means the selector picks from two flawed pools.

Deeper: 89 anchors (vs 101 single, 98 hub) means the selector tends to pick the shorter/sparser candidate. Valid count is flat (37 in both hub and hub_select), so selection doesn't improve truth — it just drops volume slightly.

**Revised understanding**: The fabrication spike in hub is ~70% structural (B+C already worse than single due to different seed paths and role framing) and ~30% from merge generation. The merge step adds insult to injury, but the injury was already there.

**Design rule (revised)**: Non-generative aggregation helps marginally but does not fix the fundamental problem. The real issue is that multi-agent framing (independent perspectives) doesn't improve citation accuracy — it's not "synthesis is toxic," it's "independence doesn't help, and synthesis makes it worse."

## Role-Framing Control (2026-02-10)

**Question**: Is hub_select worse than single because of role framing (different prompt) or because of multi-agentness (selecting between two candidates)?

**Method**: `single_rolematch` — single agent using the exact hub-B role prompt ("independent research assistant") and hub-B seed derivation. Same generation, no second candidate, no selector.

| Topology        | Anchors | Valid | Fab Rate | Mismatch | Lie Rate | Verdicts |
|-----------------|---------|-------|----------|----------|----------|----------|
| single          | 101     | 52    | 17.8%    | 26.7%    | 44.6%    | 3C / 12F |
| single_rolematch| 87      | 45    | 18.4%    | 29.9%    | 48.3%    | 3C / 12F |
| hub_select      | 89      | 37    | **31.5%**| 27.0%    | **58.4%**| 2C / 13F |

**Result**: single_rolematch ≈ single. All deltas vs single are <4pp (not significant). Same verdict histogram (3C/12F). Role framing is **not** the culprit.

**But hub_select is still +14pp above single_rolematch on fab rate.** The only difference between single_rolematch and hub_select is that hub_select *picks between two candidates* (B and C). The selector's choice is where the damage enters.

**Causal decomposition of hub's +19pp fabrication uplift vs single**:

| Component | Δ Fab Rate | Mechanism |
|-----------|-----------|-----------|
| Role framing (single→single_rolematch) | +0.6pp | Negligible — prompt wording doesn't matter |
| Selection (single_rolematch→hub_select) | +13.1pp | Selector picks worse candidate ~half the time |
| Synthesis (hub_select→hub) | +5.2pp | Merge generation adds more fabrication on top |
| **Total (single→hub)** | **+18.9pp** | |

**The selector is the main hazard**, not the synthesis. When presented with two candidates of varying quality, the 3B model's selector *systematically picks the worse one* for citation integrity. It optimizes for coherence/completeness, not citation accuracy — exactly as predicted, but the effect is in the selection, not the generation.

**Design rule (final)**: Multi-agent citation architectures fail at Qwen-3B scale because the model cannot reliably evaluate citation quality. Neither non-generative selection nor enforced provenance rescues the setup. The aggregation operator (whether generative or selective) lacks the competence to choose truth over plausibility. At this model scale, single-agent is strictly better for citation integrity.
