#!/usr/bin/env python3
"""
Overnight harness: lane-based eval suite with nightly report.

Lanes:
  0. Canary    — tiny corpus, fast, catches regressions (~5 min)
  1. Baseline  — full v3 corpus, default config (~25 min)
  2. Citations — citation-forcing corpus, exercises Inv-3 (~15 min)
  3. Ladder    — citation ladder L1-L5, 50 prompts (~25 min)
  4. Sweep     — stability sweep: temp × n_perturb grid (~2 hr)
  5. Replay    — CPU-only threshold scan on all new runs (minutes)

Usage:
    bin/overnight.sh                    # full suite
    bin/overnight.sh --lanes canary     # just canary
    bin/overnight.sh --lanes canary,baseline,citations
    bin/overnight.sh --lanes sweep      # just stability sweep
"""

import argparse
import json
import sys
import os
import time
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.config import DetectorConfig, RISK_PROFILES
from detector.core import DeltaTDetector
from detector.eval import load_jsonl, compute_guard_flags
from detector.run_store import PredictionRecord, store_run, _utcnow_iso

# Defaults
CORPORA = {
    "canary": "data/canary_10.jsonl",
    "baseline": "data/eval_seed_v3.jsonl",
    "citations": "data/citation_forcing_30.jsonl",
    "ladder": "data/citation_ladder_60.jsonl",
}
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEVICE = "cuda"
PROFILE_NAME = "general"

# Stability sweep grid
TEMP_GRID = [0.5, 0.7, 0.9]
NPERTURB_GRID = [3, 5]

ALL_LANES = ["canary", "baseline", "citations", "ladder", "sweep", "replay"]


def run_one_eval(detector, corpus_path, profile_name, label=""):
    """Run a single eval pass, return (prediction_records, started_at)."""
    profile = RISK_PROFILES[profile_name]
    started_at = _utcnow_iso()
    prediction_records = []
    n_items = 0

    for item in load_jsonl(corpus_path):
        n_items += 1
        result = detector.detect_multi_invariant(item.prompt, test_semantic=True)
        features = result.features or {}
        anomaly_score = None
        flags = compute_guard_flags(
            features, profile, result.temporal_debt, anomaly_score, result.prediction
        )

        inv_results_dict = None
        n_violated = None
        aggregate_score = None

        if result.report:
            report_inv = result.report.invariant_results
            if isinstance(report_inv, dict):
                inv_results_dict = {}
                for inv_name, inv_data in report_inv.items():
                    if hasattr(inv_data, "to_dict"):
                        inv_results_dict[inv_name] = inv_data.to_dict()
                    elif isinstance(inv_data, dict):
                        inv_results_dict[inv_name] = inv_data
            n_violated = result.report.n_invariants_violated
            aggregate_score = getattr(result.report, "confidence", None)

        prediction_records.append(PredictionRecord(
            id=item.id,
            prompt=item.prompt,
            expected_risk=item.expected_risk,
            prediction=result.prediction,
            confidence=result.confidence,
            temporal_debt=result.temporal_debt,
            anomaly_score=anomaly_score,
            features=features,
            guard_flags=flags,
            notes=label,
            invariant_results=inv_results_dict,
            n_violated=n_violated,
            aggregate_score=aggregate_score,
            fail_type=getattr(result, 'fail_type', None),
            fail_subjects=getattr(result, 'fail_subjects', None),
            expected_min_anchors=item.expected_min_anchors,
        ))

        if n_items % 10 == 0:
            print(f"    [{label}] {n_items} prompts done...")

    return prediction_records, started_at


def run_lane(detector, config, lane_name, corpus_path, profile_name, label, results):
    """Run a single lane and store the result."""
    print(f"\n{'='*60}")
    print(f"LANE: {lane_name} ({label})")
    print(f"  corpus: {corpus_path}")
    print(f"{'='*60}")

    t0 = time.time()
    preds, started = run_one_eval(detector, corpus_path, profile_name, label=label)
    rd = store_run(
        preds, corpus_path, config.model_name if hasattr(config, 'model_name') else MODEL,
        profile_name, device=config.device, started_at=started, multi_invariant=True,
    )
    elapsed = time.time() - t0
    results.append({"lane": lane_name, "label": label, "run_dir": rd, "elapsed": elapsed,
                     "corpus": corpus_path, "n_items": len(preds)})
    print(f"  Stored: {rd}  ({elapsed:.0f}s, {len(preds)} items)")
    return rd


def run_replay(new_run_dirs, results):
    """Run CPU-only replay scan on new runs."""
    print(f"\n{'='*60}")
    print("LANE: replay (CPU-only threshold scan)")
    print(f"{'='*60}")

    replay_script = Path(__file__).parent / "replay.py"
    if not replay_script.exists():
        print("  replay.py not found, skipping")
        return

    for rd in new_run_dirs:
        run_id = rd.name.split("_")[0]
        print(f"\n  Replaying {run_id}...")
        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(replay_script), "--run", str(rd)],
                capture_output=True, text=True, timeout=300,
            )
            elapsed = time.time() - t0
            output = result.stdout.strip()
            if output:
                # Just print the Pareto lines
                lines = output.split("\n")
                pareto = [l for l in lines if "Pareto" in l or "accel=" in l or "HR_recall" in l]
                for l in pareto[:5]:
                    print(f"    {l}")
            results.append({"lane": "replay", "label": f"replay_{run_id}",
                           "run_dir": rd, "elapsed": elapsed})
            print(f"  ({elapsed:.0f}s)")
        except subprocess.TimeoutExpired:
            print(f"  replay timed out for {run_id}")
        except Exception as e:
            print(f"  replay failed: {e}")


def _write_two_axis_row(lines, label, preds):
    """Compute fabrication + evasion axes for a set of predictions, append table row."""
    total_anchors = 0
    valid_anchors = 0
    fmt_invalid = 0
    resolve_invalid = 0
    zero_anchor_prompts = 0
    evasion_prompts = 0
    abstention_prompts = 0

    for p in preds:
        eg = (p.get("invariant_results") or {}).get("epistemic_grounding", {}).get("details", {})
        tc = eg.get("total_citations", 0)
        total_anchors += tc
        valid_anchors += eg.get("valid_count", 0)
        fmt_invalid += eg.get("format_invalid", 0)
        resolve_invalid += eg.get("resolve_invalid", 0)

        if tc == 0:
            zero_anchor_prompts += 1

        expected_min = p.get("expected_min_anchors")
        if expected_min is not None and tc < expected_min:
            evasion_prompts += 1

        # Abstention detection requires stored generation text (not yet available)

    n = len(preds)
    inv = fmt_invalid + resolve_invalid
    # Fall back to total_anchors - valid_anchors when split not available (old runs)
    if inv == 0 and total_anchors > valid_anchors:
        inv = total_anchors - valid_anchors
    fab_rate = f"{inv/total_anchors:.0%}" if total_anchors > 0 else "N/A"
    zero_rate = f"{zero_anchor_prompts}/{n}"
    evasion_rate = f"{evasion_prompts}/{n}"

    lines.append(
        f"| {label} | {total_anchors} | {valid_anchors} | {fmt_invalid} | {resolve_invalid} "
        f"| {fab_rate} | {zero_rate} | {evasion_rate} | {abstention_prompts}/{n} |"
    )


def generate_report(results, stamp, report_dir):
    """Generate a nightly.md report summarizing all lanes."""
    report_path = Path(report_dir) / f"nightly-{stamp}.md"
    lines = [
        f"# Nightly Report {stamp}",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Run Inventory",
        "",
        "| Lane | Label | Items | Time | Run ID |",
        "|------|-------|-------|------|--------|",
    ]

    for r in results:
        if r["lane"] == "replay":
            continue
        rd = r["run_dir"]
        run_id = rd.name.split("_")[0] if hasattr(rd, 'name') else str(rd)
        lines.append(f"| {r['lane']} | {r['label']} | {r.get('n_items','?')} | {r['elapsed']:.0f}s | {run_id} |")

    lines.extend(["", "## Tier Counts", ""])
    lines.append("| Lane | CLEAN | WARN | FAIL | HR Recall | LR Precision |")
    lines.append("|------|-------|------|------|-----------|--------------|")

    for r in results:
        if r["lane"] == "replay":
            continue
        rd = r["run_dir"]
        try:
            summary = json.loads((rd / "summary.json").read_text())
            pc = summary.get("prediction_counts", {})
            hr = summary.get("high_risk_recall", "N/A")
            lp = summary.get("low_risk_precision", "N/A")
            if isinstance(hr, float):
                hr = f"{hr:.3f}"
            if isinstance(lp, float):
                lp = f"{lp:.3f}"
            lines.append(f"| {r['lane']} | {pc.get('CLEAN',0)} | {pc.get('WARN',0)} | {pc.get('FAIL',0)} | {hr} | {lp} |")
        except Exception:
            lines.append(f"| {r['lane']} | ? | ? | ? | ? | ? |")

    # Invariant violation counts
    lines.extend(["", "## Invariant Violations", ""])
    lines.append("| Lane | TC | SC | EG |")
    lines.append("|------|----|----|----|")

    for r in results:
        if r["lane"] == "replay":
            continue
        rd = r["run_dir"]
        try:
            summary = json.loads((rd / "summary.json").read_text())
            inv = summary.get("invariant_violation_counts", {})
            tc = inv.get("temporal_coherence", 0)
            sc = inv.get("semantic_conservation", 0)
            eg = inv.get("epistemic_grounding", 0)
            lines.append(f"| {r['lane']} | {tc} | {sc} | {eg} |")
        except Exception:
            lines.append(f"| {r['lane']} | ? | ? | ? |")

    # FAILs detail
    lines.extend(["", "## FAIL Details", ""])
    for r in results:
        if r["lane"] == "replay":
            continue
        rd = r["run_dir"]
        try:
            preds = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().strip().split("\n")]
            fails = [p for p in preds if p["prediction"] == "FAIL"]
            if fails:
                lines.append(f"### {r['lane']} ({r['label']})")
                for p in fails:
                    ft = p.get('fail_type', '?')
                    fs = p.get('fail_subjects') or []
                    subj_str = ', '.join(fs[:3]) if fs else 'none'
                    lines.append(f"- **{p['id']}** [{p['expected_risk']}] type={ft} subjects=[{subj_str}] n_violated={p.get('n_violated','?')}")
                lines.append("")
        except Exception:
            pass

    # Two-axis report: Fabrication + Evasion
    lines.extend(["", "## Two-Axis Report (Fabrication + Evasion)", ""])
    lines.append("| Lane | Anchors | Valid | Fmt-Inv | Resolve-Inv | Fab Rate | Zero-Anchor | Evasion | Abstain |")
    lines.append("|------|---------|-------|---------|-------------|----------|-------------|---------|---------|")

    for r in results:
        if r["lane"] in ("replay", "sweep"):
            continue
        rd = r["run_dir"]
        try:
            preds = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().strip().split("\n")]
            _write_two_axis_row(lines, r["lane"], preds)
        except Exception as e:
            lines.append(f"| {r['lane']} | ? | ? | ? | ? | ? | ? | ? | ? |")

    # Per-level breakdown for ladder
    for r in results:
        if r["lane"] != "ladder":
            continue
        rd = r["run_dir"]
        try:
            preds = [json.loads(l) for l in (rd / "predictions.jsonl").read_text().strip().split("\n")]
            levels = {}
            for p in preds:
                level = p["id"].split("-")[1] if "-" in p["id"] else "?"
                levels.setdefault(level, []).append(p)
            lines.extend(["", "### Ladder Per-Level Breakdown", ""])
            lines.append("| Level | Anchors | Valid | Fmt-Inv | Resolve-Inv | Fab Rate | Zero-Anchor | Evasion | Abstain |")
            lines.append("|-------|---------|-------|---------|-------------|----------|-------------|---------|---------|")
            for level in sorted(levels):
                _write_two_axis_row(lines, level, levels[level])
        except Exception:
            pass

    report_text = "\n".join(lines) + "\n"
    report_path.write_text(report_text)
    print(f"\nReport written: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Overnight eval harness (lane-based)")
    parser.add_argument("--lanes", default="all",
                        help="Comma-separated lanes to run: canary,baseline,citations,sweep,replay (or 'all')")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--profile", default=PROFILE_NAME)
    # Legacy compat
    parser.add_argument("--canary-only", action="store_true", help="(legacy) same as --lanes canary")
    parser.add_argument("--sweep-only", action="store_true", help="(legacy) same as --lanes sweep")
    parser.add_argument("--corpus", default=None, help="Override corpus for canary/baseline lane")
    args = parser.parse_args()

    # Parse lanes
    if args.canary_only:
        lanes = ["canary"]
    elif args.sweep_only:
        lanes = ["sweep"]
    elif args.lanes == "all":
        lanes = ALL_LANES
    else:
        lanes = [l.strip() for l in args.lanes.split(",")]

    # Corpus overrides
    corpora = dict(CORPORA)
    if args.corpus:
        corpora["canary"] = args.corpus
        corpora["baseline"] = args.corpus

    # Load model once (skip if only running replay)
    detector = None
    config = DetectorConfig()
    config.device = args.device

    gpu_lanes = [l for l in lanes if l != "replay"]
    if gpu_lanes:
        print(f"Loading model: {args.model} on {args.device}")
        detector = DeltaTDetector(model_name=args.model, config=config)
        detector.set_risk_profile(args.profile)

    results = []
    new_run_dirs = []
    wall_start = time.time()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # --- Lane 0: Canary ---
    if "canary" in lanes and detector:
        rd = run_lane(detector, config, "canary", corpora["canary"],
                      args.profile, "canary", results)
        new_run_dirs.append(rd)

    # --- Lane 1: Baseline ---
    if "baseline" in lanes and detector:
        rd = run_lane(detector, config, "baseline", corpora["baseline"],
                      args.profile, "baseline_nightly", results)
        new_run_dirs.append(rd)

    # --- Lane 2: Citations ---
    if "citations" in lanes and detector:
        corpus = corpora.get("citations")
        if corpus and Path(corpus).exists():
            rd = run_lane(detector, config, "citations", corpus,
                          args.profile, "citations_nightly", results)
            new_run_dirs.append(rd)
        else:
            print(f"\nSkipping citations lane: {corpus} not found")

    # --- Lane 3: Citation Ladder ---
    if "ladder" in lanes and detector:
        corpus = corpora.get("ladder")
        if corpus and Path(corpus).exists():
            rd = run_lane(detector, config, "ladder", corpus,
                          args.profile, "ladder_nightly", results)
            new_run_dirs.append(rd)
        else:
            print(f"\nSkipping ladder lane: {corpus} not found")

    # --- Lane 4: Sweep ---
    if "sweep" in lanes and detector:
        original_temp_base = config.temperature_base
        original_temp_step = config.temperature_step
        profile = RISK_PROFILES[args.profile]
        original_n_perturb = profile.n_perturbations

        for temp_base in TEMP_GRID:
            for n_perturb in NPERTURB_GRID:
                label = f"sweep_t{temp_base}_n{n_perturb}"
                config.temperature_base = temp_base
                profile.n_perturbations = n_perturb

                rd = run_lane(detector, config, "sweep", corpora["baseline"],
                              args.profile, label, results)
                new_run_dirs.append(rd)

        config.temperature_base = original_temp_base
        config.temperature_step = original_temp_step
        profile.n_perturbations = original_n_perturb

    # --- Lane 4: Replay ---
    if "replay" in lanes and new_run_dirs:
        run_replay(new_run_dirs, results)

    # --- Summary ---
    wall_total = time.time() - wall_start
    print(f"\n\n{'='*60}")
    print(f"OVERNIGHT SUMMARY  ({wall_total:.0f}s total, {len(results)} runs)")
    print(f"{'='*60}")

    for r in results:
        if r["lane"] == "replay":
            continue
        rd = r["run_dir"]
        try:
            summary = json.loads((rd / "summary.json").read_text())
            pc = summary["prediction_counts"]
            inv_v = summary.get("invariant_violation_counts", {})
            print(f"\n  {r['label']} ({r['elapsed']:.0f}s, {r.get('n_items','?')} items)")
            print(f"    Predictions: {pc}")
            hr = summary.get('high_risk_recall', 'N/A')
            lp = summary.get('low_risk_precision', 'N/A')
            print(f"    HR recall: {hr}  |  LR precision: {lp}")
            if inv_v:
                print(f"    Invariant violations: {inv_v}")
        except Exception as e:
            print(f"\n  {r['label']} ({r['elapsed']:.0f}s) — error reading summary: {e}")

    # Generate report
    report_dir = Path("reports")
    report_dir.mkdir(exist_ok=True)
    try:
        report_path = generate_report(results, stamp, report_dir)
    except Exception as e:
        print(f"\nFailed to generate report: {e}")


if __name__ == "__main__":
    main()
