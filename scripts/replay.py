#!/usr/bin/env python3
"""
CPU-only threshold replay: re-score stored predictions with different thresholds.

No GPU needed — reads features from a stored run and recomputes
TC/SC violation flags + WARN/FAIL/CLEAN tiers.

Usage:
    # Scan a grid of thresholds against the latest run:
    python3 scripts/replay.py --run <run_id_prefix>

    # Specify custom grids:
    python3 scripts/replay.py --run <run_id> \
        --accel-grid 0.25 0.30 0.35 0.40 0.45 \
        --tokens-grid 0 2 4 6 8 \
        --slope-grid 0.30 0.35 0.40 \
        --min-violated 1 2
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.run_store import load_run

# Default grids (coarse but useful)
DEFAULT_ACCEL_GRID = [0.25, 0.30, 0.35, 0.40, 0.45]
DEFAULT_TOKENS_GRID = [0, 2, 4, 6, 8]
DEFAULT_SLOPE_GRID = [0.30, 0.35, 0.40]
DEFAULT_MIN_VIOLATED = [1, 2]


def resolve_run_dir(prefix: str, runs_root: str = "runs") -> Optional[Path]:
    """Find a run directory by prefix match."""
    root = Path(runs_root)
    if not root.is_dir():
        return None
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and entry.name.startswith(prefix):
            if (entry / "manifest.json").exists():
                return entry
    return None


def replay_tc(features: Dict[str, Any], slope_thresh: float,
              accel_thresh: float, tokens_thresh: int) -> bool:
    """Replay TC violation check from stored features (no model needed)."""
    slope = features.get("max_confidence_slope", 0)
    accel = features.get("confidence_acceleration", 0)
    tokens = features.get("tokens_to_high_conf", 10)

    if slope > slope_thresh:
        return True
    if accel > accel_thresh:
        return True
    if tokens_thresh > 0 and tokens < tokens_thresh:
        return True
    return False


def replay_sc(inv_results: Optional[Dict], sc_threshold: float) -> bool:
    """Replay SC violation check from stored invariant results."""
    if not inv_results:
        return False
    sc = inv_results.get("semantic_conservation")
    if not sc:
        return False
    details = sc.get("details", {})
    min_sim = details.get("min_similarity")
    if min_sim is not None and min_sim < sc_threshold:
        return True
    return False


def replay_one(predictions: List[Dict], slope: float, accel: float,
               tokens: int, sc_threshold: float,
               min_violated: int) -> Dict[str, Any]:
    """Replay all predictions with given thresholds, return summary."""
    tier_counts = {"FAIL": 0, "WARN": 0, "CLEAN": 0}
    by_expected = {}  # {expected_risk: {tier: count}}

    for pred in predictions:
        features = pred.get("features", {})
        inv_results = pred.get("invariant_results")
        expected = pred.get("expected_risk", "?")

        tc_violated = replay_tc(features, slope, accel, tokens)
        sc_violated = replay_sc(inv_results, sc_threshold)
        n_violated = int(tc_violated) + int(sc_violated)

        if n_violated >= min_violated:
            tier = "FAIL"
        elif n_violated > 0:
            tier = "WARN"
        else:
            tier = "CLEAN"

        tier_counts[tier] += 1
        bucket = by_expected.setdefault(expected, {"FAIL": 0, "WARN": 0, "CLEAN": 0})
        bucket[tier] += 1

    # Compute high-risk recall (FAIL on high-risk items)
    hr_total = sum(v for k, v in by_expected.get("high", {}).items())
    hr_caught = by_expected.get("high", {}).get("FAIL", 0)
    hr_recall = hr_caught / hr_total if hr_total > 0 else None

    # Compute controls FP (FAIL on low-risk items)
    lr_total = sum(v for k, v in by_expected.get("low", {}).items())
    lr_fp = by_expected.get("low", {}).get("FAIL", 0)
    lr_fp_rate = lr_fp / lr_total if lr_total > 0 else None

    return {
        "tier_counts": tier_counts,
        "by_expected": by_expected,
        "high_risk_recall": hr_recall,
        "low_risk_fp_rate": lr_fp_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="CPU-only threshold replay")
    parser.add_argument("--run", required=True, help="Run ID prefix")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--accel-grid", type=float, nargs="+", default=DEFAULT_ACCEL_GRID)
    parser.add_argument("--tokens-grid", type=int, nargs="+", default=DEFAULT_TOKENS_GRID)
    parser.add_argument("--slope-grid", type=float, nargs="+", default=DEFAULT_SLOPE_GRID)
    parser.add_argument("--sc-threshold", type=float, default=0.70,
                        help="SC similarity threshold (fixed for replay)")
    parser.add_argument("--min-violated", type=int, nargs="+", default=DEFAULT_MIN_VIOLATED)
    parser.add_argument("--output", type=str, default=None,
                        help="Write results JSON to file")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run, args.runs_root)
    if run_dir is None:
        print(f"No run matching '{args.run}' found in {args.runs_root}/", file=sys.stderr)
        sys.exit(1)

    data = load_run(str(run_dir))
    predictions = data["predictions"]
    manifest = data["manifest"]
    print(f"Replaying run {manifest['run_id']} ({len(predictions)} items)")
    print(f"  Model: {manifest['model']}")
    print(f"  Profile: {manifest['profile']}")
    print()

    # Header
    print(f"{'SLOPE':>6} {'ACCEL':>6} {'TOKENS':>7} {'MIN_V':>6}  "
          f"{'FAIL':>5} {'WARN':>5} {'CLEAN':>5}  "
          f"{'HR_RECALL':>10} {'LR_FP':>8}")
    print("-" * 75)

    all_results = []

    for min_v in args.min_violated:
        for slope in args.slope_grid:
            for accel in args.accel_grid:
                for tokens in args.tokens_grid:
                    r = replay_one(predictions, slope, accel, tokens,
                                   args.sc_threshold, min_v)
                    tc = r["tier_counts"]
                    hr = f"{r['high_risk_recall']:.1%}" if r['high_risk_recall'] is not None else "N/A"
                    fp = f"{r['low_risk_fp_rate']:.1%}" if r['low_risk_fp_rate'] is not None else "N/A"

                    print(f"{slope:>6.2f} {accel:>6.2f} {tokens:>7d} {min_v:>6d}  "
                          f"{tc['FAIL']:>5d} {tc['WARN']:>5d} {tc['CLEAN']:>5d}  "
                          f"{hr:>10s} {fp:>8s}")

                    all_results.append({
                        "slope": slope, "accel": accel, "tokens": tokens,
                        "min_violated": min_v, "sc_threshold": args.sc_threshold,
                        **r,
                    })

    # Find Pareto-dominant configs (maximize HR recall, minimize LR FP)
    print(f"\n{'='*75}")
    print("PARETO CANDIDATES (HR recall > 0, LR FP = 0)")
    print(f"{'='*75}")
    pareto = [r for r in all_results
              if r["high_risk_recall"] is not None
              and r["high_risk_recall"] > 0
              and (r["low_risk_fp_rate"] is None or r["low_risk_fp_rate"] == 0)]
    if pareto:
        pareto.sort(key=lambda r: r["high_risk_recall"], reverse=True)
        for r in pareto[:10]:
            tc = r["tier_counts"]
            print(f"  slope={r['slope']:.2f} accel={r['accel']:.2f} "
                  f"tokens={r['tokens']} min_v={r['min_violated']}  "
                  f"HR={r['high_risk_recall']:.1%}  "
                  f"FAIL={tc['FAIL']} WARN={tc['WARN']} CLEAN={tc['CLEAN']}")
    else:
        print("  (none found — try relaxing min_violated to 1)")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
