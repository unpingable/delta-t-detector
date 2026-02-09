#!/usr/bin/env python3
"""
Flapper report: identify prompts whose prediction tier changes across runs.

Flappers are prompts in "unstable epistemic territory" — they're prime
candidates for Inv-2 (existence checks) because the detector can't
decide based on temporal/semantic signals alone.

Usage:
    python3 scripts/flappers.py                        # compare all runs
    python3 scripts/flappers.py --runs <id1> <id2> ... # specific runs
    python3 scripts/flappers.py --latest 3             # last N runs
"""

import argparse
import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.run_store import list_runs, load_run

# Normalize prediction labels across naming conventions
# so that pre-tier and post-tier runs are comparable.
_LABEL_MAP = {
    "temporal_stable": "CLEAN",
    "truthful": "CLEAN",
    "CLEAN": "CLEAN",
    "temporal_unstable": "FAIL",
    "hallucination": "FAIL",
    "FAIL": "FAIL",
    "WARN": "WARN",
    "uncertain": "uncertain",
    "unknown": "unknown",
}


def _normalize_label(label: str) -> str:
    return _LABEL_MAP.get(label, label)


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


def find_all_run_dirs(runs_root: str = "runs") -> List[Path]:
    """Find all completed run directories, sorted by timestamp."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    dirs = []
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and (entry / "manifest.json").exists():
            dirs.append(entry)
    return dirs


def main():
    parser = argparse.ArgumentParser(description="Flapper report")
    parser.add_argument("--runs", nargs="+", help="Run ID prefixes to compare")
    parser.add_argument("--latest", type=int, default=0,
                        help="Compare the latest N runs")
    parser.add_argument("--runs-root", default="runs")
    args = parser.parse_args()

    # Resolve run directories
    if args.runs:
        run_dirs = []
        for prefix in args.runs:
            rd = resolve_run_dir(prefix, args.runs_root)
            if rd is None:
                print(f"No run matching '{prefix}' found.", file=sys.stderr)
                sys.exit(1)
            run_dirs.append(rd)
    elif args.latest > 0:
        all_dirs = find_all_run_dirs(args.runs_root)
        run_dirs = all_dirs[-args.latest:]
    else:
        run_dirs = find_all_run_dirs(args.runs_root)

    if len(run_dirs) < 2:
        print("Need at least 2 runs to compare.", file=sys.stderr)
        sys.exit(1)

    # Load predictions from each run
    run_data = []
    for rd in run_dirs:
        data = load_run(str(rd))
        manifest = data["manifest"]
        preds_by_id = {p["id"]: p for p in data["predictions"]}
        run_data.append({
            "dir": rd,
            "run_id": manifest["run_id"],
            "timestamp": manifest["finished_at"],
            "notes": data["predictions"][0].get("notes", "") if data["predictions"] else "",
            "preds": preds_by_id,
        })

    # Collect all prompt IDs
    all_ids = set()
    for rd in run_data:
        all_ids.update(rd["preds"].keys())

    # For each prompt, collect its normalized prediction across runs
    prompt_tiers: Dict[str, List[str]] = {}
    prompt_raw: Dict[str, List[str]] = {}  # original labels for display
    prompt_meta: Dict[str, Dict] = {}
    for pid in sorted(all_ids):
        tiers = []
        raw = []
        for rd in run_data:
            pred = rd["preds"].get(pid)
            if pred:
                label = pred.get("prediction", "?")
                raw.append(label)
                tiers.append(_normalize_label(label))
                if pid not in prompt_meta:
                    prompt_meta[pid] = {
                        "expected_risk": pred.get("expected_risk", "?"),
                        "prompt": pred.get("prompt", "")[:80],
                    }
            else:
                tiers.append("MISSING")
                raw.append("MISSING")
        prompt_tiers[pid] = tiers
        prompt_raw[pid] = raw

    # Identify flappers: prompts where the normalized tier set has more than one unique value
    flappers = {}
    for pid, tiers in prompt_tiers.items():
        unique = set(t for t in tiers if t != "MISSING")
        if len(unique) > 1:
            flappers[pid] = {
                "tiers": tiers,
                "unique_tiers": sorted(unique),
                "n_flips": len(unique),
                **prompt_meta.get(pid, {}),
            }

    # Report
    print(f"FLAPPER REPORT")
    print(f"{'='*70}")
    print(f"Comparing {len(run_data)} runs, {len(all_ids)} unique prompts\n")

    # Run legend
    print("Runs:")
    for i, rd in enumerate(run_data):
        print(f"  [{i}] {rd['run_id']} @ {rd['timestamp']}  {rd['notes']}")
    print()

    if not flappers:
        print("No flappers found — all prompts have stable tiers across runs.")
        return

    # Sort by number of unique tiers (most unstable first), then by ID
    sorted_flappers = sorted(flappers.items(),
                              key=lambda x: (-x[1]["n_flips"], x[0]))

    print(f"{'ID':<12} {'EXPECTED':>8}  TIERS{'':>20}  PROMPT")
    print("-" * 90)

    for pid, info in sorted_flappers:
        tier_str = " → ".join(info["tiers"])
        prompt_snip = info.get("prompt", "")[:40]
        print(f"{pid:<12} {info.get('expected_risk', '?'):>8}  {tier_str:<30}  {prompt_snip}")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  Total prompts: {len(all_ids)}")
    print(f"  Flappers: {len(flappers)} ({100*len(flappers)/len(all_ids):.0f}%)")
    print(f"  Stable: {len(all_ids) - len(flappers)}")

    # By expected risk
    flapper_by_risk = defaultdict(int)
    for info in flappers.values():
        flapper_by_risk[info.get("expected_risk", "?")] += 1
    print(f"\n  Flappers by expected risk:")
    for risk, count in sorted(flapper_by_risk.items()):
        print(f"    {risk}: {count}")

    print(f"\n  These prompts are in unstable epistemic territory —")
    print(f"  prime candidates for Inv-2 (existence checks).")


if __name__ == "__main__":
    main()
