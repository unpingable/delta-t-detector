#!/usr/bin/env python3
"""
Overnight drift check + resolver health audit.

Option A: Stability/regression check
  - 3 models × 2 locked lanes × 2 temps × 3 seeds = 36 coupling runs
  - Phi-3 two-stage controller: 2 lanes × 3 seeds = 6 retry runs
  Total: 42 runs
  Pass/fail gate: deltas within ±5pp across seeds → stable

Option B: Resolver health audit
  - Parse all run artifacts for resolver metadata
  - Report error rate, status distributions per endpoint

Usage:
    .venv/bin/python scripts/overnight_drift.py
    .venv/bin/python scripts/overnight_drift.py --models qwen3b
    .venv/bin/python scripts/overnight_drift.py --skip-retry
    .venv/bin/python scripts/overnight_drift.py --seeds 42 137
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = str(ROOT / ".venv" / "bin" / "python")

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = {
    "qwen3b": {
        "name": "Qwen/Qwen2.5-3B-Instruct",
        "quantize": None,
        "label": "Qwen-3B",
    },
    "qwen7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "quantize": "4bit",
        "label": "Qwen-7B-4bit",
    },
    "phi3": {
        "name": "microsoft/Phi-3-mini-4k-instruct",
        "quantize": None,
        "label": "Phi-3-Mini",
    },
}

LOCKED_LANES = {
    "pypi_locked": "data/canonical_10_pypi_locked.jsonl",
    "cve_locked": "data/canonical_10_cve_locked.jsonl",
}

# Additional lanes for resolver health (Option B)
RESOLVER_LANES = {
    "rfc": "data/canonical_15_rfc.jsonl",
    "citation_forcing": "data/citation_forcing_30.jsonl",
}

SEEDS = [42, 137, 271]
TEMPS = [0.7, 0.0]

# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run_coupling(model_key, lane, temp, seed, device="cuda"):
    """Run coupling.py and return (success, elapsed_sec, report_path)."""
    m = MODELS[model_key]
    cmd = [
        VENV_PYTHON, "scripts/coupling.py",
        "--model", m["name"],
        "--device", device,
        "--corpus", lane,
        "--topologies", "single",
        "--temperature", str(temp),
        "--seed", str(seed),
    ]
    if m["quantize"]:
        cmd += ["--quantize", m["quantize"]]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.monotonic() - t0

    # Extract report path from output
    report_path = None
    for line in result.stdout.splitlines():
        if "Report:" in line:
            report_path = line.split("Report:")[-1].strip()

    return result.returncode == 0, elapsed, report_path, result.stdout, result.stderr


def run_retry(model_key, lane, seed, device="cuda", retry_temp=0.0):
    """Run retry_enforcement.py with two-stage policy."""
    m = MODELS[model_key]
    cmd = [
        VENV_PYTHON, "scripts/retry_enforcement.py",
        "--model", m["name"],
        "--device", device,
        "--corpus", lane,
        "--temperature", "0.7",
        "--retry-temperature", str(retry_temp),
        "--seed", str(seed),
    ]
    if m["quantize"]:
        cmd += ["--quantize", m["quantize"]]

    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    elapsed = time.monotonic() - t0

    report_path = None
    for line in result.stdout.splitlines():
        if "Report:" in line:
            report_path = line.split("Report:")[-1].strip()

    return result.returncode == 0, elapsed, report_path, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def parse_coupling_output(stdout):
    """Extract fabrication rate from coupling.py stdout."""
    for line in stdout.splitlines():
        if line.strip().startswith("single"):
            parts = line.split()
            # Format: single  anchors_total  anchors_valid  fab%  mismatch%  lie%  evasion%
            for p in parts:
                if p.endswith("%") and p != "0.0%":
                    pass
            # Find the fabrication_rate column (4th field)
            try:
                fab_str = parts[3]  # e.g. "10.0%"
                return float(fab_str.rstrip("%"))
            except (IndexError, ValueError):
                pass
    return None


def parse_retry_output(stdout):
    """Extract retry summary from retry_enforcement.py stdout."""
    info = {
        "retries": 0,
        "resolved_clean": 0,
        "converted_to_fail": 0,
        "persistent_evasion": 0,
        "abstained": 0,
        "harm_rate": None,
        "benefit_rate": None,
    }
    for line in stdout.splitlines():
        line = line.strip()
        if "Retry triggered:" in line:
            try:
                info["retries"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "resolved_clean:" in line:
            try:
                info["resolved_clean"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "converted_to_fail:" in line:
            try:
                info["converted_to_fail"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "persistent_evasion:" in line:
            try:
                info["persistent_evasion"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "abstained" in line and "UNKNOWN" in line:
            try:
                info["abstained"] = int(line.split(":")[-1].strip())
            except ValueError:
                pass
        elif "Intervention harm rate:" in line:
            try:
                info["harm_rate"] = line.split(":")[-1].strip()
            except (ValueError, IndexError):
                pass
        elif "Intervention benefit rate:" in line:
            try:
                info["benefit_rate"] = line.split(":")[-1].strip()
            except (ValueError, IndexError):
                pass
    return info


# ---------------------------------------------------------------------------
# Resolver health audit (Option B)
# ---------------------------------------------------------------------------

def audit_resolvers():
    """Scan all run artifacts for resolver metadata and aggregate stats.

    Validation results are stored as:
      invariant_results.epistemic_grounding.details.validation_results
    which is a dict mapping anchor → {valid, reason, resolver, response_class, timestamp}.
    """
    runs_dir = ROOT / "runs"
    if not runs_dir.exists():
        print("  No runs/ directory found, skipping resolver audit.")
        return {}

    stats = defaultdict(lambda: defaultdict(int))
    status_codes = defaultdict(int)
    total_anchors = 0
    total_errors = 0

    for run_dir in sorted(runs_dir.iterdir()):
        pred_file = run_dir / "predictions.jsonl"
        if not pred_file.exists():
            continue
        with open(pred_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                inv = rec.get("invariant_results") or {}
                eg = inv.get("epistemic_grounding") or {}
                details = eg.get("details") or {}
                vr = details.get("validation_results") or {}

                if not isinstance(vr, dict):
                    continue

                for anchor, meta in vr.items():
                    if not isinstance(meta, dict):
                        continue
                    resolver = meta.get("resolver", "unknown")
                    resp_class = meta.get("response_class", "unknown")
                    reason = meta.get("reason", "")
                    total_anchors += 1
                    stats[resolver][resp_class] += 1

                    # Track HTTP status codes from reason field
                    if reason.startswith("HTTP "):
                        status_codes[reason] += 1
                    elif "error" in reason.lower() or "timeout" in reason.lower():
                        total_errors += 1
                        status_codes[reason] += 1

                    if resp_class in ("error", "timeout"):
                        total_errors += 1

    return {
        "total_anchors_resolved": total_anchors,
        "total_errors": total_errors,
        "error_rate": f"{total_errors/total_anchors:.1%}" if total_anchors > 0 else "n/a",
        "by_resolver": {r: dict(v) for r, v in sorted(stats.items())},
        "status_distribution": dict(sorted(status_codes.items(),
                                           key=lambda x: -x[1])),
    }


# ---------------------------------------------------------------------------
# Drift analysis
# ---------------------------------------------------------------------------

def analyze_drift(results):
    """Analyze fabrication rate drift across seeds. Returns (stable, report)."""
    # Group by (model, lane, temp)
    groups = defaultdict(list)
    for r in results:
        key = (r["model"], r["lane"], r["temp"])
        if r["fab_rate"] is not None:
            groups[key].append(r["fab_rate"])

    report_lines = []
    all_stable = True
    for key, rates in sorted(groups.items()):
        model, lane, temp = key
        if len(rates) < 2:
            continue
        min_r, max_r = min(rates), max(rates)
        spread = max_r - min_r
        stable = spread <= 5.0
        if not stable:
            all_stable = False
        status = "STABLE" if stable else "DRIFT"
        report_lines.append(
            f"  {status}  {model:20s} {lane:15s} t={temp}  "
            f"rates={[f'{r:.1f}%' for r in rates]}  spread={spread:.1f}pp"
        )

    return all_stable, "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Overnight drift check + resolver audit")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Models to test (default: all)")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS,
                        help="Seeds for repeats (default: 42 137 271)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-retry", action="store_true",
                        help="Skip retry enforcement runs")
    parser.add_argument("--skip-resolver", action="store_true",
                        help="Skip resolver-heavy lanes")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"{'='*70}")
    print(f"OVERNIGHT DRIFT CHECK + RESOLVER AUDIT: {stamp}")
    print(f"{'='*70}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Seeds:  {args.seeds}")
    print(f"  Temps:  {TEMPS}")
    print(f"  Lanes:  {list(LOCKED_LANES.keys())}")
    print()

    coupling_results = []
    retry_results = []
    total_runs = 0
    total_failures = 0
    t_start = time.monotonic()

    # -----------------------------------------------------------------------
    # Option A: Drift check — coupling.py runs
    # -----------------------------------------------------------------------
    for model_key in args.models:
        m = MODELS[model_key]
        print(f"\n{'='*70}")
        print(f"MODEL: {m['label']}")
        print(f"{'='*70}")

        for lane_name, lane_path in LOCKED_LANES.items():
            for temp in TEMPS:
                for seed in args.seeds:
                    total_runs += 1
                    label = f"{m['label']} | {lane_name} | t={temp} | seed={seed}"
                    print(f"\n  [{total_runs}] {label} ... ", end="", flush=True)

                    ok, elapsed, report, stdout, stderr = run_coupling(
                        model_key, lane_path, temp, seed, args.device
                    )

                    fab_rate = parse_coupling_output(stdout) if ok else None

                    if ok:
                        fab_str = f"{fab_rate:.1f}%" if fab_rate is not None else "?"
                        print(f"OK  fab={fab_str}  ({elapsed:.0f}s)")
                    else:
                        total_failures += 1
                        print(f"FAIL ({elapsed:.0f}s)")
                        # Print last few lines of stderr for debugging
                        for line in (stderr or "").strip().splitlines()[-3:]:
                            print(f"    ! {line}")

                    coupling_results.append({
                        "model": m["label"],
                        "model_key": model_key,
                        "lane": lane_name,
                        "temp": temp,
                        "seed": seed,
                        "fab_rate": fab_rate,
                        "ok": ok,
                        "elapsed": elapsed,
                        "report": report,
                    })

        # ---------------------------------------------------------------
        # Option A: Two-stage retry (Phi-3 only, or any model with evasion)
        # ---------------------------------------------------------------
        if not args.skip_retry:
            for lane_name, lane_path in LOCKED_LANES.items():
                for seed in args.seeds:
                    total_runs += 1
                    label = f"{m['label']} | {lane_name} | two-stage | seed={seed}"
                    print(f"\n  [{total_runs}] {label} ... ", end="", flush=True)

                    ok, elapsed, report, stdout, stderr = run_retry(
                        model_key, lane_path, seed, args.device
                    )

                    info = parse_retry_output(stdout) if ok else {}

                    if ok:
                        print(f"OK  retries={info.get('retries', '?')}  "
                              f"clean={info.get('resolved_clean', 0)}  "
                              f"fail={info.get('converted_to_fail', 0)}  "
                              f"({elapsed:.0f}s)")
                    else:
                        total_failures += 1
                        print(f"FAIL ({elapsed:.0f}s)")
                        for line in (stderr or "").strip().splitlines()[-3:]:
                            print(f"    ! {line}")

                    retry_results.append({
                        "model": m["label"],
                        "model_key": model_key,
                        "lane": lane_name,
                        "seed": seed,
                        "ok": ok,
                        "elapsed": elapsed,
                        **info,
                    })

    # -----------------------------------------------------------------------
    # Option B: Resolver-heavy lanes (one seed, one model = Qwen 3B)
    # -----------------------------------------------------------------------
    if not args.skip_resolver and "qwen3b" in args.models:
        print(f"\n{'='*70}")
        print("RESOLVER HEALTH LANES")
        print(f"{'='*70}")
        for lane_name, lane_path in RESOLVER_LANES.items():
            total_runs += 1
            label = f"Qwen-3B | {lane_name} | t=0.7 | seed=42"
            print(f"\n  [{total_runs}] {label} ... ", end="", flush=True)

            ok, elapsed, report, stdout, stderr = run_coupling(
                "qwen3b", lane_path, 0.7, 42, args.device
            )
            if ok:
                fab_rate = parse_coupling_output(stdout)
                fab_str = f"{fab_rate:.1f}%" if fab_rate is not None else "?"
                print(f"OK  fab={fab_str}  ({elapsed:.0f}s)")
            else:
                total_failures += 1
                print(f"FAIL ({elapsed:.0f}s)")

    total_elapsed = time.monotonic() - t_start

    # -----------------------------------------------------------------------
    # Drift analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("DRIFT ANALYSIS (±5pp gate)")
    print(f"{'='*70}")
    all_stable, drift_report = analyze_drift(coupling_results)
    if drift_report:
        print(drift_report)
    else:
        print("  (insufficient data for drift analysis)")
    print()
    if all_stable:
        print("  PASS: All conditions within ±5pp across seeds.")
    else:
        print("  WARNING: Some conditions show >5pp spread — seed-sensitive territory.")

    # -----------------------------------------------------------------------
    # Retry stability
    # -----------------------------------------------------------------------
    if retry_results:
        print(f"\n{'='*70}")
        print("RETRY CONTROLLER STABILITY")
        print(f"{'='*70}")
        # Group by (model, lane)
        retry_groups = defaultdict(list)
        for r in retry_results:
            if r["ok"]:
                retry_groups[(r["model"], r["lane"])].append(r)
        for (model, lane), runs in sorted(retry_groups.items()):
            retries = [r["retries"] for r in runs]
            cleans = [r["resolved_clean"] for r in runs]
            fails = [r["converted_to_fail"] for r in runs]
            print(f"  {model:20s} {lane:15s}  "
                  f"retries={retries}  clean={cleans}  fail={fails}")

    # -----------------------------------------------------------------------
    # Resolver health audit
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RESOLVER HEALTH AUDIT")
    print(f"{'='*70}")
    resolver_stats = audit_resolvers()
    if resolver_stats:
        print(f"  Total anchors resolved: {resolver_stats['total_anchors_resolved']}")
        print(f"  Total errors:           {resolver_stats['total_errors']}")
        print(f"  Error rate:             {resolver_stats['error_rate']}")
        print()
        for resolver, classes in resolver_stats.get("by_resolver", {}).items():
            total = sum(classes.values())
            print(f"  {resolver}:")
            for cls, count in sorted(classes.items()):
                pct = count / total * 100 if total else 0
                print(f"    {cls:20s}  {count:4d}  ({pct:.0f}%)")
        status_dist = resolver_stats.get("status_distribution", {})
        if status_dist:
            print()
            print("  HTTP status distribution:")
            for status, count in status_dist.items():
                print(f"    {status:30s}  {count:4d}")
    else:
        print("  No resolver data found.")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("OVERNIGHT SUMMARY")
    print(f"{'='*70}")
    print(f"  Total runs:    {total_runs}")
    print(f"  Failures:      {total_failures}")
    print(f"  Elapsed:       {total_elapsed/60:.1f} min")
    print(f"  Drift stable:  {'YES' if all_stable else 'NO'}")
    print()

    # Write summary JSON
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    summary = {
        "stamp": stamp,
        "total_runs": total_runs,
        "total_failures": total_failures,
        "elapsed_min": round(total_elapsed / 60, 1),
        "drift_stable": all_stable,
        "coupling_results": coupling_results,
        "retry_results": retry_results,
        "resolver_stats": resolver_stats,
    }
    report_path = report_dir / f"overnight_drift_{stamp}.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
