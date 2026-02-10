#!/usr/bin/env python3
"""
Cross-platform replication sweep.

Runs resolver-heavy lanes on the current platform and writes a
comparison-friendly summary.  Designed to be run identically on
Linux and Mac to detect platform-shaped measurement artifacts.

Lanes:
  1. RFC canonical (plumbing control — should be 0% fab)
  2. PyPI soft  (format shift + fabrication)
  3. PyPI locked (fabrication only)
  4. CVE soft   (resolver-heavy, network artifacts)
  5. CVE locked (pure CVE validation)

Usage:
  .venv/bin/python scripts/replicate.py --device cuda   # Linux GPU
  .venv/bin/python scripts/replicate.py --device cpu     # Mac / no GPU
"""

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LANES = [
    ("rfc",        "data/canonical_15_rfc.jsonl"),
    ("pypi_soft",  "data/canonical_15_pypi_n2.jsonl"),
    ("pypi_locked","data/canonical_10_pypi_locked.jsonl"),
    ("cve_soft",   "data/canonical_15_cve_n2.jsonl"),
    ("cve_locked", "data/canonical_10_cve_locked.jsonl"),
]


def run_lane(name: str, corpus: str, device: str) -> dict:
    """Run a single lane via coupling.py and return parsed metrics."""
    print(f"\n{'='*60}")
    print(f"  Lane: {name}  ({corpus})")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "scripts/coupling.py",
        "--device", device,
        "--corpus", corpus,
        "--topologies", "single",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-300:]}")
        return {"error": result.stderr[-300:]}

    # Parse the report file (most recent)
    reports = sorted(Path("reports").glob("coupling_report_*.md"), reverse=True)
    if not reports:
        return {"error": "no report found"}

    report_text = reports[0].read_text()
    # Extract JSON metrics block
    import re
    m = re.search(r'```json\n({.*?})\n```', report_text, re.DOTALL)
    if not m:
        return {"error": "no metrics in report"}

    metrics = json.loads(m.group(1))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Cross-platform replication sweep")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    plat = platform.platform()
    node = platform.node()

    print(f"Replication sweep — {ts}")
    print(f"Platform: {plat}")
    print(f"Node: {node}")
    print(f"Device: {args.device}")
    print(f"Python: {sys.version}")

    results = {}
    for name, corpus in LANES:
        if not Path(corpus).exists():
            print(f"\n  SKIP: {corpus} not found")
            continue
        metrics = run_lane(name, corpus, args.device)
        results[name] = metrics

    # Summary table
    print(f"\n{'='*70}")
    print(f"  REPLICATION SUMMARY — {node} ({plat})")
    print(f"{'='*70}")
    print(f"{'Lane':<15} {'Anchors':>8} {'Valid':>6} {'Fab%':>6} {'Lie%':>6} {'EvasShift':>10} {'ResErr':>7} {'Verdicts'}")
    print("-" * 70)
    for name, m in results.items():
        if "error" in m:
            print(f"{name:<15} ERROR: {m['error'][:40]}")
            continue
        vh = m.get("verdict_histogram", {})
        verdicts = "/".join(f"{vh.get(k,0)}{k[0]}" for k in ["CLEAN","FAIL","WARN"] if vh.get(k,0))
        print(f"{name:<15} {m.get('anchors_total',0):>8} {m.get('anchors_valid',0):>6} "
              f"{m.get('fabrication_rate',0)*100:>5.1f}% {m.get('lie_rate',0)*100:>5.1f}% "
              f"{m.get('evasion_format_shift',0):>10} {m.get('evasion_missing_anchors',0):>7} "
              f"{verdicts}")

    # Write machine-readable output
    out_path = Path(f"reports/replication_{node}_{ts}.json")
    out_path.parent.mkdir(exist_ok=True)
    bundle = {
        "timestamp": ts,
        "platform": plat,
        "node": node,
        "device": args.device,
        "python": sys.version,
        "lanes": results,
    }
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
