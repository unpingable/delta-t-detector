#!/usr/bin/env python3
"""
Resolver-only cross-platform replication.

Takes predictions.jsonl from a Linux run, re-validates all anchors on the
current platform, and compares resolver results. No model/torch needed.

Usage:
  .venv/bin/python scripts/resolver_replication.py predictions.jsonl
"""

import json
import platform
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from detector.invariants import EpistemicGroundingTest
from detector.utils import extract_citations, count_citations


def main():
    if len(sys.argv) < 2:
        print("Usage: resolver_replication.py <predictions.jsonl> [predictions2.jsonl ...]")
        sys.exit(1)

    plat = platform.platform()
    node = platform.node()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"Resolver replication — {ts}")
    print(f"Platform: {plat}")
    print(f"Node: {node}")
    print(f"Python: {sys.version}")
    print()

    eg = EpistemicGroundingTest(max_fabricated=1, validate_urls=True, timeout=8, use_cache=False)

    all_results = []

    for pred_file in sys.argv[1:]:
        print(f"{'='*60}")
        print(f"  File: {pred_file}")
        print(f"{'='*60}")

        with open(pred_file) as f:
            predictions = [json.loads(line) for line in f if line.strip()]

        file_results = []
        for pred in predictions:
            pid = pred["id"]
            orig_inv = pred.get("invariant_results", {}).get("epistemic_grounding", {})
            orig_details = orig_inv.get("details", {})
            orig_vr = orig_details.get("validation_results", {})

            # Re-validate using EG test on the response text
            # We don't have the original response text, but we have the anchors
            # Re-validate each anchor individually
            revalidated = {}
            for anchor_key, orig_result in orig_vr.items():
                resolver = orig_result.get("resolver", "")
                reason_orig = orig_result.get("reason", "")
                valid_orig = orig_result.get("valid", False)

                # Re-validate based on anchor type
                if anchor_key.startswith("doi:"):
                    doi = anchor_key[4:]
                    valid_new, reason_new = eg.validate_doi(doi)
                elif anchor_key.startswith("rfc:"):
                    rfc = anchor_key[4:]
                    valid_new, reason_new = eg.validate_rfc(rfc)
                elif anchor_key.startswith("arxiv:"):
                    arxiv_id = anchor_key[6:]
                    valid_new, reason_new = eg.validate_arxiv(arxiv_id)
                elif anchor_key.startswith("pypi:"):
                    spec = anchor_key[5:]
                    valid_new, reason_new = eg.validate_pypi(spec)
                elif anchor_key.startswith("cve:"):
                    cve_id = anchor_key[4:]
                    valid_new, reason_new = eg.validate_cve(cve_id)
                elif anchor_key.startswith("http"):
                    results = eg.validate_urls([anchor_key])
                    if anchor_key in results:
                        valid_new, reason_new = results[anchor_key]
                    else:
                        valid_new, reason_new = False, "not checked"
                else:
                    continue

                is_error = eg._is_resolver_error(reason_new)
                changed = valid_new != valid_orig
                revalidated[anchor_key] = {
                    "orig_valid": valid_orig,
                    "orig_reason": reason_orig,
                    "new_valid": valid_new,
                    "new_reason": reason_new,
                    "resolver_error": is_error,
                    "changed": changed,
                }

            n_changed = sum(1 for r in revalidated.values() if r["changed"])
            n_errors = sum(1 for r in revalidated.values() if r["resolver_error"])
            n_total = len(revalidated)

            status = "OK" if n_changed == 0 else f"CHANGED({n_changed})"
            print(f"  {pid}: {n_total} anchors, {n_changed} changed, {n_errors} resolver_errors  [{status}]")

            if n_changed > 0:
                for k, r in revalidated.items():
                    if r["changed"]:
                        print(f"    {k}: {r['orig_valid']}({r['orig_reason']}) → {r['new_valid']}({r['new_reason']})")

            file_results.append({
                "id": pid,
                "anchors": n_total,
                "changed": n_changed,
                "resolver_errors": n_errors,
                "details": revalidated,
            })

        all_results.append({
            "file": pred_file,
            "prompts": len(file_results),
            "total_anchors": sum(r["anchors"] for r in file_results),
            "total_changed": sum(r["changed"] for r in file_results),
            "total_resolver_errors": sum(r["resolver_errors"] for r in file_results),
            "per_prompt": file_results,
        })

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESOLVER REPLICATION SUMMARY — {node}")
    print(f"{'='*60}")
    print(f"{'File':<45} {'Anchors':>8} {'Changed':>8} {'ResErr':>8}")
    print("-" * 60)
    for r in all_results:
        fname = Path(r["file"]).name
        print(f"{fname:<45} {r['total_anchors']:>8} {r['total_changed']:>8} {r['total_resolver_errors']:>8}")

    total_anchors = sum(r["total_anchors"] for r in all_results)
    total_changed = sum(r["total_changed"] for r in all_results)
    total_errors = sum(r["total_resolver_errors"] for r in all_results)
    print("-" * 60)
    print(f"{'TOTAL':<45} {total_anchors:>8} {total_changed:>8} {total_errors:>8}")

    if total_changed == 0:
        print("\nAll anchors agree across platforms. Resolver behavior is platform-invariant.")
    else:
        print(f"\n{total_changed}/{total_anchors} anchors changed. Resolver behavior differs across platforms.")

    # Save
    out_path = Path(f"reports/resolver_replication_{node}_{ts}.json")
    out_path.parent.mkdir(exist_ok=True)
    bundle = {
        "timestamp": ts,
        "platform": plat,
        "node": node,
        "python": sys.version,
        "results": all_results,
    }
    out_path.write_text(json.dumps(bundle, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
