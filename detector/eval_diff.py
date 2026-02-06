"""
Δt Hallucination Detector - Eval Diff
Compare two CSV eval runs and summarize changes.
"""

import csv
import sys
from typing import Dict, Any, Optional, Tuple


def _load_csv(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = row.get("id", "")
            if rid:
                rows[rid] = row
    return rows


def _to_float(val: Any) -> Optional[float]:
    try:
        if val is None or val == "":
            return None
        return float(val)
    except Exception:
        return None


def diff_csv(
    old_path: str,
    new_path: str,
    output_csv: Optional[str] = None,
    confidence_delta: float = 0.1
) -> Tuple[int, int, int]:
    old_rows = _load_csv(old_path)
    new_rows = _load_csv(new_path)
    
    changed_pred = 0
    changed_conf = 0
    missing = 0
    
    out_f = open(output_csv, "w", newline="") if output_csv else None
    writer = None
    if out_f:
        writer = csv.DictWriter(out_f, fieldnames=[
            "id",
            "old_prediction",
            "new_prediction",
            "old_confidence",
            "new_confidence"
        ])
        writer.writeheader()
    
    for rid, new_row in new_rows.items():
        old_row = old_rows.get(rid)
        if not old_row:
            missing += 1
            continue
        old_pred = old_row.get("prediction", "")
        new_pred = new_row.get("prediction", "")
        if old_pred != new_pred:
            changed_pred += 1
        old_conf = _to_float(old_row.get("confidence"))
        new_conf = _to_float(new_row.get("confidence"))
        if old_conf is not None and new_conf is not None:
            if abs(new_conf - old_conf) >= confidence_delta:
                changed_conf += 1
        
        if writer and (old_pred != new_pred or (old_conf is not None and new_conf is not None and abs(new_conf - old_conf) >= confidence_delta)):
            writer.writerow({
                "id": rid,
                "old_prediction": old_pred,
                "new_prediction": new_pred,
                "old_confidence": old_conf,
                "new_confidence": new_conf
            })
    
    if out_f:
        out_f.close()
    
    total = len(new_rows)
    print(f"Total rows: {total}", file=sys.stdout)
    print(f"Prediction changes: {changed_pred}", file=sys.stdout)
    print(f"Confidence changes (>= {confidence_delta:.2f}): {changed_conf}", file=sys.stdout)
    print(f"Missing in old: {missing}", file=sys.stdout)
    
    return total, changed_pred, changed_conf
