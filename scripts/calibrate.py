"""
Calibration pass: extract raw features from control prompts.
Dumps per-feature stats to help pick sigmoid params and thresholds.
"""
import json
import csv
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.config import DetectorConfig
from detector.core import DeltaTDetector
from detector.features import TemporalFeatures, compute_temporal_debt


def main():
    config = DetectorConfig()
    config.device = "cuda"

    detector = DeltaTDetector(model_name="Qwen/Qwen2.5-3B-Instruct", config=config)

    jsonl_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration_controls.jsonl"
    )

    prompts = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))

    all_features = []

    for item in prompts:
        print(f"\n--- {item['id']}: {item['prompt'][:60]}...")
        features = detector.compute_features(item['prompt'])
        if features is None:
            print(f"  SKIP: feature extraction failed")
            continue

        d = features.to_dict()
        debt = compute_temporal_debt(features)
        d['temporal_debt'] = debt
        d['id'] = item['id']
        all_features.append(d)

        print(f"  tokens_to_high_conf: {d['tokens_to_high_conf']}")
        print(f"  max_confidence_slope: {d['max_confidence_slope']:.4f}")
        print(f"  confidence_acceleration: {d['confidence_acceleration']:.4f}")
        print(f"  perturbation_sensitivity: {d['perturbation_sensitivity']:.4f}")
        print(f"  entropy_variance: {d['entropy_variance']:.4f}")
        print(f"  temporal_debt: {debt:.4f}")

    # Dump CSV
    out_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration_features.csv"
    )

    numeric_cols = TemporalFeatures.feature_columns()
    fieldnames = ['id'] + numeric_cols + ['temporal_debt']

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in all_features:
            writer.writerow(row)

    print(f"\n\nCalibration features written to {out_path}")

    # Print summary stats
    print("\n" + "="*70)
    print("CALIBRATION SUMMARY (n={})".format(len(all_features)))
    print("="*70)

    key_features = [
        'tokens_to_high_conf', 'max_confidence_slope',
        'confidence_acceleration', 'perturbation_sensitivity',
        'entropy_variance', 'temporal_debt'
    ]

    for feat in key_features:
        values = [d[feat] for d in all_features if feat in d]
        if values:
            arr = np.array(values)
            print(f"\n{feat}:")
            print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}")
            print(f"  median={np.median(arr):.4f}  iqr={np.percentile(arr,75)-np.percentile(arr,25):.4f}")
            print(f"  min={arr.min():.4f}  max={arr.max():.4f}")
            print(f"  p5={np.percentile(arr,5):.4f}  p95={np.percentile(arr,95):.4f}")


if __name__ == "__main__":
    main()
