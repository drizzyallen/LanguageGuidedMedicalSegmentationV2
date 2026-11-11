"""Common final evaluator (Tasks 0.4/0.5): score saved test predictions against the prepared test masks.

Same code for every method. Evaluation size 224x224, binary masks (prediction PNG > 0 = lesion).
Writes per_case_metrics.csv (sample_id, dice, miou, hd95, assd, empty_prediction) and test_summary.json.

Usage: python src/evaluation/evaluate.py --method unet --dataset qata --seed 1001
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import mask_path, sample_ids  # noqa: E402
from evaluation.metrics import WORST_DISTANCE, all_metrics  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METRICS = ["dice", "miou", "hd95", "assd"]


def evaluate(method, dataset, seed):
    test_dir = os.path.join(ROOT, "runs", method, dataset, f"seed_{seed}", "test")
    pred_dir = os.path.join(test_dir, "predictions")
    rows = []
    for sample_id in sample_ids(dataset, "test"):
        pred = cv2.imread(os.path.join(pred_dir, f"{sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(mask_path(dataset, "test", sample_id), cv2.IMREAD_GRAYSCALE)
        assert pred is not None, f"missing prediction {sample_id}"
        assert pred.shape == target.shape == (224, 224)
        pred, target = pred > 0, target > 0
        rows.append({"sample_id": sample_id, **all_metrics(pred, target), "empty_prediction": int(not pred.any())})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(test_dir, "per_case_metrics.csv"), index=False)

    with open(os.path.join(test_dir, "prediction_info.json")) as f:
        info = json.load(f)
    summary = {
        "method": method, "dataset": dataset, "seed": seed, "n_cases": len(df),
        **{m: float(df[m].mean()) for m in METRICS},
        **{f"{m}_sd": float(df[m].std(ddof=1)) for m in METRICS},
        "n_empty_predictions": int(df["empty_prediction"].sum()),
        "n_empty_targets": int(sum(cv2.imread(mask_path(dataset, "test", s), 0).max() == 0
                                   for s in df["sample_id"])),
        "evaluation": {"size": 224, "threshold": info["threshold"], "distance_units": "pixels (224x224)",
                       "undefined_distance_value": WORST_DISTANCE, "miou": "mean over images of lesion IoU"},
        "checkpoint": info["checkpoint"],
    }
    with open(os.path.join(test_dir, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{method} {dataset} {seed}] n={len(df)} dice {summary['dice']:.4f} miou {summary['miou']:.4f} "
          f"hd95 {summary['hd95']:.2f} assd {summary['assd']:.2f} empty_pred {summary['n_empty_predictions']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["unet", "transunet", "nnunet"])
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    evaluate(args.method, args.dataset, args.seed)


if __name__ == "__main__":
    main()
