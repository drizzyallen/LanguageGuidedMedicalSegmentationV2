"""Task 1.7: common final evaluation check for all six methods (Phase 0 + Phase 1), 36 runs.

For every run (6 methods x 2 datasets x 3 seeds) this checks that:
  1. per_case_metrics.csv lists exactly the LViT test IDs, in split order (same test samples for all runs);
  2. one 224x224 binary PNG prediction ({0,255}) exists per test ID and nothing else;
  3. every metric recomputed from the saved prediction and the common prepared mask (same targets for all runs)
     with the common evaluator (src/evaluation/metrics.py: 224x224, lesion = PNG > 0 after the 0.5 threshold)
     equals the stored per-case value;
  4. test_summary.json means equal the per-case means and the evaluation settings are identical
     (size 224, threshold 0.5, same undefined-distance value, pixel units);
  5. prediction_info.json uses threshold 0.5 and records a checkpoint selected on validation Dice.
The target masks are hashed once per dataset, so all runs are scored against the same files.

Writes results/phase1_evaluation_check.csv (one row per run). Exits non-zero on any mismatch.
Usage (env lgms_phase0): python src/checks/check_phase1_evaluation.py
"""
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import mask_path, sample_ids  # noqa: E402
from evaluation.metrics import WORST_DISTANCE, all_metrics  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METHODS = ["unet", "transunet", "nnunet", "lvit", "reclmis", "ddcmd"]
DATASETS = ["qata", "mosmed"]
SEEDS = (1001, 1002, 1003)
METRICS = ["dice", "miou", "hd95", "assd"]


def check_run(job):
    method, dataset, seed = job
    test = os.path.join(ROOT, "runs", method, dataset, f"seed_{seed}", "test")
    ids = sample_ids(dataset, "test")
    df = pd.read_csv(os.path.join(test, "per_case_metrics.csv"))
    assert list(df["sample_id"]) == ids, "per-case IDs differ from the test split"
    pred_dir = os.path.join(test, "predictions")
    assert sorted(os.listdir(pred_dir)) == sorted(f"{s}.png" for s in ids), "prediction files differ from test IDs"
    max_diff = 0.0
    for row in df.itertuples(index=False):
        pred = cv2.imread(os.path.join(pred_dir, f"{row.sample_id}.png"), cv2.IMREAD_GRAYSCALE)
        assert pred.shape == (224, 224) and set(np.unique(pred).tolist()) <= {0, 255}, row.sample_id
        target = cv2.imread(mask_path(dataset, "test", row.sample_id), cv2.IMREAD_GRAYSCALE) > 0
        m = all_metrics(pred > 0, target)
        for k in METRICS:
            max_diff = max(max_diff, abs(m[k] - getattr(row, k)))
        assert int(not (pred > 0).any()) == row.empty_prediction, row.sample_id
    assert max_diff < 1e-9, f"recomputed metrics differ (max {max_diff})"
    with open(os.path.join(test, "test_summary.json")) as f:
        summary = json.load(f)
    with open(os.path.join(test, "prediction_info.json")) as f:
        info = json.load(f)
    for k in METRICS:
        assert abs(summary[k] - df[k].mean()) < 1e-12, k
    ev = summary["evaluation"]
    assert ev["size"] == 224 and ev["threshold"] == 0.5 and ev["undefined_distance_value"] == WORST_DISTANCE
    # selected on validation Dice: value recorded (U-Net, TransUNet, Phase 1) or nnU-Net's val-Dice checkpoint file
    assert info["threshold"] == 0.5
    assert "val_dice" in info["checkpoint"] or info["checkpoint"]["path"].endswith("checkpoint_best_valdice.pth")
    return {"method": method, "dataset": dataset, "seed": seed, "n_test_cases": len(df),
            "n_predictions": len(os.listdir(pred_dir)), "n_empty_predictions": int(df.empty_prediction.sum()),
            "dice": summary["dice"], "max_abs_diff_recomputed": max_diff, "threshold": info["threshold"],
            "eval_size": ev["size"], "checkpoint_epoch": info["checkpoint"]["epoch"],
            "checkpoint_val_dice": info["checkpoint"].get("val_dice"), "status": "ok"}


def target_hash(dataset):
    h = hashlib.sha256()
    for s in sample_ids(dataset, "test"):
        with open(mask_path(dataset, "test", s), "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def main():
    jobs = [(m, d, s) for d in DATASETS for m in METHODS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=18) as pool:
        rows = list(pool.map(check_run, jobs))
    df = pd.DataFrame(rows)
    hashes = {d: target_hash(d) for d in DATASETS}
    df["target_masks_sha256"] = df["dataset"].map(hashes)
    out = os.path.join(ROOT, "results", "phase1_evaluation_check.csv")
    df.to_csv(out, index=False)
    print(df.drop(columns=["target_masks_sha256"]).to_string(index=False))
    print("target masks:", hashes)
    print(f"ALL {len(df)} RUNS CONSISTENT -> {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
