"""Tasks 0.6 and 0.7: Phase 0 main results and pairwise comparisons from the 18 test evaluations.

Task 0.6 (results/phase0_main_results.csv): per method and dataset
  - mean ± SD across the 3 seeds of each run's test mean (Dice, mIoU, HD95, ASSD)
  - Dice 95% CI: percentile bootstrap over test cases (10,000 resamples) of the seed-averaged per-case Dice

Task 0.7 (results/phase0_pairwise_comparisons.csv): within each dataset, for each pair of methods
  1. average each test case across the three seeds within each method
  2. align the same test cases
  3. paired Dice differences (A - B)
  4. 95% CI: paired percentile bootstrap over test cases (10,000 resamples) of the mean difference
  5. paired p-value: two-sided Wilcoxon signed-rank test
  6. Holm correction across the three comparisons within each dataset

Usage: python src/analysis/phase0_results.py
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
METHODS = {"unet": "U-Net", "transunet": "TransUNet", "nnunet": "nnU-Net"}
DATASETS = {"qata": "QaTa-COV19", "mosmed": "MosMedData+"}
SEEDS = (1001, 1002, 1003)
METRICS = ("dice", "miou", "hd95", "assd")
N_BOOT = 10_000
BOOT_SEED = 0


def per_case(method, dataset, seed):
    path = os.path.join(ROOT, "runs", method, dataset, f"seed_{seed}", "test", "per_case_metrics.csv")
    return pd.read_csv(path).set_index("sample_id")


def seed_averaged(method, dataset):
    """Per-case metrics averaged across the three seeds (same test cases in every run)."""
    frames = [per_case(method, dataset, s)[list(METRICS)] for s in SEEDS]
    assert all(f.index.equals(frames[0].index) for f in frames)
    return sum(frames) / len(frames)


def bootstrap_ci(values, rng):
    values = np.asarray(values)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def holm(pvalues):
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def main_results():
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for dataset in DATASETS:
        for method in METHODS:
            run_means = pd.DataFrame([per_case(method, dataset, s)[list(METRICS)].mean() for s in SEEDS])
            avg = seed_averaged(method, dataset)
            lo, hi = bootstrap_ci(avg["dice"], rng)
            row = {"dataset": DATASETS[dataset], "method": METHODS[method], "n_seeds": len(SEEDS),
                   "n_test_cases": len(avg)}
            for m in METRICS:
                row[f"{m}_mean"] = run_means[m].mean()
                row[f"{m}_sd"] = run_means[m].std(ddof=1)
            row["dice_ci95_low"], row["dice_ci95_high"] = lo, hi
            for s, (_, r) in zip(SEEDS, run_means.iterrows()):
                row[f"dice_seed{s}"] = r["dice"]
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise():
    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for dataset in DATASETS:
        avg = {m: seed_averaged(m, dataset)["dice"] for m in METHODS}
        block = []
        for a, b in itertools.combinations(METHODS, 2):
            assert avg[a].index.equals(avg[b].index)
            diff = (avg[a] - avg[b]).to_numpy()
            lo, hi = bootstrap_ci(diff, rng)
            p = wilcoxon(avg[a], avg[b], alternative="two-sided").pvalue
            block.append({"dataset": DATASETS[dataset], "method_a": METHODS[a], "method_b": METHODS[b],
                          "n_test_cases": len(diff), "delta_dice": diff.mean(), "ci95_low": lo, "ci95_high": hi,
                          "p_raw": p})
        for row, p_holm in zip(block, holm([r["p_raw"] for r in block])):
            row["p_holm"] = p_holm
        rows.extend(block)
    return pd.DataFrame(rows)


def main():
    out = os.path.join(ROOT, "results")
    os.makedirs(out, exist_ok=True)
    main_df = main_results()
    pair_df = pairwise()
    main_df.to_csv(os.path.join(out, "phase0_main_results.csv"), index=False)
    pair_df.to_csv(os.path.join(out, "phase0_pairwise_comparisons.csv"), index=False)
    pd.set_option("display.width", 250)
    print(main_df.round(4).to_string(index=False))
    print(pair_df.to_string(index=False))


if __name__ == "__main__":
    main()
