"""Tasks 1.8-1.10: Phase 1 result table, final six-method table and the 15 pairwise comparisons per dataset.

Uses the statistics of src/analysis/phase0_results.py unchanged (same per-case files, seed averaging, percentile
bootstrap with 10,000 resamples, Wilcoxon signed-rank test, Holm correction).

Task 1.8  results/phase1_main_results.csv          LViT-T, RecLMIS, DD-CMD
          - mean ± SD across the 3 seeds of each run's test mean (Dice, mIoU, HD95, ASSD)
          - Dice 95% CI: bootstrap over test cases of the seed-averaged per-case Dice
Task 1.9  results/final_six_method_results.csv     Phase 0 rows (identical to results/phase0_main_results.csv,
          recomputed and checked) + Phase 1 rows, with Text at Train / Text at Test
Extra     results/phase1_published_comparison.csv  our Dice next to the published Dice of each method: per-image
          Dice (our protocol) and whole-dataset Dice (all test pixels pooled, the definition DD-CMD's paper uses)
Task 1.10 results/final_six_method_pairwise.csv    per dataset, all 15 method pairs:
          1. average each test case across the three seeds within each method; 2. align the same cases;
          3. paired differences A - B (Dice, mIoU, HD95, ASSD); 4. 95% CI: paired bootstrap over test cases;
          5. Dice p-value: two-sided Wilcoxon signed-rank test; 6. Holm correction across the 15 Dice comparisons.

Usage (env lgms_phase0): python src/analysis/phase1_results.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_results as P0  # noqa: E402

ROOT = P0.ROOT
PHASE1 = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
ALL = {**P0.METHODS, **PHASE1}
TEXT = {m: ("No", "No") for m in P0.METHODS}
TEXT.update({m: ("Yes", "Yes") for m in PHASE1})
METRICS = P0.METRICS


# Published Dice (%), QaTa-COV19 / MosMedData+, same LViT splits. U-Net, TransUNet, nnU-Net, LViT-T, RecLMIS:
# RecLMIS paper (IEEE TMI 2025, arXiv 2404.02845v2), Table I (definition not stated; consistent with per-image Dice).
# DD-CMD: DD-CMD paper (arXiv 2608.11335), whole-dataset Dice.
PUBLISHED = {"unet": (79.02, 64.60), "transunet": (78.63, 71.24), "nnunet": (80.42, 72.59), "lvit": (83.66, 74.57),
             "reclmis": (85.22, 77.48), "ddcmd": (91.46, 81.95)}
PUBLISHED_SOURCE = {m: "RecLMIS paper, Table I" for m in PUBLISHED}
PUBLISHED_SOURCE["ddcmd"] = "DD-CMD paper (whole-dataset Dice)"


def whole_dataset_dice(method, dataset, seed):
    """Dice with all test pixels pooled: 2 * sum(TP) / (sum(pred) + sum(target)) over the test set."""
    import cv2
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from data.dataset import mask_path, sample_ids
    tp = pred_sum = target_sum = 0
    for s in sample_ids(dataset, "test"):
        pred = cv2.imread(os.path.join(ROOT, "runs", method, dataset, f"seed_{seed}", "test", "predictions",
                                       f"{s}.png"), cv2.IMREAD_GRAYSCALE) > 0
        target = cv2.imread(mask_path(dataset, "test", s), cv2.IMREAD_GRAYSCALE) > 0
        tp += int((pred & target).sum())
        pred_sum += int(pred.sum())
        target_sum += int(target.sum())
    return 2.0 * tp / (pred_sum + target_sum)


def published_comparison(six):
    rows = []
    for dataset, name in P0.DATASETS.items():
        for k, (method, pub) in enumerate((m, PUBLISHED[m][0 if dataset == "qata" else 1]) for m in ALL):
            ours = six[(six.dataset == name) & (six.method == ALL[method])].iloc[0]
            pooled = float(np.mean([whole_dataset_dice(method, dataset, s) for s in P0.SEEDS]))
            matching = pooled if method == "ddcmd" else ours.dice_mean
            rows.append({"dataset": name, "method": ALL[method], "dice_per_image": 100 * ours.dice_mean,
                         "dice_whole_dataset": 100 * pooled, "published_dice": pub,
                         "published_source": PUBLISHED_SOURCE[method],
                         "ours_matching_definition": 100 * matching, "difference": 100 * matching - pub})
    return pd.DataFrame(rows)


def main_results(methods):
    """Same computation as phase0_results.main_results, for any set of methods (one RNG stream per call)."""
    rng = np.random.default_rng(P0.BOOT_SEED)
    rows = []
    for dataset in P0.DATASETS:
        for method in methods:
            run_means = pd.DataFrame([P0.per_case(method, dataset, s)[list(METRICS)].mean() for s in P0.SEEDS])
            avg = P0.seed_averaged(method, dataset)
            lo, hi = P0.bootstrap_ci(avg["dice"], rng)
            row = {"dataset": P0.DATASETS[dataset], "method": ALL[method], "n_seeds": len(P0.SEEDS),
                   "n_test_cases": len(avg)}
            for m in METRICS:
                row[f"{m}_mean"] = run_means[m].mean()
                row[f"{m}_sd"] = run_means[m].std(ddof=1)
            row["dice_ci95_low"], row["dice_ci95_high"] = lo, hi
            for s, (_, r) in zip(P0.SEEDS, run_means.iterrows()):
                row[f"dice_seed{s}"] = r["dice"]
            rows.append(row)
    return pd.DataFrame(rows)


def pairwise():
    rng = np.random.default_rng(P0.BOOT_SEED)
    rows = []
    for dataset in P0.DATASETS:
        avg = {m: P0.seed_averaged(m, dataset) for m in ALL}
        block = []
        for a, b in itertools.combinations(ALL, 2):
            assert avg[a].index.equals(avg[b].index)
            row = {"dataset": P0.DATASETS[dataset], "method_a": ALL[a], "method_b": ALL[b],
                   "n_test_cases": len(avg[a])}
            for m in METRICS:
                diff = (avg[a][m] - avg[b][m]).to_numpy()
                lo, hi = P0.bootstrap_ci(diff, rng)
                row[f"delta_{m}"], row[f"{m}_ci95_low"], row[f"{m}_ci95_high"] = diff.mean(), lo, hi
            row["dice_p_raw"] = wilcoxon(avg[a]["dice"], avg[b]["dice"], alternative="two-sided").pvalue
            block.append(row)
        assert len(block) == 15
        for row, p_holm in zip(block, P0.holm([r["dice_p_raw"] for r in block])):
            row["dice_p_holm"] = p_holm
        rows.extend(block)
    return pd.DataFrame(rows)


def main():
    out = os.path.join(ROOT, "results")
    phase1 = main_results(PHASE1)
    phase1.to_csv(os.path.join(out, "phase1_main_results.csv"), index=False)

    phase0 = main_results(P0.METHODS)
    frozen = pd.read_csv(os.path.join(out, "phase0_main_results.csv"))
    pd.testing.assert_frame_equal(phase0.reset_index(drop=True), frozen, check_exact=False, rtol=1e-12)
    six = pd.concat([phase0, phase1], ignore_index=True)
    six.insert(2, "text_at_train", six["method"].map({ALL[m]: t[0] for m, t in TEXT.items()}))
    six.insert(3, "text_at_test", six["method"].map({ALL[m]: t[1] for m, t in TEXT.items()}))
    order = {d: i for i, d in enumerate(P0.DATASETS.values())}
    six = six.sort_values("dataset", key=lambda s: s.map(order), kind="stable").reset_index(drop=True)
    six.to_csv(os.path.join(out, "final_six_method_results.csv"), index=False)

    published_comparison(six).to_csv(os.path.join(out, "phase1_published_comparison.csv"), index=False)

    pairs = pairwise()
    pairs.to_csv(os.path.join(out, "final_six_method_pairwise.csv"), index=False)

    pd.set_option("display.width", 250)
    print(six.round(4).to_string(index=False))
    print(pairs[["dataset", "method_a", "method_b", "delta_dice", "dice_ci95_low", "dice_ci95_high", "dice_p_raw",
                 "dice_p_holm"]].to_string(index=False))
    print("Phase 0 rows identical to results/phase0_main_results.csv")


if __name__ == "__main__":
    main()
