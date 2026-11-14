# Phase 0 — Image-Only Benchmark: Summary

Phase 0 sets up an image-only baseline on the two datasets that Phase 1 uses for the language-guided methods.
Three image-only models (U-Net, TransUNet and nnU-Net) were each trained with 3 seeds on QaTa-COV19 and
MosMedData+ (18 runs) and evaluated in the same way. These results are the reference that the Phase 1
language-guided methods (LViT-T, RecLMIS, DD-CMD) are compared against.

Detailed reports: `PHASE0_DATA_SUMMARY.md` (data), `PHASE0_MODEL_SETUP.md` (models and training),
`PHASE0_RESULTS.md` (full results). Tables: `results/phase0_main_results.csv`, `results/phase0_pairwise_comparisons.csv`.

## Methodology

**Data.** QaTa-COV19 is a set of chest X-rays (5716 train / 1429 validation / 2113 test images). MosMedData+ is a
set of lung CT slices (2183 / 273 / 273). Both use the official LViT splits, unchanged. Every image and mask was
resized to 224×224 the same way as the official LViT data loader, and masks were binarized (lesion vs background).
All methods were trained and tested on exactly these prepared files. Phase 0 did not use the text annotations.
Two properties of MosMedData+ matter when reading its results. The LViT split is made per slice, so 183 of the 273
test slices come from CT scans that also have slices in the training set. Lesions are also small, about 1.5% of the
image on average.

**Models.** Each model comes from its official public code, pinned to a fixed version. The upstream code is not
modified; small wrappers in `src/` handle data loading, seeds and checkpoint selection.
- **U-Net** is the classic CNN baseline. It is the version distributed with LViT and uses LViT's training recipe
  (Adam, lr 3e-4, batch size 2, Dice + BCE loss, early stopping), so it shares a recipe with LViT-T in Phase 1.
- **TransUNet** is a CNN–Transformer hybrid (R50-ViT-B/16) initialised with ImageNet-21k weights and trained with
  its official recipe: SGD, lr 0.01, batch size 24, 150 epochs.
- **nnU-Net v2** uses its 2D configuration, which configures itself from the data, with its default plans and
  trainer (1000 epochs).

**Training protocol.** Each model–dataset pair was trained with seeds 1001, 1002 and 1003. In every run the
checkpoint was chosen by validation Dice, computed the same way for all methods: per-image Dice at 224×224,
averaged over the validation set. Two of the official trainers needed small additions to make this possible.
TransUNet's trainer has no validation step, and nnU-Net's own "best" checkpoint uses a different score.
No setting was ever chosen using the test data. All runs were trained on RTX 3090 GPUs.

**Evaluation.** One shared evaluator scores all runs on the same test images and the same 224×224 masks. A pixel is
counted as lesion when its predicted probability is at least 0.5. Each test image gets four scores:
- **Dice**, the primary metric;
- **mIoU**, the lesion IoU averaged over images;
- **HD95**, the 95th-percentile boundary distance, in pixels;
- **ASSD**, the average surface distance, in pixels.

If a model predicts no lesion on an image that has one, the distance metrics for that image are set to the worst
case, the image diagonal (316.8 px). Results are reported as the mean ± SD across the 3 seeds. The 95% confidence
interval for Dice is a bootstrap over test images. Methods are compared image by image: each pairwise comparison
gives the mean Dice difference with a bootstrap CI and a Wilcoxon signed-rank test, with Holm correction across the
three comparisons on each dataset.

## Results

### Main results

Dice and mIoU in % (higher is better); HD95 and ASSD in pixels (lower is better). Mean ± SD over 3 seeds.
**Bold** = best on that dataset.

| Dataset | Method | Dice | Dice 95% CI | mIoU | HD95 | ASSD |
|---|---|---|---|---|---|---|
| QaTa-COV19 | U-Net | 78.35 ± 0.24 | [77.46, 79.22] | 68.73 ± 0.25 | 32.30 ± 0.66 | 11.02 ± 0.49 |
| QaTa-COV19 | TransUNet | 79.41 ± 0.22 | [78.49, 80.29] | **70.27 ± 0.15** | **29.02 ± 0.37** | 9.41 ± 0.99 |
| QaTa-COV19 | nnU-Net | **79.59 ± 0.43** | [78.72, 80.43] | 70.23 ± 0.49 | 29.29 ± 1.10 | **9.08 ± 1.14** |
| MosMedData+ | U-Net | 70.36 ± 2.25 | [68.03, 72.54] | 57.44 ± 2.05 | 27.14 ± 6.69 | 12.97 ± 6.47 |
| MosMedData+ | TransUNet | 68.46 ± 0.81 | [66.08, 70.77] | 55.21 ± 0.82 | 25.93 ± 3.01 | 12.12 ± 3.48 |
| MosMedData+ | nnU-Net | **72.03 ± 1.22** | [69.57, 74.31] | **59.58 ± 1.11** | **22.86 ± 2.80** | **9.48 ± 3.03** |

### Test Dice per seed (%)

| Dataset | Method | Seed 1001 | Seed 1002 | Seed 1003 |
|---|---|---:|---:|---:|
| QaTa-COV19 | U-Net | 78.09 | 78.40 | 78.55 |
| QaTa-COV19 | TransUNet | 79.16 | 79.49 | 79.57 |
| QaTa-COV19 | nnU-Net | 80.02 | 79.15 | 79.61 |
| MosMedData+ | U-Net | 72.89 | 68.60 | 69.59 |
| MosMedData+ | TransUNet | 69.26 | 67.64 | 68.48 |
| MosMedData+ | nnU-Net | 72.50 | 72.94 | 70.65 |

### Pairwise comparisons

ΔDice = Dice of A minus Dice of B, in percentage points. It is computed image by image on per-image Dice averaged
over the 3 seeds. "Significant" means the CI excludes 0 and Holm p < 0.05.

| Dataset | A vs B | ΔDice | 95% CI | Holm p | Conclusion |
|---|---|---:|---|---:|---|
| QaTa-COV19 | U-Net vs TransUNet | −1.06 | [−1.45, −0.66] | 4.0e-22 | TransUNet better |
| QaTa-COV19 | U-Net vs nnU-Net | −1.25 | [−1.66, −0.84] | 8.1e-21 | nnU-Net better |
| QaTa-COV19 | TransUNet vs nnU-Net | −0.19 | [−0.62, 0.24] | 0.019 | no clear difference (CI includes 0) |
| MosMedData+ | U-Net vs TransUNet | +1.90 | [−0.22, 4.00] | 6.7e-4 | U-Net tends better (CI includes 0) |
| MosMedData+ | U-Net vs nnU-Net | −1.67 | [−3.71, 0.26] | 0.55 | not significant |
| MosMedData+ | TransUNet vs nnU-Net | −3.57 | [−4.66, −2.43] | 2.4e-16 | nnU-Net better |

Figures: `figures/phase0_qata_dice_ci.png`, `figures/phase0_mosmed_dice_ci.png`, `figures/phase0_seed_stability.png`.

## Conclusion

**nnU-Net is the strongest image-only baseline overall.** It has the highest Dice on both datasets: 79.6% on
QaTa-COV19 and 72.0% on MosMedData+. On QaTa-COV19 it and TransUNet are statistically tied, and both beat U-Net by
about 1 point. On MosMedData+ nnU-Net beats TransUNet clearly (+3.6 points). It also beats U-Net there
(+1.7 points), but that difference is not significant.

**MosMedData+ is harder and noisier.** Dice is about 8 points lower than on QaTa-COV19, and results vary more
between seeds (SD 0.8–2.3 vs 0.2–0.4 points). CIs are wider because the test set is small and the lesions are
small. Its scores may also be optimistic because the split is made per slice. The boundary metrics (HD95, ASSD)
mostly follow Dice. Their seed-to-seed spread is driven by the few images where a model misses the lesion entirely.

**Implications for Phase 1.** Language-guided methods should be compared mainly against nnU-Net, the strongest
baseline (~79.6 / ~72.0 Dice). Compared with U-Net (~78.4 / ~70.4), they show whether adding text helps the CNN
architecture LViT builds on. Differences of about 1 point on QaTa-COV19 can be detected reliably. On MosMedData+,
differences need to be roughly 2–3 points or more before they stand out from seed and sampling noise.
