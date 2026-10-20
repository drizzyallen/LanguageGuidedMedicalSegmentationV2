# Phase 0 — Model Setup (Task 0.3)

All three methods use the official public implementations (pinned in `external/`, see `external/SOURCES.md`),
the common 224×224 data from Task 0.2 (`data/prepared/`), and the official LViT splits from Task 0.1.
Upstream code is not modified; wrappers in `src/` import the official model, loss, augmentation and scheduler code.

| Method | Source | Configuration | Input | Pretraining | Notes |
|---|---|---|---|---|---|
| U-Net | LViT (`HUANGLIZI/LViT` @ `ba90775`, `nets/UNet.py`) | binary U-Net, 1 output channel, sigmoid | 224×224, grayscale duplicated to 3 channels | none | LViT `Config.py` recipe (Covid19 settings) for both datasets |
| TransUNet | Official (`Beckschen/TransUNet` @ `02ef001`) | R50-ViT-B_16, 2 classes, `n_skip=3`, grid 14×14 | 224×224, 1 channel (model repeats to 3) | ImageNet-21k `R50+ViT-B_16.npz` (official URL) | official recipe; validation added for checkpoint selection |
| nnU-Net | Official (`MIC-DKFZ/nnUNet` v2.8.1) | 2D, default `nnUNetPlans`, `PlainConvUNet`, patch 224×224, batch 65 | 224×224 prepared data, 1 channel | none | default `nnUNetTrainer`; checkpoint selection by full validation Dice |

Configs: `configs/unet.yaml`, `configs/transunet.yaml`, `configs/nnunet.yaml`.

## Training settings

| Setting | U-Net | TransUNet | nnU-Net |
|---|---|---|---|
| Input size | 224×224 | 224×224 | 224×224 (patch size from planning) |
| Input scaling | uint8 / 255 | uint8 / 255 | z-score per image |
| Pretrained initialization | none | ImageNet-21k R50+ViT-B_16 | none |
| Loss | 0.5 Dice + 0.5 BCE (`WeightedDiceBCE`) | 0.5 CE + 0.5 Dice (softmax) | Dice + CE, deep supervision |
| Optimizer | Adam | SGD, momentum 0.9, weight decay 1e-4 | SGD Nesterov, momentum 0.99, weight decay 3e-5 |
| Learning rate | 3e-4 | 0.01 | 0.01 |
| Scheduler | CosineAnnealingWarmRestarts (`T_0=10`, `eta_min=1e-4`), per epoch | poly (power 0.9), per iteration | poly (power 0.9), per epoch |
| Batch size | 2 | 24 | 65 (nnU-Net planning) |
| Augmentation | LViT `RandomGenerator`: rot90 + flip, or rotation ±20° | TransUNet `RandomGenerator`: rot90 + flip, or rotation ±20° | nnU-Net default (rotation, scaling, noise, blur, brightness, contrast, low resolution, gamma, mirroring) |
| Maximum epochs | 2000, early stopping patience 50 | 150 | 1000 × 250 iterations |
| Checkpoint selection | best validation Dice (saved after epoch 5, as in LViT) | best validation Dice | best validation Dice (`checkpoint_best_valdice.pth`) |
| Seeds | 1001, 1002, 1003 | 1001, 1002, 1003 | 1001, 1002, 1003 |

Validation Dice is computed the same way for all three methods: per-image Dice on the 224×224 prediction
(lesion probability ≥ 0.5) against the prepared mask, averaged over all LViT validation images.
No test data is used for any setting or checkpoint choice.

## Decisions

| | Decision | Reason |
|---|---|---|
| A | TransUNet: validation Dice computed after every epoch; best checkpoint kept. Training itself unchanged. | The official trainer has no validation and saves only the last epoch; Phase 0 requires selection by validation Dice. |
| B | nnU-Net: after every epoch, all validation images are predicted with nnU-Net's own predictor (sliding window, no test-time mirroring), mapped back to 224×224 with nnU-Net's export logic, and scored; best saved as `checkpoint_best_valdice.pth`. Training itself unchanged. | nnU-Net's own `checkpoint_best.pth` uses an EMA "pseudo Dice" on random patches, not the validation Dice; this keeps one selection rule for all methods. |
| C | U-Net: learning rate 3e-4 and batch size 2 for both datasets. | LViT `Config.py` gives 3e-4 for Covid19 (QaTa) and no MosMed value; LViT-T in Phase 1 uses the same `Config.py`, so U-Net and LViT-T share one recipe. |
| D | nnU-Net: default `nnUNetPlans` (not the newer ResEnc presets). | The Phase 0 document cites the original nnU-Net method (Nature Methods 2021); neither document mentions ResEnc. |

nnU-Net uses a fixed `splits_final.json` with one fold: training = LViT train, validation = LViT val.
LViT test images are kept separate (`imagesTs`) for Task 0.4.

## Quick check

Run with `bash src/checks/run_quick_checks.sh <gpu>` (2026-10-20, RTX 3090). For each method and dataset:
one batch loaded, one forward pass, output/mask shapes checked, loss computed, short training on a small subset
(U-Net / TransUNet: first 64 training and 32 validation images, 10 epochs; nnU-Net: 5 epochs × 50 iterations on the
full training set, validation Dice on the first 32 validation images).

| Method | Dataset | Batch (image / mask) | Output | Loss | Validation Dice, start → end of short training |
|---|---|---|---|---|---|
| U-Net | QaTa-COV19 | 2×3×224×224 / 2×224×224, {0,1} | 2×1×224×224 | 0.732 | 0.27 → 0.54 |
| U-Net | MosMedData+ | 2×3×224×224 / 2×224×224, {0,1} | 2×1×224×224 | 0.689 | 0.11 → 0.30 |
| TransUNet | QaTa-COV19 | 24×1×224×224 / 24×224×224, {0,1} | 24×2×224×224 | 0.564 | 0.00 → 0.58 |
| TransUNet | MosMedData+ | 24×1×224×224 / 24×224×224, {0,1} | 24×2×224×224 | 0.588 | 0.00 → 0.00 (10 epochs); 0.00 → 0.54 (40 epochs) |
| nnU-Net | QaTa-COV19 | 65×1×224×224 / 65×1×224×224, {0,1} | 65×2×224×224 (+4 deep supervision) | 0.903 | 0.00 → 0.61 |
| nnU-Net | MosMedData+ | 65×1×224×224 / 65×1×224×224, {0,1} | 65×2×224×224 (+4 deep supervision) | 1.168 | 0.00 → 0.61 |

TransUNet on MosMedData+ predicted only background within the 10-epoch check (30 iterations) while its loss fell
normally (0.55 → 0.19); with 40 epochs (120 iterations) it started predicting lesions at epoch 20 and reached 0.54,
so the 10-epoch check was too short for MosMed's small lesions (≈1.5% of the image), not a fault.

The quick-check numbers only show that training works; they are not results.

## Compute estimate

Measured 2026-10-20 on one RTX 3090 (GPU 7, no other job on it): one full-data epoch including the full
validation Dice (nnU-Net: second epoch, the first includes `torch.compile`).

| Method | QaTa-COV19 epoch | MosMedData+ epoch | Epochs per run | Estimated time per run (QaTa / MosMed) |
|---|---:|---:|---|---|
| U-Net | 82 s | 31 s | early stopping (max 2000) | ≈ 2–7 h / 1–3 h if it stops after 100–300 epochs; at most 46 h / 17 h |
| TransUNet | 129 s | 52 s | 150 | ≈ 5.4 h / 2.2 h |
| nnU-Net | 92 s | 58 s | 1000 | ≈ 26 h / 16 h |

Total for 18 runs (3 seeds): about 125 GPU-hours for nnU-Net, 23 for TransUNet and roughly 10–30 for U-Net,
i.e. ≈ 160–180 GPU-hours. With 4 GPUs running in parallel this is ≈ 2 days of wall-clock time.
Times may be longer when several jobs share the machine's CPUs (data augmentation runs on the CPU).

## Environment

Conda env `lgms_phase0` (`environment.yml`): Python 3.10, PyTorch 2.5.1 + CUDA 12.1, nnU-Net 2.8.1, OpenCV 4.10.
