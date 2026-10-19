# Phase 0 — Public Implementations Check

Date: 2026-10-19. No QaTa-COV19 or MosMedData+ data was used; checks use random inputs or a synthetic dataset.

## Environment

| Item | Value |
|---|---|
| Conda env | `lgms_phase0` (`environment.yml`) |
| Python | 3.10.20 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| NumPy | 2.2.6 |
| OpenCV | 4.10.0 (pinned to 4.x, to match the OpenCV 4 resizing used by the LViT loader) |
| nnU-Net | 2.8.1 (editable install of `external/nnUNet`) |
| GPU | NVIDIA RTX 3090 (24 GB), driver 535.86.05 |

## Implementations

| Method | Source (pinned) | Configuration | Check | Result |
|---|---|---|---|---|
| U-Net | LViT `ba90775`, `nets/UNet.py` | `UNet(n_channels=3, n_classes=1)` as in LViT `Config.py` (`n_labels = 1`, sigmoid output) | forward 2×3×224×224 | output 2×1×224×224, values in [0, 1]; 14.75M params |
| TransUNet | TransUNet `02ef001` | `R50-ViT-B_16`, `n_classes=2`, `n_skip=3`, grid 14×14 (as in official `train.py`) | load ImageNet-21k weights with official `load_from()`; forward 2×3×224×224 | weights loaded (patch embedding matches checkpoint); output 2×2×224×224; 105.28M params |
| nnU-Net | nnU-Net `v2.8.1` | v2, `2d` configuration only | `nnUNetv2_plan_and_preprocess -c 2d` + `nnUNetv2_train ... 2d 0 -tr nnUNetTrainer_1epoch` on a 12-image synthetic 224×224 dataset | only `2d` planned (patch 224×224, `PlainConvUNet`); 1 epoch trained, validation run, `checkpoint_best.pth` saved |

Script: `src/checks/check_public_implementations.py` (U-Net, TransUNet). The nnU-Net check was run on a
temporary synthetic dataset outside the repository.

## Notes for Task 0.3

- LViT `train_model.py` (at the pinned commit) only builds LViT models; U-Net training uses our wrapper
  with the LViT recipe (`WeightedDiceBCE`, Adam, LViT `Config.py` settings).
- TransUNet expects weights at `../model/vit_checkpoint/imagenet21k/`; our wrapper sets
  `config.pretrained_path` to `external/pretrained/imagenet21k/R50+ViT-B_16.npz`.
- nnU-Net prints a notice recommending the newer residual-encoder (ResEnc) presets. We use the default
  `nnUNetPlans` (the original nnU-Net method cited in the Phase 0 document).
- nnU-Net needs a custom `splits_final.json` (LViT train/val), seed control, and validation-Dice checkpoint
  selection — to be implemented as wrappers in Task 0.3.
