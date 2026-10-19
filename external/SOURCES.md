# External Implementations (Phase 0)

Official public implementations, pinned as git submodules. Upstream code is not modified;
any adaptation (data loading, seeds, checkpoint selection) lives in `src/` as wrappers.

| Method | Repository (from Phase 0 doc) | Pinned version | Commit | Used component |
|---|---|---|---|---|
| U-Net | https://github.com/HUANGLIZI/LViT | latest `main` (2026-07-21) | `ba90775ead37d6f445f65ce83d1a0be64157f76f` | `nets/UNet.py` — binary U-Net (background / lesion) |
| TransUNet | https://github.com/Beckschen/TransUNet | latest `main` (2026-02-24) | `02ef0010b36eb8328b5e689eadaf613602edf9b8` | `networks/vit_seg_modeling.py`, config `R50-ViT-B_16`, 2 classes |
| nnU-Net | https://github.com/MIC-DKFZ/nnUNet | release `v2.8.1` | `468cf803df9b267150ae2b6c0c59b8ac84f16227` | nnU-Net v2, `2d` configuration only (no 3D) |

## Papers

- U-Net reference model as distributed with LViT: Li et al., *LViT: Language meets Vision Transformer in Medical Image Segmentation*, IEEE TMI 2023.
- TransUNet: Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, 2021.
- nnU-Net: Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 2021.

## Pretrained weights

TransUNet `R50+ViT-B_16.npz` (Google ViT, ImageNet-21k), downloaded from the URL given in the
official TransUNet README (step "1. Download Google pre-trained ViT models"):

| Field | Value |
|---|---|
| URL | `https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz` |
| Local path | `external/pretrained/imagenet21k/R50+ViT-B_16.npz` (not committed) |
| Size | 461,217,452 bytes |
| Upstream last-modified | 2020-11-30 |
| MD5 | `fb93b71c850df1b0390448d4bafc4262` (matches Google `x-goog-hash`) |
| SHA-256 | `ff009bf39bb4f9198b834cfe46aba2bfdaf730e933ab3e3c4b1edf4226eaafbe` |
| Downloaded | 2026-10-19 |

The official code expects `../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz`; our wrapper sets
`config.pretrained_path` to the local path above instead of modifying upstream code.

Re-download:

```bash
mkdir -p external/pretrained/imagenet21k
curl -o "external/pretrained/imagenet21k/R50+ViT-B_16.npz" \
  "https://storage.googleapis.com/vit_models/imagenet21k/R50%2BViT-B_16.npz"
md5sum "external/pretrained/imagenet21k/R50+ViT-B_16.npz"   # fb93b71c850df1b0390448d4bafc4262
```
