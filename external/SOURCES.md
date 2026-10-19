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

TransUNet `R50+ViT-B_16.npz` (ImageNet-21k), from the URL given in the official TransUNet README:
`https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz` → `external/pretrained/` (not committed).
