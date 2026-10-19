# Language-Guided Medical Segmentation (V2)

Benchmark of image-only and language-guided segmentation on QaTa-COV19 and MosMedData+,
using the official LViT train/validation/test splits and a common 224×224 evaluation.

- Phase 0 — image-only benchmark: U-Net, TransUNet, nnU-Net (`instructions/PHASE_0_IMAGE_ONLY_BENCHMARK_FINAL.md`)
- Phase 1 — language-guided segmentation: LViT-T, RecLMIS, DD-CMD (`instructions/PHASE_1_LANGUAGE_GUIDED_SEGMENTATION_FINAL.md`)

## Structure

```text
external/   official implementations (git submodules, pinned; see external/SOURCES.md)
src/        our code: data preparation, wrappers, evaluation
configs/    one config per method
data/       prepared datasets (not committed)
runs/       training outputs and checkpoints (not committed)
results/    result tables (CSV)
reports/    phase reports
datasets/   raw downloaded data (not committed)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/drizzyallen/LanguageGuidedMedicalSegmentationV2.git
cd LanguageGuidedMedicalSegmentationV2
conda env create -f environment.yml        # env: lgms_phase0 (Python 3.10, PyTorch 2.5.1 + CUDA 12.1)
conda activate lgms_phase0
```

TransUNet pretrained weights (`R50+ViT-B_16.npz`) are not committed; download instructions and
checksums are in `external/SOURCES.md`.

Check that the implementations build:

```bash
python src/checks/check_public_implementations.py
```

Raw datasets and model weights are never stored in this repository.
