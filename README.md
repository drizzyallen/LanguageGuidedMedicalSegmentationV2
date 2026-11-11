# Language-Guided Medical Segmentation (V2)

Benchmark of image-only and language-guided segmentation on QaTa-COV19 and MosMedData+,
using the official LViT train/validation/test splits and a common 224×224 evaluation.

- Phase 0 — image-only benchmark: U-Net, TransUNet, nnU-Net (`instructions/PHASE_0_IMAGE_ONLY_BENCHMARK_FINAL.md`) — **complete**, see `reports/PHASE0_RESULTS.md`
- Phase 1 — language-guided segmentation: LViT-T, RecLMIS, DD-CMD (`instructions/PHASE_1_LANGUAGE_GUIDED_SEGMENTATION_FINAL.md`) — in progress

## Structure

```text
instructions/        phase protocols (what to do)
external/            official implementations, pinned git submodules — see external/SOURCES.md
  LViT/ TransUNet/ nnUNet/       Phase 0 (U-Net from LViT) — LViT also provides LViT-T for Phase 1
  RecLMIS/ DD-CMD/               Phase 1
  pretrained/                    pretrained weights (not committed): imagenet21k/, clip/, huggingface/, mxnet/
envs/                one conda environment per set of official requirements — see envs/README.md
configs/             one training config per method
src/                 our code
  data/                splits (Task 0.1), common 224×224 preparation (Task 0.2), shared dataset
  evaluation/          common final evaluator and metrics
  analysis/            result tables, statistics and figures
  checks/              implementation checks (phase0_, phase1_) and quick checks
  nnunet/              nnU-Net conversion, trainer and launcher
  train_*.py, predict.py, run_phase0.py, test_phase0.sh    Phase 0 training / testing
data/                splits/ (committed); raw/, prepared/, nnunet/ (not committed)
runs/                training outputs, checkpoints, test predictions (not committed)
results/             result tables (CSV)
reports/             phase reports (PHASE0_*, PHASE1_*) and figures/
datasets/            raw downloaded data (not committed)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/drizzyallen/LanguageGuidedMedicalSegmentationV2.git
cd LanguageGuidedMedicalSegmentationV2
conda env create -f envs/phase0_environment.yml   # lgms_phase0: Phase 0, data, evaluation, analysis
bash envs/build_envs.sh lvit                      # lgms_lvit:    LViT-T
bash envs/build_envs.sh reclmis                   # lgms_reclmis: RecLMIS
bash envs/build_envs.sh ddcmd                     # lgms_ddcmd:   DD-CMD
```

Pretrained weights are not committed; download instructions and checksums are in `external/SOURCES.md`.

Check that the implementations build:

```bash
conda activate lgms_phase0 && python src/checks/check_phase0_implementations.py
conda activate lgms_lvit && MXNET_HOME=external/pretrained/mxnet python src/checks/check_phase1_implementations.py --method lvit
conda activate lgms_reclmis && python src/checks/check_phase1_implementations.py --method reclmis
conda activate lgms_ddcmd && python src/checks/check_phase1_implementations.py --method ddcmd
```

Raw datasets and model weights are never stored in this repository.
