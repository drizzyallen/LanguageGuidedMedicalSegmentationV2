# Environments

One conda environment per set of official requirements. Each repository's own pins are kept; the only changes
are CUDA builds that support the RTX 3090, packages the code imports but the requirements file omits, and pins
that cannot be installed together. Exact installed versions: `*_freeze.txt`.

| Env | Used for | Python | PyTorch | Spec | Build |
|---|---|---|---|---|---|
| `lgms_phase0` | Phase 0: U-Net, TransUNet, nnU-Net; data preparation, evaluation, analysis | 3.10 | 2.5.1 + CUDA 12.1 | `phase0_environment.yml` | `conda env create -f envs/phase0_environment.yml` |
| `lgms_lvit` | Phase 1: LViT-T | 3.7 | 1.8.0 + CUDA 11.1 | `lvit_requirements.txt` | `bash envs/build_envs.sh lvit` |
| `lgms_reclmis` | Phase 1: RecLMIS | 3.9 | 1.8.0 + CUDA 11.1 | `reclmis_requirements.txt` | `bash envs/build_envs.sh reclmis` |
| `lgms_ddcmd` | Phase 1: DD-CMD | 3.8 | 1.12.1 + CUDA 11.3 | `ddcmd_requirements.txt` | `bash envs/build_envs.sh ddcmd` |

## Changes relative to each repository's requirements

**LViT** (`external/LViT/requirements.txt`, README: Python 3.7)
- `torch==1.8.0`, `torchvision==0.9.0` → the CUDA 11.1 builds of the same versions (`+cu111`); the default
  builds do not support the RTX 3090 (sm_86).
- `mkl-fft`, `mkl-random`, `mkl-service` (conda-only builds) are not installed; not imported by the code.
- `bert-embedding` is installed first, then `numpy==1.17.5`, as the LViT README instructs. pip reports that
  `bert-embedding` / `mxnet 1.4.0` want an older numpy; this is the conflict the README describes.
- Added `transformers==4.24.0`: imported by `nets/textlevel.py` but missing from `requirements.txt`.
- The BERT weights used by `BertEmbedding` are downloaded on first use; set
  `MXNET_HOME=external/pretrained/mxnet` so they are stored with the other pretrained files.

**RecLMIS** (`external/RecLMIS/requirements.txt`, README: Python 3.9)
- `torch==1.8.0+cu111`, `torchvision==0.9.0+cu111`: as pinned.
- `torchaudio==2.4.1+cu118` is not installed: it cannot be installed with torch 1.8.0 and is never imported.
- Added packages the code imports but `requirements.txt` omits: OpenAI `clip` (git commit `d05afc4`),
  `fairseq 0.10.2`, `ml-collections`, `scipy`, `scikit-learn`, `matplotlib`, `timm`, `thop`, `einops`, `openpyxl`.
- `fairseq 0.10.2` has no Python 3.9 wheel and its PyPI source package fails to build (it ships
  `fairseq/examples` as a directory, which `setup.py` then tries to unlink). It is installed from the official
  git tag `v0.10.2` (`83e615d`) with `setuptools<60` and `--no-build-isolation`.
- `pip check` prints `ninja 1.11.1.1 is not supported on this platform`; ninja is installed and works.

**DD-CMD** (`external/DD-CMD/requirements.txt`, README: Python 3.8)
- `torch=1.12.1`, `torchvision=0.13.1` → the CUDA 11.3 builds of the same versions (`+cu113`).
- Other pins as listed: `pytorch_lightning 1.9.0`, `torchmetrics 0.10.3`, `transformers 4.24.0`, `monai 1.0.1`.
- Added `thop`, `pyyaml`, `openpyxl`: imported by the code but missing from `requirements.txt`.
- The Hugging Face backbones are loaded from local pinned copies in `external/pretrained/huggingface/`.
