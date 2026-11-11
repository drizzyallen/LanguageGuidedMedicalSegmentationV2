# External Implementations

Official public implementations, pinned as git submodules. Upstream code is not modified;
any adaptation (data loading, seeds, checkpoint selection) lives in `src/` as wrappers.

## Phase 0 (image-only)

| Method | Repository (from Phase 0 doc) | Pinned version | Commit | Used component |
|---|---|---|---|---|
| U-Net | https://github.com/HUANGLIZI/LViT | latest `main` (2026-07-21) | `ba90775ead37d6f445f65ce83d1a0be64157f76f` | `nets/UNet.py` — binary U-Net (background / lesion) |
| TransUNet | https://github.com/Beckschen/TransUNet | latest `main` (2026-02-24) | `02ef0010b36eb8328b5e689eadaf613602edf9b8` | `networks/vit_seg_modeling.py`, config `R50-ViT-B_16`, 2 classes |
| nnU-Net | https://github.com/MIC-DKFZ/nnUNet | release `v2.8.1` | `468cf803df9b267150ae2b6c0c59b8ac84f16227` | nnU-Net v2, `2d` configuration only (no 3D) |

### Papers

- U-Net reference model as distributed with LViT: Li et al., *LViT: Language meets Vision Transformer in Medical Image Segmentation*, IEEE TMI 2023.
- TransUNet: Chen et al., *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*, 2021.
- nnU-Net: Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 2021.

### Pretrained weights

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

---

## Phase 1 (language-guided)

Repositories from `instructions/PHASE_1_LANGUAGE_GUIDED_SEGMENTATION_FINAL.md`, pinned as git submodules.
Each method runs in its own conda environment built from its own requirements (see `envs/README.md`).

| Method | Repository (from Phase 1 doc) | Pinned version | Commit | Used component |
|---|---|---|---|---|
| LViT-T | https://github.com/HUANGLIZI/LViT | latest `main` (2026-07-21), same as Phase 0 | `ba90775ead37d6f445f65ce83d1a0be64157f76f` | `nets/LViT.py`, `Config.get_CTranS_config()`, text via `bert_embedding.BertEmbedding` |
| RecLMIS | https://github.com/ShawnHuang497/RecLMIS | latest `main` (2025-06-19) | `d3265c8b4d4582a94595fe1c8b1892c618bc9715` | `nets/RecLMIS.py`, `Config_covid19.py` / `Config_MosMedPlus.py`, text via `clip.tokenize` |
| DD-CMD | https://github.com/maklachur/DD-CMD | latest `main` (2026-08-12) | `27387c47163971448f8fcdf5e37f310509dbd948` | `net/model.py` (`TextGuidedSegmentationModel`), `config/train.yaml` |

Papers:

- LViT: Li et al., *LViT: Language meets Vision Transformer in Medical Image Segmentation*, IEEE TMI 2023.
- RecLMIS: Huang et al., *Cross-Modal Conditioned Reconstruction for Language-Guided Medical Image Segmentation*, IEEE TMI.
- DD-CMD: Rahman and Hammond, *Dual-Domain Cross-Modal Decoding for Clinical Text-Guided Medical Image Segmentation*, MICCAI 2026.

### Pretrained components (stored in `external/pretrained/`, not committed)

| Used by | Component | Source (as specified by the repository) | Local path | Verification |
|---|---|---|---|---|
| LViT-T | BERT `bert_12_768_12`, `book_corpus_wiki_en_uncased` (default of `bert_embedding.BertEmbedding`) | downloaded automatically by gluonnlp on first use | `external/pretrained/mxnet/models/` (set `MXNET_HOME=external/pretrained/mxnet`) | SHA-1 `75cc780f…` (params) and `a6607397…` (vocab) match the hashes in gluonnlp's file names |
| RecLMIS | CLIP ViT-B/32 (`ViT-B-32.pt`) | README: `https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt`, "put it in the folder `nets/`" | `external/pretrained/clip/ViT-B-32.pt`; symlink `external/RecLMIS/nets/ViT-B-32.pt` → it | SHA-256 `40d36571…eba950af` matches the hash in the official URL |
| DD-CMD | ConvNeXt-Tiny | README: `facebook/convnext-tiny-224` (Hugging Face), revision `6166b7613034066690a621d8bf25ffdf181a34f0` | `external/pretrained/huggingface/convnext-tiny-224/` | `pytorch_model.bin` SHA-256 `1fe15b2e…f477` matches the Hub LFS hash |
| DD-CMD | PubMedBERT / BiomedBERT | README: `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (Hugging Face), revision `e1354b7a3a09615f6aba48dfad4b7a613eef7062` | `external/pretrained/huggingface/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/` | `pytorch_model.bin` SHA-256 `ad7bbb66…c193` matches the Hub LFS hash |

Re-download:

```bash
# CLIP (RecLMIS)
mkdir -p external/pretrained/clip
curl -L -o external/pretrained/clip/ViT-B-32.pt \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
ln -sfn ../../pretrained/clip/ViT-B-32.pt external/RecLMIS/nets/ViT-B-32.pt
# Hugging Face backbones (DD-CMD): files config.json, pytorch_model.bin (+ preprocessor_config.json / tokenizer files)
#   https://huggingface.co/facebook/convnext-tiny-224/resolve/6166b7613034066690a621d8bf25ffdf181a34f0/<file>
#   https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/resolve/e1354b7a3a09615f6aba48dfad4b7a613eef7062/<file>
# BERT (LViT-T): first run with MXNET_HOME=external/pretrained/mxnet downloads it automatically
```

Not used: the released trained RecLMIS checkpoints (Google Drive) and DD-CMD checkpoints (not yet public) —
all Phase 1 models are trained from these pretrained components under the common protocol.
