# Phase 1 — Public Implementations Check

Date: 2026-11-11. No QaTa-COV19 or MosMedData+ data was used; checks use a random 224×224 image and one example
sentence. Script: `src/checks/check_phase1_implementations.py` (run once per method, in that method's env).

## Implementations

| Method | Source (pinned) | Env | Built as | Text pipeline (official) | Pretrained component | Result |
|---|---|---|---|---|---|---|
| LViT-T | `HUANGLIZI/LViT` @ `ba90775` | `lgms_lvit` (Py 3.7, torch 1.8.0+cu111) | `LViT(Config.get_CTranS_config(), n_channels=3, n_classes=1)` as in `train_model.py` | `BertEmbedding()` on the sentence, first ≤10 token embeddings → 1×10×768 | BERT `bert_12_768_12` / `book_corpus_wiki_en_uncased` (gluonnlp download, hash-verified) | output 1×1×224×224, sigmoid; 39.93M params |
| RecLMIS | `ShawnHuang497/RecLMIS` @ `d3265c8` | `lgms_reclmis` (Py 3.9, torch 1.8.0+cu111) | `RecLMIS(config, config.get_ViT_config(), n_channels, n_labels)` with `Config_covid19` as in `train_model.py` | `clip.tokenize(context_length=18, truncate=True)` → 1×18 tokens + mask | CLIP ViT-B/32 (`nets/ViT-B-32.pt`, SHA-256 verified); weights confirmed loaded, frozen as configured | output 1×1×224×224 (`mode="test"`); 220.72M params |
| DD-CMD | `maklachur/DD-CMD` @ `27387c4` | `lgms_ddcmd` (Py 3.8, torch 1.12.1+cu113) | `TextGuidedSegmentationModel(bert_type, vision_type)` from `config/train.yaml` | BiomedBERT tokenizer, `padding="max_length"`, `max_length=24` | BiomedBERT + ConvNeXt-Tiny (pinned Hugging Face revisions, SHA-256 verified); weights confirmed identical to the checkpoints | output 1×1×224×224; 157.26M params |

Hugging Face prints "some weights were not used" when DD-CMD loads its encoders: the classification / masked-LM
heads in the checkpoints are not part of the encoders. This is expected.

## Environment notes

Details in `envs/README.md`. In short: CUDA 11.1 / 11.3 builds of the pinned torch versions (RTX 3090 support);
packages imported by the code but missing from the requirements files were added; RecLMIS's
`torchaudio 2.4.1+cu118` pin was dropped (incompatible with torch 1.8.0, never imported); `fairseq 0.10.2` for
RecLMIS was installed from its official git tag because its PyPI source package does not build.

## Points for Tasks 1.2–1.4 (not changed here)

- LViT `Config.py` sets `CUDA_VISIBLE_DEVICES="0"` when imported; the training wrapper must restore the chosen GPU.
- LViT offers `LViT` and `LViT_pretrain`; `LViT_pretrain` loads a hard-coded MoNuSeg checkpoint path. Which variant
  is the "official LViT-T training design" for QaTa/MosMed is decided in Task 1.2.
- RecLMIS and DD-CMD ship their own data loaders and text formats (LViT-style `img/labelcol` + xlsx; DD-CMD CSV
  prompt files that are not included in its repository). These are connected to the common data in Task 1.1–1.4.
- DD-CMD's config selects checkpoints by `val_MIoU`; Phase 1 requires selection by validation Dice.
- DD-CMD's released trained checkpoints are "to be shared later"; not needed, as all models are trained here.
