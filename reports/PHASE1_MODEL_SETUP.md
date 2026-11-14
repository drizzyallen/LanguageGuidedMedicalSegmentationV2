# Phase 1 — Model Setup (Tasks 1.2–1.5)

The three language-guided methods use their official public implementations (pinned in `external/`, see
`external/SOURCES.md`), the common 224×224 data and the official LViT splits from Phase 0, and the Task 1.1 text.
Upstream code is not modified. Wrappers in `src/phase1/` import the official dataset classes, text pipelines,
models, losses and schedulers, and run the official training loop. They change only the seed, the data paths, and
checkpoint selection by validation Dice.

| Method | Repository | Input | Text Encoder | Main Cross-Modal Mechanism | Notes |
|---|---|---|---|---|---|
| LViT-T | HUANGLIZI/LViT | 224×224 | BERT-base `bert_12_768_12` (book_corpus_wiki_en_uncased) via `bert-embedding`, frozen, outside the model; first 10 token embeddings (10×768) | LViT fusion | Text embeddings → Conv1d text branch → added to the image tokens of the first ViT stage (`downVit.CTBN3`), then propagated through the LViT ViT/CNN double U-shape. 39.9M parameters, image branch from scratch. Variant `LViT` (no U-Net pretraining). Adam lr 3e-4, batch 2 (repository `Config.py`, Decision 2), early stopping 50. |
| RecLMIS | ShawnHuang497/RecLMIS | 224×224 | CLIP ViT-B/32 text transformer (official `ViT-B-32.pt`), frozen; `clip.tokenize`, 18 tokens | conditioned reconstruction | Conditioned interaction (text ↔ image cross-attention at 14×14) feeds the decoder. Conditioned language and vision reconstruction from masked features, plus conditioned contrastive loss, during training. 220.7M / 208.1M parameters (QaTa / MosMed config; 69.4M / 56.8M trainable). Dataset-specific official configs (`Config_covid19`, `Config_MosMedPlus`). |
| DD-CMD | maklachur/DD-CMD | 224×224 | BiomedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`), frozen; 24 tokens | dual-domain gated decoding | At 3 decoder stages: text-guided cross-attention with a learned sigmoid gate (spatial domain), then text-conditioned DCT frequency modulation. Text FiLM in the 2 high-resolution refinement blocks. ConvNeXt-Tiny encoder (ImageNet-pretrained, trained). 157.3M parameters (47.8M trainable). Official `train.yaml`; checkpoint and early stopping monitor validation Dice instead of val_MIoU. |

Frozen configurations: `configs/lvit.yaml`, `configs/reclmis.yaml`, `configs/ddcmd.yaml`. At start-up, each wrapper
reads every training setting from the official code and asserts that it matches the frozen file. A run therefore
fails if either the official code or the frozen file changes.

## Data for the official loaders

Each official dataset class expects its own folder and text format. `src/data/build_phase1_views.py` builds them in
`data/views/` from the common data alone:
- images and masks are symlinks to `data/prepared/` (Phase 0, 224×224);
- texts are the verbatim Task 1.1 texts (`data/splits/*_text.csv`);
- file names are `<sample_id>.png`, so image, mask and text are matched by the same name.

| View | Used by | Layout |
|---|---|---|
| `data/views/lvit/<dataset>/{Train,Val,Test}_Folder/` | LViT-T, RecLMIS | `img/`, `labelcol/`, `<Split>_text.xlsx` (`Image`, `Description`), the LViT README format |
| `data/views/ddcmd/<dataset>/` | DD-CMD | `{Train,Val,Test}_Folder/{img,GTs}/`, `prompt/<Split>_text.csv` (`Image`, `text`) |

How each loader reads the same data:
- **LViT and RecLMIS** read the 1-channel image with `cv2.imread` as 3 identical channels, which is exactly the
  Phase 0 preparation. Their 224×224 resize is a no-op.
- **DD-CMD** reads the image with MONAI's PIL reader as 1 channel, and the model repeats it to 3. The 224×224
  resize is a no-op (checked).
- **Text padding:** LViT and RecLMIS pad texts shorter than 9 words with `' EOF XXX'`, as their official
  `read_text` does. DD-CMD uses the text as is.

## Training settings

| Setting | LViT-T | RecLMIS — QaTa-COV19 | RecLMIS — MosMedData+ | DD-CMD |
|---|---|---|---|---|
| Official source | `Config.py`, `train_model.py` | `Config_covid19.py` | `Config_MosMedPlus.py` | `config/train.yaml`, `net/creratemodel.py`, `train.py` |
| Input scaling | uint8 / 255 | uint8 / 255 | uint8 / 255 | z-score per image |
| Loss | 0.5 Dice + 0.5 BCE (`WeightedDiceBCE`) | 5·(0.5 Dice + 0.5 BCE) + 0.2·contrastive + 1·text rec. + 1·image rec. | same | `DiceCELoss(sigmoid=True)` |
| Optimizer | Adam | Adam | Adam | AdamW (weight decay 0.01) |
| Learning rate | 3e-4 (QaTa and MosMed; Decision 2) | 3e-4 | 3e-4 | 5e-5 |
| Scheduler | CosineAnnealingWarmRestarts (T_0 = 10, η_min = 1e-4), per epoch | CosineAnnealingWarmRestarts (T_0 = 10, η_min = 1e-4), per epoch | LambdaLR max(0.99^epoch, 0.1), per epoch | CosineAnnealingLR (T_max = 200, η_min = 1e-6), per epoch |
| Batch size | 2 | 32 | 20 | 8 |
| Augmentation | LViT `RandomGenerator` (rot90 + flip, or rotation ±20°) | Gaussian blur + photometric distortion, then `RandomGenerator` | same | random zoom, rotation, Gaussian noise, flips, rot90 (MONAI) |
| Model-specific | `get_CTranS_config()`, base channel 64; 1 Transformer block per ViT module (= LViT-T in the paper; `transformer.num_layers` is not used by `LViT.py`) | mask rates text 0.3 / image 0.5, 4 reconstruction layers, no dropout | mask rates 0.3 / 0.3, 3 layers, dropout 0.5 | text length 24 |
| Maximum epochs | 2000 | 2000 | 400 | 160 (min 1) |
| Early stopping | 50 epochs | 100 epochs | 30 epochs | 50 epochs |
| Best checkpoint from | epoch 6 (official: `epoch + 1 > 5`) | epoch 1 | epoch 1 | epoch 1 |
| Data loader workers | 8 | 8 | 8 | 0 |
| Seeds | 1001, 1002, 1003 | 1001, 1002, 1003 | 1001, 1002, 1003 | 1001, 1002, 1003 |
| Environment | `lgms_lvit` (torch 1.8.0) | `lgms_reclmis` (torch 1.8.0) | `lgms_reclmis` | `lgms_ddcmd` (torch 1.12.1, Lightning 1.9.0) |

**Checkpoint selection (all methods).** The checkpoint is chosen by validation Dice computed exactly as in Phase 0.
For each LViT validation image, Dice is computed at 224×224 with threshold 0.5, and the scores are averaged over the
images. `src/phase1/common.py:dice` gives results identical to the Phase 0 evaluator's Dice, checked on 2000 random
mask pairs including empty masks. No test data is used for any setting or checkpoint choice.

## Decisions

| | Decision | Reason |
|---|---|---|
| 1 | LViT-T: model `LViT`, not `LViT_pretrain`. | `Config.py` uses `model_name = 'LViT'` by default. `LViT_pretrain` loads a hard-coded MoNuSeg checkpoint (`MoNuSeg/LViT/Test_session_05.23_10h55/...`) that is not distributed and belongs to another dataset. The paper's implementation details describe no pretraining for QaTa-COV19 or MosMedData+. The README's only "w/o pretrain" row is for MoNuSeg. |
| 2 | LViT-T: repository settings, batch 2 and lr 3e-4 for both datasets (decided 2026-11-12). | Repository and paper disagree. `Config.py` sets batch 2 ("For LViT-T, 2 is better than 4") and lr 3e-4 for Covid19, and has **no MosMedData+ value** (only "MoNuSeg: 1e-3, Covid19: 3e-4"). The paper (Sec. IV-C) states lr 3e-4 for QaTa-COV19 and **1e-3 for MosMedData+**, and a default batch size of **24**. In Phase 0 the U-Net used 3e-4 on MosMed with the stated aim of sharing LViT-T's recipe (PHASE0_MODEL_SETUP.md, Decision C). The repository values were chosen for three reasons: they are the code the authors released, RecLMIS and DD-CMD also follow their repositories, and the Phase 0 U-Net and LViT-T keep one identical recipe. Not used: the paper's MosMedData+ learning rate 1e-3 and batch size 24. |
| 3 | RecLMIS: the dataset's official config module is imported directly. | The official `train_model.py` maps `-c Config_MosMedPlus` to `Config_covid19`; only `-c Config_Kvasir_Clinic` loads `Config_MosMedPlus`. Following `train.sh 0 Config_MosMedPlus` literally would train MosMedData+ with the QaTa config. The repository's `test_model.py` maps `Config_MosMedPlus` correctly, which confirms the intended pairing. |
| 4 | RecLMIS: one GPU, official batch sizes (32 / 20). | The authors used 4 RTX 3090s with `nn.DataParallel`, which the code turns on automatically when several GPUs are visible. The batch sizes fit on one 3090 (peak 15.8 GB / 10.0 GB), so the effective batch is unchanged. With one GPU, BatchNorm statistics use the full batch instead of 4 sub-batches. |
| 5 | RecLMIS: repository settings, not paper settings. | As the Phase 1 document instructs ("official QaTa/MosMed implementation … specified by the repository"). The paper reports 200 / 500 epochs, a cosine schedule for both datasets, batch 32 and 3 reconstruction layers. The repository configs use 2000 / 400 maximum epochs with early stopping (patience 100 / 30), an exponential schedule and batch 20 for MosMed, and 4 layers for QaTa. |
| 6 | DD-CMD: `ModelCheckpoint` and `EarlyStopping` monitor validation Dice (`val_common_dice`) instead of `val_MIoU`. | Phase 1 requires checkpoint selection by validation Dice. Officially both callbacks share one monitor, so both were switched. The value comes from the same validation outputs, in a subclass of the official `CreateModel`. All other `train.yaml` values are used as given; they match the paper (AdamW, lr 5e-5 → 1e-6 cosine, batch 8, 160 epochs, text length 24). |
| 7 | DD-CMD: images and masks are transposed by the official loader. Saved predictions are transposed back. | MONAI's `PILReader` returns arrays in (W, H) order. Image and mask are transposed together, so training and Dice are unaffected. Before saving, predictions are transposed back to the original orientation, and the check confirms they match the prepared masks. |
| 8 | Seeds 1001–1003 replace the official seeds (LViT 666, RecLMIS 3407, DD-CMD 42). | Phase 1 seed protocol. The seeding code is otherwise the official one: LViT and RecLMIS also set cuDNN deterministic, and DD-CMD does not. |
| 9 | Side outputs of the official scripts are not reproduced. | Dropped: TensorBoard writers, validation visualizations, per-batch `torch.cuda.empty_cache()`, `thop` FLOP profiling, DD-CMD's code snapshot, last-epoch checkpoints, and RecLMIS's resume logic. None of these affect training. The one exception is LViT's `thop.profile` call on a random input before training, which would move BatchNorm running statistics by one step. |
| 10 | RecLMIS: the reconstruction branch is monitored. | The official `forward` wraps the reconstruction branch in `try/except` and permanently sets `self.aux = False` if it ever raises. Training would then continue silently without conditioned reconstruction. The wrapper logs `aux` every epoch (in `history.csv`), stores it in the checkpoint, and prints a warning if it turns off. It never turned off in any check. |

## Checks (Tasks 1.2, 1.3, 1.4)

Run with `bash src/checks/run_phase1_quick_checks.sh <gpu>` (2026-11-12, RTX 3090; logs in
`runs/quick_check/logs/`). For each method and dataset, the check does the following:
- runs the official loaders on all three splits;
- runs one forward/backward pass on a training batch;
- trains briefly on the first 256 training and first 64 validation images (10 epochs, seed 1001);
- predicts the validation subset with the best checkpoint.

These numbers only show that the models work. They are not results. The checks were run twice with the final code;
LViT-T and RecLMIS gave identical numbers, and DD-CMD differed only slightly, because its official code does not set
cuDNN to deterministic mode. The final run's numbers are shown.

**1. Image, mask and text pairing (all methods, both datasets, train / val / test).**
- Every image in every split is found by the official loader, together with its mask and its own text.
- For 16 random samples per split, the full official `__getitem__` output was compared with the expected values:
  - LViT-T and RecLMIS: the image equals the prepared image on all 3 channels, and the mask equals the prepared mask.
  - DD-CMD: the image equals the z-scored prepared image, transposed, and the mask equals the transposed mask.
  - The text input equals the method's encoding of that sample's Task 1.1 text: LViT BertEmbedding (10×768),
    RecLMIS `clip.tokenize` (18 tokens) and mask, DD-CMD BiomedBERT `input_ids` and `attention_mask` (24 tokens).

**2. Official pretrained components.**
- **RecLMIS:** all 302 tensors of `model.clip` equal the official `ViT-B-32.pt` and none are missing. All CLIP
  parameters are frozen, and the weights are unchanged after training.
- **DD-CMD:** all 200 BiomedBERT tensors and all 180 ConvNeXt-Tiny tensors equal the pinned checkpoints. The text
  encoder is frozen and the vision encoder is trainable.
- **LViT-T:** the text encoder is the `bert-embedding` default model (hash-verified,
  `reports/PHASE1_PUBLIC_IMPLEMENTATIONS_CHECK.md`). It runs inside the data loader, and the check-1 embeddings come
  from it.

**3. Forward pass.**

| Method | Batch | Output |
|---|---|---|
| LViT-T | images 2×3×224×224, masks 2×224×224 ({0,1}), text 2×10×768 | 2×1×224×224 sigmoid |
| RecLMIS | 32×3×224×224 (MosMed 20), tokens 32×18 + mask | 32×1×224×224 sigmoid, plus 4 losses |
| DD-CMD | 8×1×224×224, `input_ids` / `attention_mask` 8×24 | 8×1×224×224 logits |

All losses are finite.

**4. Text / cross-modal path is active.**

| Method | Evidence |
|---|---|
| LViT-T (text reaches the language branch) | The segmentation loss gives non-zero gradients in `text_module4`→`text_module1` and in `downVit.CTBN3`, where the text is added to the image tokens (QaTa: 2.1e-1 … 2.5e-2; MosMed: 2.7e-1 … 3.3e-2). |
| RecLMIS (conditioned reconstruction / fusion) | `model.aux = True` in every epoch. All four losses are present, finite and > 0, for example QaTa first batch: segmentation 0.68, contrastive 3.64, text reconstruction 1.09, image reconstruction 1.00. The total loss gives non-zero gradients in `text_module4`, `interact` (fusion), `mlp_woi`, `mlp_visual`, `rec_text_trans1` and `rec_img_trans1`. The reconstruction losses fall during training: QaTa text 0.73 → 0.55, image 0.62 → 0.02. |
| DD-CMD (cross-modal decoder) | Text-guided attention and DCT text modulation run at all 3 decoder stages, with attention gate g ≈ 0.49–0.50 at initialisation. The segmentation loss gives non-zero gradients in each stage's `fuse.txt_proj`, `fuse.cross`, `fuse.gate_mlp`, `spectral.txt_proj`, and in the refinement FiLM. No gradient reaches the frozen text encoder. |

**5. Training behaves normally / validation Dice improves** (10 short epochs on the subset).

| Method | Dataset | Training loss, epoch 1 → 10 | Validation Dice, epoch 1 → best (epoch) |
|---|---|---|---|
| LViT-T | QaTa-COV19 | 0.467 → 0.276 | 0.503 → 0.653 (7) |
| LViT-T | MosMedData+ | 0.536 → 0.190 | 0.551 → 0.755 (9) |
| RecLMIS | QaTa-COV19 | 5.58 → 3.03 (total) | 0.247 → 0.587 (10) |
| RecLMIS | MosMedData+ | 5.36 → 2.64 (total) | 0.048 → 0.702 (8) |
| DD-CMD | QaTa-COV19 | 0.694 → 0.299 | 0.506 → 0.777 (9) |
| DD-CMD | MosMedData+ | 0.889 → 0.400 | 0.180 → 0.759 (10) |

No NaN losses appeared, and RecLMIS's reconstruction branch stayed active in every epoch. Validation Dice for
RecLMIS fluctuates between epochs at this stage, for example 0.00 at QaTa epochs 2–4. It recovers within a few
epochs, and the official selection by the best epoch handles it.

**6. Validation predictions are valid.** The best checkpoint's predictions on the 64 validation images were saved as
PNG and read back. All are 224×224 with values {0, 255}, one per image. Their Dice against the prepared masks, in
the original orientation, equals the validation Dice computed during training in every run. Empty predictions:
1 of 64 (RecLMIS QaTa), 0 in all other runs.

**Effect of the text on the prediction** (informational). The table shows validation-subset Dice of the best
checkpoint with each image's own text versus other images' texts.

| Method | QaTa-COV19: own / other text | MosMedData+: own / other text |
|---|---|---|
| LViT-T | 0.653 / 0.640 (all-zero text 0.638) | 0.755 / 0.755 (all-zero text 0.759) |
| RecLMIS | 0.587 / 0.539 | 0.702 / 0.699 |
| DD-CMD | 0.777 / 0.697 | 0.759 / 0.754 |

After 10 short epochs, the text changes the predictions for all methods except LViT-T on MosMedData+. There the
language branch is connected (check 4) but the short subset training has not yet made the model rely on the text.
These numbers say nothing about the final models.

## Compute estimate

One full-data epoch including the full validation pass was timed with `--timing` (2026-11-12, one RTX 3090 each).
The LViT runs shared the machine's CPUs with other jobs: their text embedding runs on the CPU in the data loader.

| Method | QaTa-COV19 epoch | MosMedData+ epoch | Epochs per run | Estimated time per run (QaTa / MosMed) |
|---|---:|---:|---|---|
| LViT-T | 779 s | 202 s | early stopping (patience 50, max 2000) | ≈ 22–39 h / 8–16 h if it stops after 100–180 / 150–290 epochs (Phase 0 U-Net, same recipe) |
| RecLMIS | 107 s | 40 s | early stopping (patience 100 / 30; max 2000 / 400) | ≈ 6–18 h for 200–600 epochs / ≤ 4.4 h |
| DD-CMD | 134 s | 49 s | ≤ 160 (patience 50) | ≤ 6.0 h / ≤ 2.2 h |

The total for 18 runs is about 150–260 GPU-hours, dominated by LViT-T on QaTa-COV19. On the 4 reliable GPUs (0–3)
this is roughly 2–3 days of wall-clock time.

## Freeze status

All three configurations are final and frozen for the three-seed experiment (Task 1.6): `configs/lvit.yaml`,
`configs/reclmis.yaml`, `configs/ddcmd.yaml`. Each run logs the git commit of the code it used.
