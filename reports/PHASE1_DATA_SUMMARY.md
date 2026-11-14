# Phase 1 — Data Summary (Task 1.1)

Phase 1 uses the same datasets, the same official LViT splits and the same common 224×224 image/mask
preparation as Phase 0. No new split was created. The only addition is the text annotation for each image.

| Dataset | Train | Validation | Test | Text Available |
|---|---:|---:|---:|---|
| QaTa-COV19 | 5716 | 1429 | 2113 | Yes |
| MosMedData+ | 2183 | 273 | 273 | Yes |

Every image in every split has exactly one text annotation, and every row of the text spreadsheets is used.

## Sources

Images, masks and splits: unchanged from Phase 0 (`reports/PHASE0_DATA_SUMMARY.md`):
`data/splits/{qata,mosmed}/{train,val,test}.csv` and `data/prepared/{qata,mosmed}/{split}/{images,masks}/`
(uint8 224×224 grayscale images; masks in {0, 1}; same resize and binarization as the LViT loader).

Text: the LViT text-annotation spreadsheets from the dataset package
(`https://drive.google.com/drive/folders/1BjkeVIYwiV4d94AMDZfBtIk_7MNIb2AH`). In Phase 0 (2026-10-19) all 7 local
spreadsheets were compared with the 7 in this Drive folder and found to have identical contents.

| Dataset | Split | Spreadsheet (in `datasets/`) | Key column | Text column |
|---|---|---|---|---|
| QaTa-COV19 | train, val | `QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/WITH-TEXT-NONSPLIT/Train_text_for_Covid19.xlsx` (Drive: `Train_Val_text_for_Covid19`, 7145 rows = 5716 + 1429) | `Image` (`mask_<id>.png`) | `Description` |
| QaTa-COV19 | test | `QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/Test_text_for_Covid19.xlsx` | `Image` (`mask_<id>.png`) | `Description` |
| MosMedData+ | train | `MosMedData+/LViT-TextAnnotationSplit/Train_text_MosMedData+ 1.xlsx` | `Image` (`<id>.png`) | `text` |
| MosMedData+ | val | `MosMedData+/LViT-TextAnnotationSplit/Val_text_MosMedData+ 1.xlsx` | `Image` (`<id>.png`) | `text` |
| MosMedData+ | test | `MosMedData+/LViT-TextAnnotationSplit/Test_text_MosMedData+.xlsx` | `Image` (`<id>.png`) | `text` |

The SHA-256 checksums of all spreadsheets except one are in `reports/PHASE0_DATA_SUMMARY.md`. In Phase 0 only their
`Image` columns were read; now their text columns are used too. The QaTa train/val text file was not used in Phase 0.
Its checksum:

| File | SHA-256 |
|---|---|
| `QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/WITH-TEXT-NONSPLIT/Train_text_for_Covid19.xlsx` | `1a7abc0f3a9fbb46bfce977eccfc99d1cf0d0d9af7dfe7216d70d90be805852e` |

This file pairs QaTa train and val images with their text in the same way as the official LViT and RecLMIS code
(`read_text(task_dataset + 'Train_Val_text.xlsx')`, used for both the train and the val dataset).

## Outputs

- `src/data/build_text.py`: pairs the text with the Phase 0 split IDs.
- `data/splits/{qata,mosmed}/{train,val,test}_text.csv`: `sample_id,source_key,text`, with the same rows in the
  same order as `{split}.csv`. `source_key` is the spreadsheet key the text came from. Committed.
- `src/checks/check_phase1_text_pairing.py`: the pairing check below.

The text is copied verbatim, with no cleaning, padding or truncation. Each method's own handling of the text
stays in that method's code and is set up in Tasks 1.2–1.4. Examples are LViT/RecLMIS `read_text` padding with
`' EOF XXX'`, RecLMIS CLIP tokenization (length 18) and DD-CMD BiomedBERT tokenization (length 24).

## Verification

| Check | QaTa-COV19 | MosMedData+ |
|---|---|---|
| Splits identical to Phase 0 (same IDs, same order) | yes | yes |
| Every split ID has a text | yes (9258 / 9258) | yes (2729 / 2729) |
| Every spreadsheet row used exactly once | yes (7145 + 2113) | yes (2183 + 273 + 273) |
| Duplicate keys or empty texts | none | none |
| Distinct texts (train / val / test) | 273 / 128 / 198 | 101 / 45 / 61 |
| Words per text | 8–17 | 6–13 |

**Pairing check.** Each text starts with "Unilateral", "Bilateral" or "No" pulmonary infection. This is compared
with where the lesion is in the 224×224 mask. Lesion pixels in both image halves (the smaller half holding more
than 5% of the lesion) count as bilateral, pixels in one half as unilateral, and an empty mask as no infection.
If the pairing is correct, agreement is high. The same comparison with the texts randomly shuffled within the split
gives chance level (mean of 100 permutations).

| Dataset | Split | n | Text: uni / bi / no | Mask: uni / bi / no | Agreement | Shuffled text |
|---|---|---:|---|---|---:|---:|
| QaTa-COV19 | train | 5716 | 1166 / 4550 / 0 | 1166 / 4550 / 0 | 99.2% | 67.5% |
| QaTa-COV19 | val | 1429 | 316 / 1113 / 0 | 319 / 1110 / 0 | 99.1% | 65.6% |
| QaTa-COV19 | test | 2113 | 438 / 1675 / 0 | 440 / 1673 / 0 | 98.2% | 67.1% |
| MosMedData+ | train | 2183 | 1079 / 1061 / 43 | 1109 / 1073 / 1 | 96.0% | 49.1% |
| MosMedData+ | val | 273 | 155 / 114 / 4 | 165 / 108 / 0 | 95.6% | 50.9% |
| MosMedData+ | test | 273 | 106 / 167 / 0 | 102 / 171 / 0 | 97.8% | 53.0% |

Agreement is 96–99% on every split, compared with 49–68% for shuffled text, so the texts are paired with the
correct images. The small remainder comes from the 5% threshold and from the annotation itself (see notes).
Left/right also matches the masks. For QaTa unilateral texts, "left lung" means the lesion is mostly in the left half
of the image (902 / 935 cases) and "right lung" means the right half (930 / 963). This is the image-side
convention used by the annotation.

## Annotation properties (official text, kept unchanged)

- **MosMedData+ "No pulmonary infection".** 47 texts (train 43, val 4, test 0) say
  "No pulmonary infection, zero infected areas.", but 46 of these images have a small lesion in the mask
  (median 0.02% of the image). Only one of them has an empty mask: `bjorke_49`, the dataset's only empty mask.
  None are in the test split.
- **MosMedData+ unilateral right-side lesions.** These are written as "Unilateral pulmonary infection, …, left lung
  and \<zone\> right lung." (457 texts). In all 457 the lesion is in the right image half, so the text is
  unilateral right, despite the extra "left lung". MosMed texts never use "right lung" alone.
- **Small typos.** Examples are a doubled final period ("lung..", MosMed 47 texts), double spaces (QaTa 27 texts)
  and leading/trailing spaces (QaTa test, 2 texts). All are kept as in the official files.

## Repositories

All three official repositories are pinned as git submodules and pass the implementation check
(`reports/PHASE1_PUBLIC_IMPLEMENTATIONS_CHECK.md`, `external/SOURCES.md`):

| Method | Repository | Commit | Env | Data format the official code expects |
|---|---|---|---|---|
| LViT-T | `HUANGLIZI/LViT` | `ba90775` | `lgms_lvit` | `{Train,Val,Test}_Folder/{img,labelcol}/` and xlsx with `Image`, `Description` (QaTa: `Train_Val_text.xlsx` for train + val) |
| RecLMIS | `ShawnHuang497/RecLMIS` | `d3265c8` | `lgms_reclmis` | same layout as LViT (`Train_text.xlsx` / `Val_text.xlsx` for MosMed, `Train_Val_text.xlsx` for QaTa, `Test_text.xlsx`) |
| DD-CMD | `maklachur/DD-CMD` | `27387c4` | `lgms_ddcmd` | `{Train,Val,Test}_Folder/{img,GTs}/` and CSV with `Image` + `text`/`Description` (CSV files not shipped with the repo) |

All three formats can be generated directly from `data/prepared/` and `data/splits/*_text.csv`. Each method will
then load exactly the same 224×224 images, masks and texts. The per-method dataset views are built with each
method's wrapper in Tasks 1.2–1.4, without modifying upstream code.

## Milestone 1.1

- [x] Official splits loaded (same as Phase 0, counts match).
- [x] Text annotations paired correctly (complete 1:1 pairing; mask-consistency check passed on all splits).
- [x] All three official repositories available.
