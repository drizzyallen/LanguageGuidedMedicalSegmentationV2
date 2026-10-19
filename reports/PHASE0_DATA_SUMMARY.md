# Phase 0 — Data Summary (Task 0.1)

Both datasets use the official LViT train/validation/test splits. No new split was created.
Only images and masks are used; the text annotations in the LViT spreadsheets are never read
(only the `Image` ID column is used).

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| QaTa-COV19 | 5716 | 1429 | 2113 |
| MosMedData+ | 2183 | 273 | 273 |

## Sources

Downloaded from the dataset package linked by the LViT project
(`https://drive.google.com/drive/folders/1BjkeVIYwiV4d94AMDZfBtIk_7MNIb2AH`), stored in `datasets/` (not committed).

| File | Use | SHA-256 |
|---|---|---|
| `QaTa-COV19/Images-notSplitted/QaTa-COV19-v2.zip` | QaTa images + masks | `7d890c64234f81453c9879113b2e036a0b86287c0225eb8df629b87278686c95` |
| `QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/Train_ID.xlsx` | QaTa train IDs | `f59bc19e2f120efab7b9b6e21ea8e55885073f8ac798fc3a2ce108b471162946` |
| `QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/Val_ID.xlsx` | QaTa val IDs | `e1ad082d5fd2e7eb7aaa211781eec1d04f1f7bde2c707373f77780c708b57cc6` |
| `QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/Test_text_for_Covid19.xlsx` | QaTa test IDs (`Image` column only) | `3f349445978a0f58178f174e90caaf1c072c1b5b319ebbe4061242cd40c58cc5` |
| `MosMedData+/Images-notSplitted/archive.zip` | MosMed images + masks | `d83e8f12eae2d06e0ed4ba06f09eeb7f56a18ed581bdad3baee1ce834765bba6` |
| `MosMedData+/LViT-TextAnnotationSplit/Train_text_MosMedData+ 1.xlsx` | MosMed train IDs (`Image` column only) | `032a56032d787b0a575af8c783b0fa751e3bedd94b8f2f565edc31291fe11d8e` |
| `MosMedData+/LViT-TextAnnotationSplit/Val_text_MosMedData+ 1.xlsx` | MosMed val IDs (`Image` column only) | `d23257a11b7e878b6da1af7d8493eb9fb3c5f2018d371dd3fa3b88ee72e4463d` |
| `MosMedData+/LViT-TextAnnotationSplit/Test_text_MosMedData+.xlsx` | MosMed test IDs (`Image` column only) | `6afc6a50c156d30f5a4f3b31ec0c0622869dde286c1767535fd5760df5e92d60` |

Original/public references: QaTa-COV19 — `https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset`;
MosMedData+ — `https://medicalsegmentation.com/covid19/`
(mirror: `https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset`).

## Outputs

- `src/data/build_splits.py` — extracts the archives to `data/raw/` (not committed) and writes the split lists.
- `data/splits/{qata,mosmed}/{train,val,test}.csv` — `sample_id,image,mask` (paths relative to `data/raw/<dataset>/`), committed.

## Verification

| Check | QaTa-COV19 | MosMedData+ |
|---|---|---|
| Split counts match LViT | yes (5716 / 1429 / 2113) | yes (2183 / 273 / 273) |
| Every split ID has an image and a mask | yes | yes |
| No duplicate IDs within a split | yes | yes |
| No ID shared between train / val / test | yes | yes |
| Splits cover the whole archive | train + val = the 7145-image `Train Set`; test = the 2113-image `Test Set` | train + val + test = all 2729 frames |

## Raw data properties (before Task 0.2 preparation)

| | QaTa-COV19 | MosMedData+ |
|---|---|---|
| Image size | 224×224 (all 9258) | 512×512 (all 2729) |
| Image channels | 1 (grayscale PNG) | 3 (PNG) |
| Mask values | {0, 255} (all) | {0, 255} for 2629; 5 distinct values for 99; all-zero for 1 |
| Empty masks | 0 | 1 |

Mask binarization (`> 0 → 1`) after resizing to 224×224 is applied in Task 0.2, following the LViT/RecLMIS loader.

## Notes

- QaTa split spreadsheets list mask names (`mask_<image>.png`); the image is the same name without the `mask_` prefix.
- The LViT MosMedData+ split is slice-level. Some CT volumes have slices in more than one split:
  `Jun_coronacases_case8` (train/val/test), `Jun_radiopaedia_40_86625_0_case18` (train/test),
  `Jun_radiopaedia_4_85506_1_case19` (val/test), `Morozov_study_0295` (train/test), `Morozov_study_0299` (val/test).
  `bjorke_*` slices appear in all three splits; their file names do not identify the patient.
  This is a property of the official benchmark split, which is used unchanged as required.
