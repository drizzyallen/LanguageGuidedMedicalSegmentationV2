"""Task 1.1: pair the official LViT text annotations with the Phase 0 splits.

The splits are the Phase 0 ones (data/splits/{qata,mosmed}/{split}.csv); no new split is created.
Text is taken verbatim from the LViT spreadsheets (no cleaning, no padding); any method-specific text
handling (e.g. LViT's ' EOF XXX' padding, tokenization) stays in that method's own code.

Inputs (datasets/, not committed; same files as Phase 0, see reports/PHASE0_DATA_SUMMARY.md):
  QaTa-COV19  train + val: LViT-TextAnnotationSplit-WithoutTEXT/WITH-TEXT-NONSPLIT/Train_text_for_Covid19.xlsx
                           (LViT "Train_Val_text" file, 7145 rows = train 5716 + val 1429), column Description
              test:        LViT-TextAnnotationSplit-WithoutTEXT/Test_text_for_Covid19.xlsx, column Description
  MosMedData+ train/val/test: LViT-TextAnnotationSplit/{Train,Val}_text_MosMedData+ 1.xlsx,
                              Test_text_MosMedData+.xlsx, column text

Outputs (committed):
  data/splits/{qata,mosmed}/{train,val,test}_text.csv   sample_id,source_key,text
     same rows and order as {split}.csv; source_key = the spreadsheet "Image" key the text came from.

Usage: python src/data/build_text.py
"""
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "datasets")
SPLITS = os.path.join(ROOT, "data", "splits")

QATA = os.path.join(SRC, "QaTa-COV19", "LViT-TextAnnotationSplit-WithoutTEXT")
MOSMED = os.path.join(SRC, "MosMedData+", "LViT-TextAnnotationSplit")
TEXT_FILES = {
    "qata": {
        "train": os.path.join(QATA, "WITH-TEXT-NONSPLIT", "Train_text_for_Covid19.xlsx"),
        "val": os.path.join(QATA, "WITH-TEXT-NONSPLIT", "Train_text_for_Covid19.xlsx"),
        "test": os.path.join(QATA, "Test_text_for_Covid19.xlsx"),
    },
    "mosmed": {
        "train": os.path.join(MOSMED, "Train_text_MosMedData+ 1.xlsx"),
        "val": os.path.join(MOSMED, "Val_text_MosMedData+ 1.xlsx"),
        "test": os.path.join(MOSMED, "Test_text_MosMedData+.xlsx"),
    },
}
TEXT_COLUMN = {"qata": "Description", "mosmed": "text"}
EXPECTED = {
    "qata": {"train": 5716, "val": 1429, "test": 2113},
    "mosmed": {"train": 2183, "val": 273, "test": 273},
}


def source_key(dataset, sample_id):
    # QaTa spreadsheets are keyed by mask name, MosMed by image name (both .png).
    return f"mask_{sample_id}.png" if dataset == "qata" else f"{sample_id}.png"


def read_text(path, column):
    df = pd.read_excel(path, usecols=["Image", column])
    keys = df["Image"].astype(str).str.strip()
    assert not keys.duplicated().any(), f"duplicate keys in {path}"
    assert df[column].notna().all(), f"missing text in {path}"
    text = df[column].astype(str)
    assert (text.str.strip() != "").all(), f"empty text in {path}"
    return dict(zip(keys, text))


def main():
    for dataset, files in TEXT_FILES.items():
        used = {}
        for split, path in files.items():
            text = read_text(path, TEXT_COLUMN[dataset])
            ids = pd.read_csv(os.path.join(SPLITS, dataset, f"{split}.csv"))["sample_id"]
            assert len(ids) == EXPECTED[dataset][split]
            keys = [source_key(dataset, s) for s in ids]
            missing = [k for k in keys if k not in text]
            assert not missing, f"{dataset}/{split}: {len(missing)} IDs without text, e.g. {missing[:3]}"
            out = pd.DataFrame({"sample_id": ids, "source_key": keys, "text": [text[k] for k in keys]})
            out.to_csv(os.path.join(SPLITS, dataset, f"{split}_text.csv"), index=False)
            used.setdefault(path, set()).update(keys)
            print(f"[{dataset}] {split}: {len(out)} / {len(ids)} samples paired with text "
                  f"({out['text'].nunique()} distinct texts)")
        # Every spreadsheet row is used exactly once: no text left over, no image without text.
        for path, keys in used.items():
            n_rows = len(read_text(path, TEXT_COLUMN[dataset]))
            assert len(keys) == n_rows, f"{path}: {n_rows} rows, {len(keys)} used"
            print(f"[{dataset}] {os.path.basename(path)}: all {n_rows} rows used")


if __name__ == "__main__":
    main()
