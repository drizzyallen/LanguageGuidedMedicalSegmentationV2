"""Task 0.1: extract raw data and build the official LViT splits (image IDs only, no text).

Inputs  (datasets/, not committed):
  QaTa-COV19/Images-notSplitted/QaTa-COV19-v2.zip         Train Set/{Images,Ground-truths}, Test Set/{...}
  QaTa-COV19/LViT-TextAnnotationSplit-WithoutTEXT/         Train_ID.xlsx, Val_ID.xlsx, Test_text_for_Covid19.xlsx
  MosMedData+/Images-notSplitted/archive.zip               frames/, masks/
  MosMedData+/LViT-TextAnnotationSplit/                    Train/Val/Test_text_MosMedData+*.xlsx

Outputs:
  data/raw/{qata,mosmed}/...                               extracted archives (not committed)
  data/splits/{qata,mosmed}/{train,val,test}.csv           sample_id,image,mask (relative to data/raw/<dataset>)

Only the "Image" column of each spreadsheet is read; text annotations are never used.

Usage: python src/data/build_splits.py
"""
import os
import zipfile

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "datasets")
RAW = os.path.join(ROOT, "data", "raw")
SPLITS = os.path.join(ROOT, "data", "splits")

EXPECTED = {
    "qata": {"train": 5716, "val": 1429, "test": 2113},
    "mosmed": {"train": 2183, "val": 273, "test": 273},
}

QATA_DIR = os.path.join(SRC, "QaTa-COV19", "LViT-TextAnnotationSplit-WithoutTEXT")
MOSMED_DIR = os.path.join(SRC, "MosMedData+", "LViT-TextAnnotationSplit")
SPLIT_FILES = {
    "qata": {
        "train": os.path.join(QATA_DIR, "Train_ID.xlsx"),
        "val": os.path.join(QATA_DIR, "Val_ID.xlsx"),
        "test": os.path.join(QATA_DIR, "Test_text_for_Covid19.xlsx"),
    },
    "mosmed": {
        "train": os.path.join(MOSMED_DIR, "Train_text_MosMedData+ 1.xlsx"),
        "val": os.path.join(MOSMED_DIR, "Val_text_MosMedData+ 1.xlsx"),
        "test": os.path.join(MOSMED_DIR, "Test_text_MosMedData+.xlsx"),
    },
}
ARCHIVES = {
    "qata": os.path.join(SRC, "QaTa-COV19", "Images-notSplitted", "QaTa-COV19-v2.zip"),
    "mosmed": os.path.join(SRC, "MosMedData+", "Images-notSplitted", "archive.zip"),
}


def extract(name):
    out = os.path.join(RAW, name)
    if os.path.isdir(out) and os.listdir(out):
        print(f"[{name}] already extracted: {out}")
        return out
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(ARCHIVES[name]) as z:
        for member in z.namelist():
            target = os.path.realpath(os.path.join(out, member))
            assert target.startswith(os.path.realpath(out) + os.sep), f"unsafe path in archive: {member}"
        z.extractall(out)
    print(f"[{name}] extracted to {out}")
    return out


def read_ids(path):
    ids = pd.read_excel(path, usecols=["Image"])["Image"].astype(str).str.strip()
    assert not ids.duplicated().any(), f"duplicate IDs in {path}"
    return list(ids)


def qata_rows(split, mask_ids):
    # Split IDs are mask names ("mask_<image>"); train/val come from "Train Set", test from "Test Set".
    folder = "Test Set" if split == "test" else "Train Set"
    rows = []
    for mask_name in mask_ids:
        assert mask_name.startswith("mask_"), mask_name
        image_name = mask_name[len("mask_"):]
        rows.append({
            "sample_id": os.path.splitext(image_name)[0],
            "image": f"{folder}/Images/{image_name}",
            "mask": f"{folder}/Ground-truths/{mask_name}",
        })
    return rows


def mosmed_rows(split, ids):
    return [{"sample_id": os.path.splitext(i)[0], "image": f"frames/{i}", "mask": f"masks/{i}"} for i in ids]


def build(name, make_rows):
    raw = extract(name)
    os.makedirs(os.path.join(SPLITS, name), exist_ok=True)
    seen = {}
    for split in ("train", "val", "test"):
        df = pd.DataFrame(make_rows(split, read_ids(SPLIT_FILES[name][split])))
        assert len(df) == EXPECTED[name][split], f"{name}/{split}: {len(df)} != {EXPECTED[name][split]}"
        missing = [p for p in list(df["image"]) + list(df["mask"]) if not os.path.isfile(os.path.join(raw, p))]
        assert not missing, f"{name}/{split}: {len(missing)} missing files, e.g. {missing[:3]}"
        for other, ids in seen.items():
            overlap = set(df["sample_id"]) & ids
            assert not overlap, f"{name}: {split}/{other} overlap {len(overlap)}"
        seen[split] = set(df["sample_id"])
        df.to_csv(os.path.join(SPLITS, name, f"{split}.csv"), index=False)
        print(f"[{name}] {split}: {len(df)} samples, all image/mask files present")


def main():
    build("qata", qata_rows)
    build("mosmed", mosmed_rows)


if __name__ == "__main__":
    main()
