"""Task 1.1: check that the text annotations are paired with the right images.

For every sample, the extent word of the text ("Unilateral" / "Bilateral" / "No" pulmonary infection) is compared
with the 224x224 mask: lesion pixels in both image halves (smaller half > 5% of the lesion) = bilateral,
in one half = unilateral, none = no infection. Correct pairing gives high agreement; the same comparison with the
texts randomly shuffled within the split (100 permutations, seed 0) gives the chance level.
This is a heuristic consistency check, not a measure of annotation quality.

Inputs:  data/splits/{qata,mosmed}/{split}_text.csv, data/prepared/{qata,mosmed}/{split}/masks/
Usage:   python src/checks/check_phase1_text_pairing.py
"""
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLITS = os.path.join(ROOT, "data", "splits")
PREPARED = os.path.join(ROOT, "data", "prepared")


def mask_extent(mask):
    half = mask.shape[1] // 2
    left, right = int(mask[:, :half].sum()), int(mask[:, half:].sum())
    total = left + right
    if total == 0:
        return "no"
    return "bilateral" if min(left, right) > 0.05 * total else "unilateral"


def main():
    rng = np.random.default_rng(0)
    print("| Dataset | Split | n | Text: uni / bi / no | Mask: uni / bi / no | Agreement | Shuffled text |")
    print("|---|---|---:|---|---|---:|---:|")
    for dataset in ("qata", "mosmed"):
        for split in ("train", "val", "test"):
            df = pd.read_csv(os.path.join(SPLITS, dataset, f"{split}_text.csv"))
            ids = pd.read_csv(os.path.join(SPLITS, dataset, f"{split}.csv"))["sample_id"]
            assert list(df["sample_id"]) == list(ids), f"{dataset}/{split}: text rows differ from split"
            text = df["text"].str.split().str[0].str.lower().to_numpy()
            assert set(text) <= {"unilateral", "bilateral", "no"}, set(text)
            masks = np.array([mask_extent(cv2.imread(os.path.join(PREPARED, dataset, split, "masks", f"{s}.png"), 0))
                              for s in df["sample_id"]])
            agree = (text == masks).mean()
            shuffled = np.mean([(rng.permutation(text) == masks).mean() for _ in range(100)])
            count = lambda a: " / ".join(str(int((a == k).sum())) for k in ("unilateral", "bilateral", "no"))
            print(f"| {dataset} | {split} | {len(df)} | {count(text)} | {count(masks)} | "
                  f"{100 * agree:.1f}% | {100 * shuffled:.1f}% |")


if __name__ == "__main__":
    main()
