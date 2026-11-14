"""Phase 1: build each official repository's dataset layout from the common prepared data (no new data).

The official dataset classes are used unchanged, so each method needs its data in its own folder/text format.
Every view is built from the same files: images and masks are symlinks to data/prepared/ (Phase 0, 224x224),
texts come verbatim from data/splits/{dataset}/{split}_text.csv (Task 1.1). Files are named <sample_id>.png
in both the image and mask folders, so every loader pairs image, mask and text by the same name.

  data/views/lvit/{qata,mosmed}/{Train,Val,Test}_Folder/       LViT-T and RecLMIS (LViT layout)
      img/<id>.png, labelcol/<id>.png, {Train,Val,Test}_text.xlsx   columns Image (= <id>.png), Description
  data/views/ddcmd/{qata,mosmed}/{Train,Val,Test}_Folder/      DD-CMD layout
      img/<id>.png, GTs/<id>.png;  prompt/{Train,Val,Test}_text.csv  columns Image (= <id>.png), text

Usage: python src/data/build_phase1_views.py      (env lgms_phase0)
"""
import os
import shutil

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREPARED = os.path.join(ROOT, "data", "prepared")
SPLITS = os.path.join(ROOT, "data", "splits")
VIEWS = os.path.join(ROOT, "data", "views")
FOLDER = {"train": "Train", "val": "Val", "test": "Test"}
EXPECTED = {
    "qata": {"train": 5716, "val": 1429, "test": 2113},
    "mosmed": {"train": 2183, "val": 273, "test": 273},
}


def link_all(ids, src_dir, dst_dir):
    os.makedirs(dst_dir)
    for sample_id in ids:
        src = os.path.join(src_dir, f"{sample_id}.png")
        assert os.path.isfile(src), src
        os.symlink(os.path.relpath(src, dst_dir), os.path.join(dst_dir, f"{sample_id}.png"))


def main():
    if os.path.isdir(VIEWS):
        shutil.rmtree(VIEWS)
    for dataset, splits in EXPECTED.items():
        for split, n in splits.items():
            text = pd.read_csv(os.path.join(SPLITS, dataset, f"{split}_text.csv"), keep_default_na=False)
            ids = list(pd.read_csv(os.path.join(SPLITS, dataset, f"{split}.csv"))["sample_id"])
            assert list(text["sample_id"]) == ids and len(ids) == n, (dataset, split)
            names = [f"{s}.png" for s in ids]
            images = os.path.join(PREPARED, dataset, split, "images")
            masks = os.path.join(PREPARED, dataset, split, "masks")
            name = FOLDER[split]

            lvit = os.path.join(VIEWS, "lvit", dataset, f"{name}_Folder")
            link_all(ids, images, os.path.join(lvit, "img"))
            link_all(ids, masks, os.path.join(lvit, "labelcol"))
            pd.DataFrame({"Image": names, "Description": text["text"]}).to_excel(
                os.path.join(lvit, f"{name}_text.xlsx"), index=False)

            ddcmd = os.path.join(VIEWS, "ddcmd", dataset)
            link_all(ids, images, os.path.join(ddcmd, f"{name}_Folder", "img"))
            link_all(ids, masks, os.path.join(ddcmd, f"{name}_Folder", "GTs"))
            os.makedirs(os.path.join(ddcmd, "prompt"), exist_ok=True)
            pd.DataFrame({"Image": names, "text": text["text"]}).to_csv(
                os.path.join(ddcmd, "prompt", f"{name}_text.csv"), index=False)

            # Read back: same names, same texts, every link resolves.
            back_x = pd.read_excel(os.path.join(lvit, f"{name}_text.xlsx"))
            back_c = pd.read_csv(os.path.join(ddcmd, "prompt", f"{name}_text.csv"), keep_default_na=False)
            for back, col in ((back_x, "Description"), (back_c, "text")):
                assert list(back["Image"]) == names and list(back[col]) == list(text["text"]), (dataset, split, col)
            for d in (os.path.join(lvit, "img"), os.path.join(lvit, "labelcol"),
                      os.path.join(ddcmd, f"{name}_Folder", "img"), os.path.join(ddcmd, f"{name}_Folder", "GTs")):
                files = sorted(os.listdir(d))
                assert files == sorted(names) and all(os.path.isfile(os.path.join(d, f)) for f in files), d
            print(f"[{dataset}] {split}: {n} samples -> lvit/ and ddcmd/ views (images, masks, text)")


if __name__ == "__main__":
    main()
