"""Convert data/prepared/ (Task 0.2) to nnU-Net v2 raw format, plan + preprocess 2d, and fix the LViT split.

  imagesTr/labelsTr = LViT train + val (nnU-Net trains on fold 0 train, validates on fold 0 val)
  imagesTs/labelsTs = LViT test (only used for final prediction in Task 0.4)
  splits_final.json = one fold: {"train": LViT train IDs, "val": LViT val IDs}

Images are the prepared 224x224 grayscale PNGs (1 channel); labels are the prepared {0,1} masks.

Usage: python src/nnunet/convert.py --dataset qata
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import image_path, mask_path, sample_ids  # noqa: E402
from nnunet.paths import DATASET_IDS, DATASET_NAMES, NNUNET_PREPROCESSED, NNUNET_RAW, set_env  # noqa: E402


def write_raw(dataset):
    out = os.path.join(NNUNET_RAW, DATASET_NAMES[dataset])
    if os.path.isdir(out):
        shutil.rmtree(out)
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        os.makedirs(os.path.join(out, sub))
    ids = {split: sample_ids(dataset, split) for split in ("train", "val", "test")}
    for split, case_ids in ids.items():
        img_dir, lab_dir = ("imagesTs", "labelsTs") if split == "test" else ("imagesTr", "labelsTr")
        for case in case_ids:
            shutil.copyfile(image_path(dataset, split, case), os.path.join(out, img_dir, f"{case}_0000.png"))
            shutil.copyfile(mask_path(dataset, split, case), os.path.join(out, lab_dir, f"{case}.png"))
    with open(os.path.join(out, "dataset.json"), "w") as f:
        json.dump({"channel_names": {"0": "gray"}, "labels": {"background": 0, "lesion": 1},
                   "numTraining": len(ids["train"]) + len(ids["val"]), "file_ending": ".png"}, f, indent=2)
    print(f"[{dataset}] raw: {len(ids['train']) + len(ids['val'])} training cases, {len(ids['test'])} test cases")
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--num_processes", type=int, default=8)
    args = parser.parse_args()
    set_env(results="")

    ids = write_raw(args.dataset)
    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", str(DATASET_IDS[args.dataset]), "-c", "2d",
                    "--verify_dataset_integrity", "-np", str(args.num_processes)], check=True)

    splits = [{"train": ids["train"], "val": ids["val"]}]
    with open(os.path.join(NNUNET_PREPROCESSED, DATASET_NAMES[args.dataset], "splits_final.json"), "w") as f:
        json.dump(splits, f)
    print(f"[{args.dataset}] splits_final.json: fold 0 = {len(ids['train'])} train / {len(ids['val'])} val")


if __name__ == "__main__":
    main()
