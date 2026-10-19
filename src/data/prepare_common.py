"""Task 0.2: common 224x224 image/mask preparation for all methods.

Follows the official LViT loader (external/LViT/Load_Dataset.py, ImageToImage2D.__getitem__):
  image = cv2.imread(path); image = cv2.resize(image, (224, 224))
  mask  = cv2.imread(path, 0); mask = cv2.resize(mask, (224, 224)); mask[mask <= 0] = 0; mask[mask > 0] = 1

Images are grayscale (QaTa: 1 channel; MosMed: 3 identical channels) and are stored as one channel;
models that need 3 channels duplicate it at load time.

Inputs:  data/splits/{qata,mosmed}/{train,val,test}.csv, data/raw/{qata,mosmed}/
Outputs: data/prepared/{qata,mosmed}/{train,val,test}/{images,masks}/<sample_id>.png
         images: uint8 224x224 grayscale; masks: uint8 224x224 in {0, 1}

Usage: python src/data/prepare_common.py
"""
import os

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW = os.path.join(ROOT, "data", "raw")
SPLITS = os.path.join(ROOT, "data", "splits")
OUT = os.path.join(ROOT, "data", "prepared")
SIZE = 224

EXPECTED = {
    "qata": {"train": 5716, "val": 1429, "test": 2113},
    "mosmed": {"train": 2183, "val": 273, "test": 273},
}


def prepare_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 3:
        assert all(np.array_equal(image[..., 0], image[..., c]) for c in range(1, image.shape[2])), \
            f"image is not grayscale: {path}"
        image = image[..., 0]
    image = cv2.resize(image, (SIZE, SIZE))
    # Same result as the LViT loader (3-channel imread + resize), per channel.
    lvit = cv2.resize(cv2.imread(path), (SIZE, SIZE))
    assert all(np.array_equal(image, lvit[..., c]) for c in range(3)), f"differs from LViT loader: {path}"
    return image


def prepare_mask(path):
    mask = cv2.imread(path, 0)
    mask = cv2.resize(mask, (SIZE, SIZE))
    mask[mask <= 0] = 0
    mask[mask > 0] = 1
    return mask


def main():
    for dataset, splits in EXPECTED.items():
        for split, expected in splits.items():
            df = pd.read_csv(os.path.join(SPLITS, dataset, f"{split}.csv"))
            assert len(df) == expected, f"{dataset}/{split}: {len(df)} != {expected}"
            img_dir = os.path.join(OUT, dataset, split, "images")
            mask_dir = os.path.join(OUT, dataset, split, "masks")
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            empty, fg = 0, []
            for row in df.itertuples():
                image = prepare_image(os.path.join(RAW, dataset, row.image))
                mask = prepare_mask(os.path.join(RAW, dataset, row.mask))
                assert image.shape == (SIZE, SIZE) and image.dtype == np.uint8
                assert mask.shape == (SIZE, SIZE) and set(np.unique(mask)) <= {0, 1}
                assert cv2.imwrite(os.path.join(img_dir, f"{row.sample_id}.png"), image)
                assert cv2.imwrite(os.path.join(mask_dir, f"{row.sample_id}.png"), mask)
                empty += int(mask.max() == 0)
                fg.append(mask.mean())

            n_img, n_mask = len(os.listdir(img_dir)), len(os.listdir(mask_dir))
            assert n_img == n_mask == expected, f"{dataset}/{split}: {n_img} images, {n_mask} masks"
            print(f"[{dataset}] {split}: {expected} samples, 224x224, masks in {{0,1}}, "
                  f"empty masks {empty}, mean lesion fraction {np.mean(fg):.4f}")


if __name__ == "__main__":
    main()
