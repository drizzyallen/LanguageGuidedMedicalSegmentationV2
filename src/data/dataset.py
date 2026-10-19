"""Common dataset over data/prepared/ (Task 0.2 output) and the LViT split lists (Task 0.1).

Returns the prepared 224x224 grayscale image (uint8) and mask (uint8, {0,1}); each method applies
its own official transform on top.
"""
import os

import cv2
import pandas as pd
from torch.utils.data import Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PREPARED = os.path.join(ROOT, "data", "prepared")
SPLITS = os.path.join(ROOT, "data", "splits")
DATASETS = ("qata", "mosmed")


def sample_ids(dataset, split):
    return list(pd.read_csv(os.path.join(SPLITS, dataset, f"{split}.csv"))["sample_id"])


def image_path(dataset, split, sample_id):
    return os.path.join(PREPARED, dataset, split, "images", f"{sample_id}.png")


def mask_path(dataset, split, sample_id):
    return os.path.join(PREPARED, dataset, split, "masks", f"{sample_id}.png")


class PreparedDataset(Dataset):
    def __init__(self, dataset, split, transform, limit=None):
        assert dataset in DATASETS, dataset
        self.dataset, self.split, self.transform = dataset, split, transform
        self.ids = sample_ids(dataset, split)
        if limit is not None:
            self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        image = cv2.imread(image_path(self.dataset, self.split, sample_id), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path(self.dataset, self.split, sample_id), cv2.IMREAD_GRAYSCALE)
        image, mask = self.transform(image, mask)
        return image, mask, sample_id
