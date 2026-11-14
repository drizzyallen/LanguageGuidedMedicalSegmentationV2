"""Shared helpers for the Phase 1 wrappers (LViT-T, RecLMIS, DD-CMD).

Runs in all three method environments (Python 3.7-3.9), so it only uses numpy, pandas, PIL, yaml and torch.
The official code of each method is imported from external/<repo> (pinned submodules) and never modified.
"""
import csv
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXTERNAL = os.path.join(ROOT, "external")
PRETRAINED = os.path.join(EXTERNAL, "pretrained")
VIEWS = os.path.join(ROOT, "data", "views")
SPLITS = os.path.join(ROOT, "data", "splits")
PREPARED = os.path.join(ROOT, "data", "prepared")
THRESHOLD = 0.5
SIZE = 224
FOLDER = {"train": "Train", "val": "Val", "test": "Test"}


def load_config(name):
    with open(os.path.join(ROOT, "configs", "{}.yaml".format(name))) as f:
        return yaml.safe_load(f)


def check_matches(logger, official, frozen, what):
    """Assert that a value read from the official code equals the frozen value in configs/*.yaml."""
    assert official == frozen, "{}: official {!r} != frozen {!r}".format(what, official, frozen)
    logger.info("  {} = {!r} (official = frozen config)".format(what, official))


def set_seed(seed):
    # Same seeding as the official train_model.py of LViT / RecLMIS (cudnn deterministic).
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)  # as in the official LViT / RecLMIS loaders
    return worker_init_fn


def run_dir(method, dataset, seed, quick=False, timing=False):
    sub = ["quick_check"] if quick else (["timing"] if timing else [])
    parts = [ROOT, "runs"] + sub + [method, dataset, "seed_{}".format(seed)]
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def git_commit():
    """Commit of this repository (and whether the working tree had changes) for the run log."""
    import subprocess
    try:
        head = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"]).decode().strip()
        dirty = subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain", "--", "src", "configs"]).decode()
        return head + (" (uncommitted changes in src/ or configs/)" if dirty.strip() else "")
    except Exception as e:  # noqa: BLE001
        return "unknown ({})".format(e)


def get_logger(path):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.FileHandler(path), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


class History:
    def __init__(self, path, fields):
        self.path, self.fields = path, fields
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(fields)

    def add(self, **row):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in self.fields])


def dice(pred, target):
    """Per-image Dice; identical to src/evaluation/metrics.py:dice (Phase 0), which needs medpy.
    Both empty -> 1.0."""
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, target).sum() / denom


def view_dir(layout, dataset, split):
    """Root of a split in a method's dataset view (data/views/, built by src/data/build_phase1_views.py)."""
    return os.path.join(VIEWS, layout, dataset, "{}_Folder".format(FOLDER[split]))


def split_texts(dataset, split):
    """sample_id -> verbatim text (Task 1.1)."""
    df = pd.read_csv(os.path.join(SPLITS, dataset, "{}_text.csv".format(split)), keep_default_na=False)
    return dict(zip(df["sample_id"], df["text"]))


def split_ids(dataset, split):
    return list(pd.read_csv(os.path.join(SPLITS, dataset, "{}.csv".format(split)))["sample_id"])


def prepared_image(dataset, split, sample_id):
    return np.array(Image.open(os.path.join(PREPARED, dataset, split, "images", "{}.png".format(sample_id))))


def prepared_mask(dataset, split, sample_id):
    return np.array(Image.open(os.path.join(PREPARED, dataset, split, "masks", "{}.png".format(sample_id))))


def save_mask(out_dir, sample_id, pred):
    """Binary 224x224 prediction -> PNG (0 / 255), same format as Phase 0."""
    pred = np.asarray(pred, bool)
    assert pred.shape == (SIZE, SIZE), pred.shape
    Image.fromarray(pred.astype(np.uint8) * 255).save(os.path.join(out_dir, "{}.png".format(sample_id)))


def check_saved_predictions(logger, out_dir, ids, dataset, split, ref_dice):
    """Read saved predictions back: 224x224, values {0,255}, one per sample, and Dice against the prepared masks
    (in the original orientation) equals the Dice computed during validation."""
    scores, empty = [], 0
    for sample_id in ids:
        pred = np.array(Image.open(os.path.join(out_dir, "{}.png".format(sample_id))))
        assert pred.shape == (SIZE, SIZE) and pred.dtype == np.uint8, (pred.shape, pred.dtype)
        assert set(np.unique(pred).tolist()) <= {0, 255}, np.unique(pred)
        empty += int(pred.max() == 0)
        scores.append(dice(pred > 0, prepared_mask(dataset, split, sample_id) > 0))
    assert len(os.listdir(out_dir)) == len(ids)
    mean = float(np.mean(scores))
    assert abs(mean - ref_dice) < 1e-9, (mean, ref_dice)
    logger.info("  {} predictions saved and re-read: all 224x224, values in {{0, 255}}; {} empty; "
                "Dice vs prepared masks {:.4f} = validation Dice {:.4f}".format(len(ids), empty, mean, ref_dice))
    return mean


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def grad_norm(module):
    norms = [p.grad.detach().norm().item() for p in module.parameters() if p.grad is not None]
    return float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0
