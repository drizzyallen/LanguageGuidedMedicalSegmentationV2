"""Shared helpers for the U-Net and TransUNet training wrappers."""
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

from evaluation.metrics import dice

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THRESHOLD = 0.5


def load_config(name):
    with open(os.path.join(ROOT, "configs", f"{name}.yaml")) as f:
        return yaml.safe_load(f)


def set_seed(seed, deterministic=True):
    # Same seeding as the official LViT / TransUNet scripts.
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_worker_init_fn(seed):
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)  # as in the official LViT / TransUNet loaders
    return worker_init_fn


def run_dir(method, dataset, seed, quick=False):
    base = os.path.join(ROOT, "runs", "quick_check" if quick else "", method, dataset, f"seed_{seed}")
    os.makedirs(base, exist_ok=True)
    return base


def get_logger(path):
    logger = logging.getLogger(path)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
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
            csv.writer(f).writerow([row[k] for k in self.fields])


@torch.no_grad()
def validation_dice(model, loader, lesion_prob):
    """Mean per-image Dice at 224x224 with threshold 0.5. `lesion_prob(model, images)` -> (B, H, W) probabilities."""
    model.eval()
    scores = []
    for images, masks, _ in loader:
        probs = lesion_prob(model, images.cuda()).cpu().numpy()
        masks = masks.numpy()
        for p, m in zip(probs, masks):
            scores.append(dice(p >= THRESHOLD, m > 0))
    model.train()
    return float(np.mean(scores))


def quick_sanity_checks(logger, model, batch, forward, loss_fn, expected_out_channels):
    """Doc Task 0.3 quick check items 1-4 on one batch."""
    images, masks, ids = batch
    logger.info(f"[check 1] loaded one batch: images {tuple(images.shape)} {images.dtype}, "
                f"masks {tuple(masks.shape)} {masks.dtype}, mask values {sorted(masks.unique().tolist())}")
    model.train()
    out = forward(model, images.cuda())
    logger.info(f"[check 2] forward pass ok: output {tuple(out.shape)}")
    assert out.shape[0] == masks.shape[0] and out.shape[1] == expected_out_channels
    assert tuple(out.shape[-2:]) == tuple(masks.shape[-2:]) == (224, 224)
    logger.info(f"[check 3] shapes ok: output spatial {tuple(out.shape[-2:])} == mask spatial {tuple(masks.shape[-2:])}")
    loss = loss_fn(out, masks.cuda())
    assert torch.isfinite(loss)
    logger.info(f"[check 4] loss computed: {loss.item():.4f}")
