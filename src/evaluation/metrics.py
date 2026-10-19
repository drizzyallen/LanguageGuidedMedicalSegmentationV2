"""Metrics on binary 224x224 masks. Threshold 0.5 is applied before calling these functions."""
import numpy as np


def dice(pred, target):
    """Per-image Dice of two binary masks. Both empty -> 1.0."""
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, target).sum() / denom
