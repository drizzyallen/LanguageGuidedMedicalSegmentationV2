"""Metrics on binary 224x224 masks (common final evaluator, Tasks 0.4/0.5).

Threshold 0.5 is applied before calling these functions. Distances (HD95, ASSD) are in 224x224 pixel units.
If exactly one of prediction / target is empty, HD95 and ASSD are undefined; they are set to the image
diagonal (worst possible distance). Both empty -> Dice = IoU = 1, HD95 = ASSD = 0.
"""
import numpy as np
from medpy.metric import binary

IMAGE_SIZE = 224
WORST_DISTANCE = float(np.sqrt(2) * IMAGE_SIZE)  # 316.78 px


def dice(pred, target):
    """Per-image Dice of two binary masks. Both empty -> 1.0."""
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return 2.0 * np.logical_and(pred, target).sum() / denom


def iou(pred, target):
    """Per-image lesion IoU (Jaccard). Both empty -> 1.0. mIoU = mean over images."""
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0
    return np.logical_and(pred, target).sum() / union


def _distance(fn, pred, target):
    pred, target = np.asarray(pred, bool), np.asarray(target, bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return WORST_DISTANCE
    return float(fn(pred, target))


def hd95(pred, target):
    return _distance(binary.hd95, pred, target)


def assd(pred, target):
    return _distance(binary.assd, pred, target)


def all_metrics(pred, target):
    return {"dice": dice(pred, target), "miou": iou(pred, target),
            "hd95": hd95(pred, target), "assd": assd(pred, target)}
