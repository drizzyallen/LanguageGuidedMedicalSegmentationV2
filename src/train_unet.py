"""Train the LViT U-Net with the official LViT recipe (configs/unet.yaml).

Follows external/LViT/train_model.py: Adam + CosineAnnealingWarmRestarts (stepped once per epoch after
validation), WeightedDiceBCE(0.5, 0.5), LViT RandomGenerator / ValGenerator, best checkpoint saved only
after epoch 5, early stopping after 50 epochs without improvement. Checkpoint selection uses validation Dice.

Usage: python src/train_unet.py --dataset qata --seed 1001 [--quick]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream  # noqa: E402
from data.dataset import PreparedDataset  # noqa: E402
from training_common import (History, get_logger, load_config, make_worker_init_fn, quick_sanity_checks,  # noqa: E402
                             run_dir, set_seed, validation_dice)

QUICK = {"train_limit": 64, "val_limit": 32, "epochs": 10}


class LViTTransform:
    """Feed the prepared image/mask to LViT's own RandomGenerator / ValGenerator as LViT's loader does."""

    def __init__(self, generator):
        self.generator = generator
        self.correct_dims = upstream.lvit_load_dataset().correct_dims

    def __call__(self, image, mask):
        image = np.repeat(image[..., None], 3, axis=2)  # cv2.imread() 3-channel image of a grayscale PNG
        mask = self.correct_dims(mask)
        sample = self.generator({"image": image, "label": mask, "text": np.zeros((1, 1), np.float32)})
        return sample["image"], sample["label"]  # (3, H, W) float in [0, 1], (H, W) long


def lesion_prob(model, images):
    return model(images)[:, 0]  # sigmoid output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quick", action="store_true", help="Task 0.3 quick check on a small subset")
    args = parser.parse_args()

    cfg = load_config("unet")
    set_seed(args.seed, cfg["deterministic"])
    out = run_dir("unet", args.dataset, args.seed, args.quick)
    logger = get_logger(os.path.join(out, "train.log"))
    logger.info(f"config: {cfg}")
    logger.info(f"dataset={args.dataset} seed={args.seed} quick={args.quick}")

    lvit_ds, lvit_utils = upstream.lvit_load_dataset(), upstream.lvit_utils()
    size = [cfg["input_size"], cfg["input_size"]]
    train_set = PreparedDataset(args.dataset, "train", LViTTransform(lvit_ds.RandomGenerator(output_size=size)),
                                limit=QUICK["train_limit"] if args.quick else None)
    val_set = PreparedDataset(args.dataset, "val", LViTTransform(lvit_ds.ValGenerator(output_size=size)),
                              limit=QUICK["val_limit"] if args.quick else None)
    loader_kwargs = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True,
                         worker_init_fn=make_worker_init_fn(args.seed))
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    logger.info(f"train {len(train_set)} / val {len(val_set)} samples")

    model = upstream.lvit_unet().UNet(n_channels=cfg["model"]["n_channels"],
                                      n_classes=cfg["model"]["n_classes"]).cuda()
    criterion = lvit_utils.WeightedDiceBCE(dice_weight=cfg["loss"]["dice_weight"], BCE_weight=cfg["loss"]["bce_weight"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg["optimizer"]["lr"])
    sch = cfg["scheduler"]
    lr_scheduler = lvit_utils.CosineAnnealingWarmRestarts(optimizer, T_0=sch["T_0"], T_mult=sch["T_mult"],
                                                          eta_min=sch["eta_min"])
    loss_fn = lambda preds, masks: criterion(preds, masks.float())  # noqa: E731

    if args.quick:
        quick_sanity_checks(logger, model, next(iter(train_loader)), lambda m, x: m(x), loss_fn,
                            expected_out_channels=cfg["model"]["n_classes"])

    history = History(os.path.join(out, "history.csv"),
                      ["epoch", "lr", "train_loss", "val_loss", "val_dice", "seconds"])
    epochs = QUICK["epochs"] if args.quick else cfg["max_epochs"]
    max_dice, best_epoch = 0.0, 1
    for epoch in range(epochs):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        train_losses = []
        for images, masks, _ in train_loader:
            images, masks = images.cuda(), masks.cuda()
            loss = loss_fn(model(images), masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            val_losses = [loss_fn(model(images.cuda()), masks.cuda()).item() for images, masks, _ in val_loader]
        val_dice = validation_dice(model, val_loader, lesion_prob)
        lr_scheduler.step()  # LViT steps the scheduler once per epoch, at the end of validation

        history.add(epoch=epoch + 1, lr=lr, train_loss=np.mean(train_losses), val_loss=np.mean(val_losses),
                    val_dice=val_dice, seconds=round(time.time() - start, 1))
        logger.info(f"epoch {epoch + 1}/{epochs} lr {lr:.2e} train_loss {np.mean(train_losses):.4f} "
                    f"val_loss {np.mean(val_losses):.4f} val_dice {val_dice:.4f} ({time.time() - start:.0f}s)")

        if val_dice > max_dice:
            if epoch + 1 > cfg["min_epoch_to_save"] or args.quick:
                logger.info(f"  saving best model, val dice {max_dice:.4f} -> {val_dice:.4f}")
                max_dice, best_epoch = val_dice, epoch + 1
                torch.save({"epoch": epoch + 1, "val_dice": val_dice, "state_dict": model.state_dict()},
                           os.path.join(out, "checkpoint_best.pth"))
        early_stopping_count = epoch - best_epoch + 1
        if early_stopping_count > cfg["early_stopping_patience"]:
            logger.info(f"early stopping at epoch {epoch + 1}")
            break

    logger.info(f"done: best val dice {max_dice:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
