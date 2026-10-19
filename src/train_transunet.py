"""Train TransUNet (R50-ViT-B_16) with the official recipe (configs/transunet.yaml).

Follows external/TransUNet/train.py + trainer.py: ImageNet-21k weights via load_from, SGD (momentum 0.9,
weight decay 1e-4), poly LR decay per iteration, 0.5 * CE + 0.5 * Dice, Synapse RandomGenerator augmentation,
150 epochs. The official trainer has no validation; here validation Dice is computed after every epoch and the
best checkpoint is kept (Phase 0 rule: checkpoint selection by validation Dice).

Usage: python src/train_transunet.py --dataset qata --seed 1001 [--quick]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream  # noqa: E402
from data.dataset import PreparedDataset  # noqa: E402
from training_common import (History, get_logger, load_config, make_worker_init_fn, quick_sanity_checks,  # noqa: E402
                             run_dir, set_seed, validation_dice)

QUICK = {"train_limit": 64, "val_limit": 32, "epochs": 10}


class TrainTransform:
    """TransUNet's Synapse RandomGenerator on a float image in [0, 1] (Synapse slices are stored in [0, 1])."""

    def __init__(self, size):
        self.generator = upstream.transunet_dataset().RandomGenerator(output_size=[size, size])

    def __call__(self, image, mask):
        sample = self.generator({"image": image.astype(np.float32) / 255.0, "label": mask.astype(np.float32)})
        return sample["image"], sample["label"]  # (1, H, W) float, (H, W) long


def val_transform(image, mask):
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0), torch.from_numpy(mask.astype(np.int64))


def lesion_prob(model, images):
    return torch.softmax(model(images), dim=1)[:, 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quick", action="store_true", help="Task 0.3 quick check on a small subset")
    args = parser.parse_args()

    cfg = load_config("transunet")
    set_seed(args.seed, cfg["deterministic"])
    out = run_dir("transunet", args.dataset, args.seed, args.quick)
    logger = get_logger(os.path.join(out, "train.log"))
    logger.info(f"config: {cfg}")
    logger.info(f"dataset={args.dataset} seed={args.seed} quick={args.quick}")

    train_set = PreparedDataset(args.dataset, "train", TrainTransform(cfg["input_size"]),
                                limit=QUICK["train_limit"] if args.quick else None)
    val_set = PreparedDataset(args.dataset, "val", val_transform, limit=QUICK["val_limit"] if args.quick else None)
    loader_kwargs = dict(num_workers=cfg["num_workers"], pin_memory=True, worker_init_fn=make_worker_init_fn(args.seed))
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=False, **loader_kwargs)
    logger.info(f"train {len(train_set)} / val {len(val_set)} samples")

    modeling = upstream.transunet_modeling()
    mcfg = cfg["model"]
    config_vit = modeling.CONFIGS[mcfg["vit_name"]]
    config_vit.n_classes = mcfg["n_classes"]
    config_vit.n_skip = mcfg["n_skip"]
    grid = cfg["input_size"] // mcfg["vit_patches_size"]
    config_vit.patches.grid = (grid, grid)
    config_vit.pretrained_path = os.path.join(upstream.ROOT, cfg["pretraining"]["weights"])
    model = modeling.VisionTransformer(config_vit, img_size=cfg["input_size"], num_classes=config_vit.n_classes).cuda()
    model.load_from(weights=np.load(config_vit.pretrained_path))
    logger.info(f"loaded pretrained weights {config_vit.pretrained_path}")

    ce_loss = CrossEntropyLoss()
    dice_loss = upstream.transunet_utils().DiceLoss(mcfg["n_classes"])
    lw = cfg["loss"]

    def loss_fn(outputs, labels):
        return lw["ce_weight"] * ce_loss(outputs, labels[:].long()) + \
            lw["dice_weight"] * dice_loss(outputs, labels, softmax=True)

    opt = cfg["optimizer"]
    base_lr = opt["lr"]
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=opt["momentum"],
                                weight_decay=opt["weight_decay"])

    if args.quick:
        quick_sanity_checks(logger, model, next(iter(train_loader)), lambda m, x: m(x), loss_fn,
                            expected_out_channels=mcfg["n_classes"])

    history = History(os.path.join(out, "history.csv"), ["epoch", "lr", "train_loss", "val_dice", "seconds"])
    max_epochs = QUICK["epochs"] if args.quick else cfg["max_epochs"]
    max_iterations = max_epochs * len(train_loader)
    logger.info(f"{len(train_loader)} iterations per epoch, {max_iterations} max iterations")
    iter_num, best_dice, best_epoch = 0, -1.0, 0
    model.train()
    for epoch in range(max_epochs):
        start = time.time()
        losses = []
        for images, labels, _ in train_loader:
            images, labels = images.cuda(), labels.cuda()
            loss = loss_fn(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** cfg["scheduler"]["power"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_
            iter_num += 1
            losses.append(loss.item())

        val_dice = validation_dice(model, val_loader, lesion_prob)
        history.add(epoch=epoch + 1, lr=lr_, train_loss=np.mean(losses), val_dice=val_dice,
                    seconds=round(time.time() - start, 1))
        logger.info(f"epoch {epoch + 1}/{max_epochs} lr {lr_:.2e} train_loss {np.mean(losses):.4f} "
                    f"val_dice {val_dice:.4f} ({time.time() - start:.0f}s)")
        if val_dice > best_dice:
            logger.info(f"  saving best model, val dice {max(best_dice, 0):.4f} -> {val_dice:.4f}")
            best_dice, best_epoch = val_dice, epoch + 1
            torch.save({"epoch": epoch + 1, "val_dice": val_dice, "state_dict": model.state_dict()},
                       os.path.join(out, "checkpoint_best.pth"))

    logger.info(f"done: best val dice {best_dice:.4f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
