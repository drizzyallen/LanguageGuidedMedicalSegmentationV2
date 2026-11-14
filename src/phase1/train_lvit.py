"""Train LViT-T with the official LViT recipe (configs/lvit.yaml).

Uses the official code unchanged: Config.py (model and training settings), Load_Dataset.ImageToImage2D with
RandomGenerator / ValGenerator (image, mask and BertEmbedding text pipeline), utils.read_text, nets.LViT,
utils.WeightedDiceBCE and utils.CosineAnnealingWarmRestarts. The loop follows train_model.py /
Train_one_epoch.py: Adam, scheduler stepped once per epoch after validation, best checkpoint saved only after
epoch 5, early stopping after 50 epochs without improvement. Only change: the checkpoint is selected by the
common validation Dice (per-image Dice at 224x224, threshold 0.5, mean over the validation set).

Data: data/views/lvit/<dataset>/ (symlinks to the common prepared 224x224 data + Task 1.1 text).

Usage (env lgms_lvit):
  CUDA_VISIBLE_DEVICES=0 python src/phase1/train_lvit.py --dataset qata --seed 1001 [--quick]
--quick runs the Task 1.2 checks on a small subset and writes to runs/quick_check/lvit/.
--test (Task 1.6, after training) reloads checkpoint_best.pth, re-checks its validation Dice and saves binary test
predictions to runs/lvit/<dataset>/seed_<seed>/test/predictions/ (official ValGenerator input, threshold 0.5).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common as C  # noqa: E402

LVIT = os.path.join(C.EXTERNAL, "LViT")
TASK_NAME = {"qata": "Covid19", "mosmed": "MosMedDataPlus"}  # stored by ImageToImage2D, not used by it


def import_official():
    """Import the official LViT modules. Config.py sets CUDA_VISIBLE_DEVICES='0' and calls
    torch.cuda.is_available() on import, which would bind the process to GPU 0. CUDA is therefore initialised
    on our GPU first (the device list is fixed at initialisation), and the variable is restored afterwards."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible is not None, "set CUDA_VISIBLE_DEVICES to the GPU to use"
    torch.cuda.init()
    os.environ.setdefault("MXNET_HOME", os.path.join(C.PRETRAINED, "mxnet"))  # BertEmbedding weights
    sys.path.insert(0, LVIT)
    import Config
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    import Load_Dataset
    import utils
    from nets.LViT import LViT
    return Config, Load_Dataset, utils, LViT


def check_official_config(logger, cfg, Config, dataset):
    logger.info("official settings (external/LViT/Config.py, train_model.py) vs configs/lvit.yaml:")
    C.check_matches(logger, Config.model_name, cfg["model"]["variant"], "model_name")
    C.check_matches(logger, Config.n_channels, cfg["model"]["n_channels"], "n_channels")
    C.check_matches(logger, Config.n_labels, cfg["model"]["n_classes"], "n_labels")
    C.check_matches(logger, Config.img_size, cfg["input_size"], "img_size")
    C.check_matches(logger, Config.batch_size, cfg["batch_size"], "batch_size")
    C.check_matches(logger, Config.epochs, cfg["max_epochs"], "epochs")
    C.check_matches(logger, Config.early_stopping_patience, cfg["early_stopping_patience"], "early_stopping_patience")
    C.check_matches(logger, Config.cosineLR, True, "cosineLR")
    ct = Config.get_CTranS_config()
    official = {"KV_size": ct.KV_size, "num_heads": ct.transformer.num_heads, "num_layers": ct.transformer.num_layers,
                "expand_ratio": ct.expand_ratio, "embeddings_dropout_rate": ct.transformer.embeddings_dropout_rate,
                "attention_dropout_rate": ct.transformer.attention_dropout_rate,
                "dropout_rate": ct.transformer.dropout_rate, "patch_sizes": list(ct.patch_sizes),
                "base_channel": ct.base_channel}
    C.check_matches(logger, official, cfg["model"]["ctrans"], "get_CTranS_config()")
    # Config.py: "learning_rate = 1e-3  # MoNuSeg: 1e-3, Covid19: 3e-4" -> value set per dataset in the yaml.
    logger.info("  learning rate for {} = {} (configs/lvit.yaml)".format(dataset, cfg["optimizer"]["lr"][dataset]))
    return ct


def pad_text(text):
    """utils.read_text rule, for the pairing check."""
    count = len(text.split())
    return text + " EOF XXX" * (9 - count) if count < 9 else text


def make_dataset(Load_Dataset, utils, dataset, split, transform):
    root = C.view_dir("lvit", dataset, split) + "/"
    rowtext = utils.read_text(os.path.join(root, "{}_text.xlsx".format(C.FOLDER[split])))
    return Load_Dataset.ImageToImage2D(root, TASK_NAME[dataset], rowtext, transform, image_size=C.SIZE)


def subset(ds, ids):
    index = {name[:-4]: i for i, name in enumerate(ds.images_list)}
    return Subset(ds, [index[s] for s in ids])


def check_pairing(logger, ds, dataset, split, n_full=16):
    """Task 1.2 check 1: image, mask and text pairing through the official loader."""
    texts = C.split_texts(dataset, split)
    names = ds.images_list
    assert sorted(n[:-4] for n in names) == sorted(texts), "{}/{}: images differ from split".format(dataset, split)
    for name in names:  # official lookup: mask = image name with .png, text = rowtext[mask name]
        mask_name = name[:-3] + "png"
        assert os.path.isfile(os.path.join(ds.output_path, mask_name)), mask_name
        assert ds.rowtext[mask_name] == pad_text(texts[name[:-4]]), name
    rng = np.random.RandomState(0)
    for i in rng.choice(len(ds), n_full, replace=False):
        sample, name = ds[i]
        sid = name[:-4]
        image = np.round(sample["image"].numpy() * 255).astype(np.uint8)
        ref = C.prepared_image(dataset, split, sid)
        assert image.shape == (3, C.SIZE, C.SIZE) and all(np.array_equal(image[c], ref) for c in range(3)), sid
        assert np.array_equal(sample["label"].numpy(), C.prepared_mask(dataset, split, sid)), sid
        expected = np.array(ds.bert_embedding(pad_text(texts[sid]).split("\n"))[0][1])[:10]
        assert sample["text"].shape == (10, 768) and np.allclose(sample["text"].numpy(), expected), sid
    logger.info("[check 1] {} {}: all {} images have their mask and their own text (official loader keys); "
                "{} random samples fully loaded: image = prepared image (3 channels), mask = prepared mask, "
                "text = BertEmbedding of the sample's Task 1.1 text (10 x 768)".format(dataset, split, len(names),
                                                                                       n_full))


def lesion_prob(model, images, text):
    return model(images, text)[:, 0]  # sigmoid output


@torch.no_grad()
def evaluate(model, loader, criterion, out_dir=None, text_mode="own"):
    """Validation pass: mean official loss and mean per-image Dice. Optionally saves binary predictions.
    text_mode: 'own' (the sample's text), 'zero' (all-zero text), 'other' (each batch gets the text of the next
    batch, i.e. other samples' texts)."""
    model.eval()
    others = []
    if text_mode == "other":
        others = [sample["text"] for sample, _ in loader]
        others = others[1:] + others[:1]
    losses, scores = [], []
    for i, (sample, names) in enumerate(loader):
        images, masks, text = sample["image"].cuda(), sample["label"].cuda(), sample["text"].cuda()
        used = text
        if text_mode == "zero":
            used = torch.zeros_like(text)
        elif text_mode == "other":
            used = others[i].cuda() if others[i].shape == text.shape else torch.roll(text, 1, 0)
        preds = model(images, used)
        losses.append(criterion(preds, masks.float()).item())
        for name, p, m in zip(names, preds[:, 0].cpu().numpy(), masks.cpu().numpy()):
            scores.append(C.dice(p >= C.THRESHOLD, m > 0))
            if out_dir is not None:
                C.save_mask(out_dir, name[:-4], p >= C.THRESHOLD)
    model.train()
    return float(np.mean(losses)), float(np.mean(scores))


def run_test(logger, cfg, args, out, config_vit, Load_Dataset, utils, LViT):
    """Task 1.6: best checkpoint -> validation Dice re-check -> binary test predictions (no test metric here)."""
    size = [cfg["input_size"], cfg["input_size"]]
    loader_kwargs = dict(batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"], pin_memory=True)
    model = LViT(config_vit, n_channels=cfg["model"]["n_channels"], n_classes=cfg["model"]["n_classes"]).cuda()
    ckpt_path = os.path.join(out, "checkpoint_best.pth")
    ckpt = torch.load(ckpt_path, map_location="cuda")
    model.load_state_dict(ckpt["state_dict"])
    criterion = utils.WeightedDiceBCE(dice_weight=cfg["loss"]["dice_weight"], BCE_weight=cfg["loss"]["bce_weight"])
    val_set = make_dataset(Load_Dataset, utils, args.dataset, "val", Load_Dataset.ValGenerator(output_size=size))
    if args.quick:  # dry run on the quick-check checkpoint: its validation subset
        val_set = subset(val_set, C.split_ids(args.dataset, "val")[:cfg["quick_check"]["val_samples"]])
    _, val_dice = evaluate(model, DataLoader(val_set, **loader_kwargs), criterion)
    C.check_val_dice(logger, val_dice, ckpt["val_dice"], tol=1e-6)
    test_set = make_dataset(Load_Dataset, utils, args.dataset, "test", Load_Dataset.ValGenerator(output_size=size))
    assert len(test_set) == len(C.split_ids(args.dataset, "test"))
    pred_dir = C.test_pred_dir(out)
    evaluate(model, DataLoader(test_set, **loader_kwargs), criterion, out_dir=pred_dir)  # its Dice is not used
    C.finish_test(logger, out, "lvit", args.dataset, args.seed, pred_dir, ckpt_path, ckpt["epoch"], ckpt["val_dice"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timing", action="store_true", help="one full-data epoch (compute estimate), runs/timing/")
    parser.add_argument("--quick", action="store_true", help="Task 1.2 checks on a small subset")
    parser.add_argument("--test", action="store_true", help="Task 1.6: test predictions of the best checkpoint")
    args = parser.parse_args()

    Config, Load_Dataset, utils, LViT = import_official()
    cfg = C.load_config("lvit")
    C.set_seed(args.seed)
    out = C.run_dir("lvit", args.dataset, args.seed, args.quick, args.timing)
    logger = C.get_logger(os.path.join(out, "test.log" if args.test else "train.log"))
    logger.info("dataset={} seed={} quick={} GPU={} torch={}".format(args.dataset, args.seed, args.quick,
                                                                    os.environ["CUDA_VISIBLE_DEVICES"], torch.__version__))
    logger.info("git commit: {}".format(C.git_commit()))
    C.check_gpu(logger)
    logger.info("config: {}".format(cfg))
    config_vit = check_official_config(logger, cfg, Config, args.dataset)
    if args.test:
        return run_test(logger, cfg, args, out, config_vit, Load_Dataset, utils, LViT)

    size = [cfg["input_size"], cfg["input_size"]]
    train_set = make_dataset(Load_Dataset, utils, args.dataset, "train", Load_Dataset.RandomGenerator(output_size=size))
    val_set = make_dataset(Load_Dataset, utils, args.dataset, "val", Load_Dataset.ValGenerator(output_size=size))
    if args.quick:
        for split in ("train", "val", "test"):
            check_pairing(logger, make_dataset(Load_Dataset, utils, args.dataset, split,
                                               Load_Dataset.ValGenerator(output_size=size)), args.dataset, split)
        q = cfg["quick_check"]
        train_set = subset(train_set, C.split_ids(args.dataset, "train")[:q["train_samples"]])
        val_set = subset(val_set, C.split_ids(args.dataset, "val")[:q["val_samples"]])
    else:
        assert len(train_set) == len(C.split_ids(args.dataset, "train"))
        assert len(val_set) == len(C.split_ids(args.dataset, "val"))
    loader_kwargs = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True,
                         worker_init_fn=C.make_worker_init_fn(args.seed))
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    logger.info("train {} / val {} samples".format(len(train_set), len(val_set)))

    model = LViT(config_vit, n_channels=cfg["model"]["n_channels"], n_classes=cfg["model"]["n_classes"]).cuda()
    total, trainable = C.count_params(model)
    logger.info("model LViT (LViT-T config): {:.2f}M parameters ({:.2f}M trainable)".format(total / 1e6, trainable / 1e6))
    criterion = utils.WeightedDiceBCE(dice_weight=cfg["loss"]["dice_weight"], BCE_weight=cfg["loss"]["bce_weight"])
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg["optimizer"]["lr"][args.dataset])
    sch = cfg["scheduler"]
    lr_scheduler = utils.CosineAnnealingWarmRestarts(optimizer, T_0=sch["T_0"], T_mult=sch["T_mult"],
                                                     eta_min=sch["eta_min"])

    if args.quick:  # checks 2 and 3 on one training batch
        sample, _ = next(iter(train_loader))
        images, masks, text = sample["image"].cuda(), sample["label"].cuda(), sample["text"].cuda()
        logger.info("[check 2] batch: images {} {}, masks {} values {}, text {}".format(
            tuple(images.shape), images.dtype, tuple(masks.shape), sorted(masks.unique().tolist()), tuple(text.shape)))
        model.train()
        preds = model(images, text)
        assert preds.shape == (images.shape[0], 1, C.SIZE, C.SIZE) and 0 <= preds.min() and preds.max() <= 1
        loss = criterion(preds, masks.float())
        assert torch.isfinite(loss)
        logger.info("[check 2] forward pass ok: output {} (sigmoid, range [{:.3f}, {:.3f}]), loss {:.4f}".format(
            tuple(preds.shape), preds.min().item(), preds.max().item(), loss.item()))
        optimizer.zero_grad()
        loss.backward()
        branch = {"text_module4": model.text_module4, "text_module3": model.text_module3,
                  "text_module2": model.text_module2, "text_module1": model.text_module1,
                  "downVit.CTBN3 (text -> image tokens)": model.downVit.CTBN3}
        norms = {k: C.grad_norm(m) for k, m in branch.items()}
        assert all(v > 0 for v in norms.values()), norms
        logger.info("[check 3] text reaches the language branch: non-zero gradient from the segmentation loss in "
                    + ", ".join("{} {:.2e}".format(k, v) for k, v in norms.items()))
        optimizer.zero_grad()

    history = C.History(os.path.join(out, "history.csv"),
                        ["epoch", "lr", "train_loss", "val_loss", "val_dice", "seconds"])
    epochs = cfg["quick_check"]["epochs"] if args.quick else (1 if args.timing else cfg["max_epochs"])
    max_dice, best_epoch = 0.0, 1
    ckpt_path = os.path.join(out, "checkpoint_best.pth")
    for epoch in range(epochs):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        train_losses = []
        for sample, _ in train_loader:
            images, masks, text = sample["image"].cuda(), sample["label"].cuda(), sample["text"].cuda()
            loss = criterion(model(images, text), masks.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_loss, val_dice = evaluate(model, val_loader, criterion)
        lr_scheduler.step()  # official: once per epoch, at the end of the validation pass

        seconds = time.time() - start
        history.add(epoch=epoch + 1, lr=lr, train_loss=np.mean(train_losses), val_loss=val_loss, val_dice=val_dice,
                    seconds=round(seconds, 1))
        logger.info("epoch {}/{} lr {:.2e} train_loss {:.4f} val_loss {:.4f} val_dice {:.4f} ({:.0f}s)".format(
            epoch + 1, epochs, lr, np.mean(train_losses), val_loss, val_dice, seconds))
        if val_dice > max_dice:
            if epoch + 1 > cfg["min_epoch_to_save"] or args.quick:
                logger.info("  saving best model, val dice {:.4f} -> {:.4f}".format(max_dice, val_dice))
                max_dice, best_epoch = val_dice, epoch + 1
                torch.save({"epoch": epoch + 1, "val_dice": val_dice, "state_dict": model.state_dict()}, ckpt_path)
        if epoch - best_epoch + 1 > cfg["early_stopping_patience"]:
            logger.info("early stopping at epoch {}".format(epoch + 1))
            break
    logger.info("done: best val dice {:.4f} at epoch {}".format(max_dice, best_epoch))

    if args.quick:  # checks 4-6
        import pandas as pd
        h = pd.read_csv(os.path.join(out, "history.csv"))
        logger.info("[check 4] training loss: epoch 1 {:.4f} -> epoch {} {:.4f} (min {:.4f})".format(
            h.train_loss.iloc[0], len(h), h.train_loss.iloc[-1], h.train_loss.min()))
        logger.info("[check 5] validation Dice: epoch 1 {:.4f} -> best {:.4f} (epoch {})".format(
            h.val_dice.iloc[0], h.val_dice.max(), int(h.val_dice.idxmax()) + 1))
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda")["state_dict"])
        pred_dir = os.path.join(out, "val_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        _, own = evaluate(model, val_loader, criterion, out_dir=pred_dir)
        ids = C.split_ids(args.dataset, "val")[:cfg["quick_check"]["val_samples"]]
        logger.info("[check 6] validation predictions of the best checkpoint (epoch {}):".format(best_epoch))
        C.check_saved_predictions(logger, pred_dir, ids, args.dataset, "val", own)
        _, zero = evaluate(model, val_loader, criterion, text_mode="zero")
        _, other = evaluate(model, val_loader, criterion, text_mode="other")
        logger.info("[check 3] text changes the prediction (best checkpoint, validation subset): Dice with own "
                    "text {:.4f}, with all-zero text {:.4f}, with other samples' texts {:.4f}".format(
                        own, zero, other))
        logger.info("QUICK CHECK DONE")


if __name__ == "__main__":
    main()
