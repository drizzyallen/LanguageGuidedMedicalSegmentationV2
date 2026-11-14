"""Train RecLMIS with the official recipe (configs/reclmis.yaml).

Uses the official code unchanged: Config_covid19.py / Config_MosMedPlus.py (settings), Load_Dataset.ImageToImage2D
with RandomGenerator / ValGenerator (image, mask, CLIP tokenization, train-time blur + photometric distortion),
utils.read_text, nets.RecLMIS (frozen CLIP ViT-B/32 text encoder, conditioned interaction and reconstruction),
utils.WeightedDiceBCE, utils.CosineAnnealingWarmRestarts. The loop follows train_model.py / Train_one_epoch.py:
total loss = sum of loss_weight[k] * loss_k, Adam, scheduler stepped once per epoch after validation, best
checkpoint from epoch 1, early stopping by the config's patience. Only change: the checkpoint is selected by the
common validation Dice (per-image Dice at 224x224, threshold 0.5, mean over the validation set).

The official train_model.py maps "-c Config_MosMedPlus" to Config_covid19 (only "Config_Kvasir_Clinic" loads
Config_MosMedPlus); the dataset's config module is therefore imported directly here.

Data: data/views/lvit/<dataset>/ (LViT layout; symlinks to the common prepared 224x224 data + Task 1.1 text).

Usage (env lgms_reclmis):
  CUDA_VISIBLE_DEVICES=0 python src/phase1/train_reclmis.py --dataset qata --seed 1001 [--quick]
--quick runs the Task 1.3 checks on a small subset and writes to runs/quick_check/reclmis/.
"""
import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common as C  # noqa: E402

RECLMIS = os.path.join(C.EXTERNAL, "RecLMIS")
LOSS_KEYS = ("loss_criterion", "loss_ccl", "loss_text_rec", "loss_img_rec")


def import_official(module_name):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "set CUDA_VISIBLE_DEVICES to the GPU to use"
    sys.path.insert(0, RECLMIS)
    config = importlib.import_module(module_name)
    import clip
    import Load_Dataset
    import utils
    from nets.RecLMIS import RecLMIS
    return config, clip, Load_Dataset, utils, RecLMIS


def check_official_config(logger, cfg, config, dataset):
    logger.info("official settings (external/RecLMIS/{}.py, train_model.py) vs configs/reclmis.yaml:".format(
        config.__name__))
    C.check_matches(logger, config.n_channels, cfg["model"]["n_channels"], "n_channels")
    C.check_matches(logger, config.n_labels, cfg["model"]["n_classes"], "n_labels")
    C.check_matches(logger, config.img_size, cfg["input_size"], "img_size")
    C.check_matches(logger, config.token_len, cfg["text"]["token_len"], "token_len")
    C.check_matches(logger, config.optimizer, cfg["optimizer"]["name"], "optimizer")
    C.check_matches(logger, config.learning_rate, cfg["optimizer"]["lr"], "learning_rate")
    C.check_matches(logger, config.batch_size, cfg["batch_size"][dataset], "batch_size")
    C.check_matches(logger, config.epochs, cfg["max_epochs"][dataset], "epochs")
    C.check_matches(logger, config.early_stopping_patience, cfg["early_stopping_patience"][dataset],
                    "early_stopping_patience")
    C.check_matches(logger, config.loss_weight, cfg["loss"]["loss_weight"], "loss_weight")
    C.check_matches(logger, config.lr, {"qata": "cosineLR", "mosmed": "exp"}[dataset], "lr (scheduler)")
    C.check_matches(logger, config.model_name, "RecLMIS", "model_name")
    vit = config.get_ViT_config()
    C.check_matches(logger, {k: vit[k] for k in cfg["model"]["vit_config"][dataset]},
                    cfg["model"]["vit_config"][dataset], "get_ViT_config()")
    return vit


def pad_text(text):
    """utils.read_text rule, for the pairing check."""
    count = len(text.split())
    return text + " EOF XXX" * (9 - count) if count < 9 else text


def make_dataset(Load_Dataset, utils, config, dataset, split, transform, mode):
    root = C.view_dir("lvit", dataset, split) + "/"
    rowtext = utils.read_text(os.path.join(root, "{}_text.xlsx".format(C.FOLDER[split])))
    return Load_Dataset.ImageToImage2D(root, config.task_name, rowtext, transform, image_size=C.SIZE,
                                       data_name=config.task_name, token_len=config.token_len, config=config, mode=mode)


def item_names(ds):
    # ImageToImage2D indexes mask_list for Covid19 and images_list otherwise (same file names here).
    return ds.mask_list if ds.data_name == "Covid19" else ds.images_list


def subset(ds, ids):
    index = {name[:-4]: i for i, name in enumerate(item_names(ds))}
    return Subset(ds, [index[s] for s in ids])


def check_pairing(logger, clip, ds, dataset, split, token_len, n_full=16):
    """Task 1.3 check 1: image, mask and text pairing through the official loader."""
    texts = C.split_texts(dataset, split)
    names = item_names(ds)
    assert sorted(n[:-4] for n in names) == sorted(texts), "{}/{}: files differ from split".format(dataset, split)
    for name in names:
        assert os.path.isfile(os.path.join(ds.input_path, name)) and os.path.isfile(os.path.join(ds.output_path, name))
        assert ds.rowtext[name] == pad_text(texts[name[:-4]]), name
    rng = np.random.RandomState(0)
    for i in rng.choice(len(ds), n_full, replace=False):
        sample, name = ds[i]
        sid = name[:-4]
        image = np.round(sample["image"].numpy() * 255).astype(np.uint8)
        ref = C.prepared_image(dataset, split, sid)
        assert image.shape == (3, C.SIZE, C.SIZE) and all(np.array_equal(image[c], ref) for c in range(3)), sid
        assert np.array_equal(sample["label"].numpy(), C.prepared_mask(dataset, split, sid)), sid
        expected = pad_text(texts[sid])
        assert sample["text"] == [expected], sid
        tokens = clip.tokenize([expected], context_length=token_len, truncate=True).squeeze()
        assert torch.equal(sample["text_token"], tokens) and torch.equal(sample["text_mask"], (tokens != 0).int()), sid
    logger.info("[check 1] {} {}: all {} samples have image, mask and their own text (official loader keys); "
                "{} random samples fully loaded: image = prepared image (3 channels), mask = prepared mask, "
                "tokens = clip.tokenize of the sample's Task 1.1 text".format(dataset, split, len(names), n_full))


def check_clip(logger, model, clip_path):
    """Task 1.3 check 2: the official CLIP ViT-B/32 weights are loaded into model.clip and frozen."""
    ref = torch.jit.load(clip_path, map_location="cpu").state_dict()
    mine = model.clip.state_dict()
    shared = [k for k in mine if k in ref]
    missing = [k for k in mine if k not in ref]
    same = all(torch.equal(mine[k].detach().cpu().float(), ref[k].float()) for k in shared)
    assert same and not missing, (same, missing[:5])
    frozen = all(not p.requires_grad for p in model.clip.parameters())
    assert frozen
    logger.info("[check 2] CLIP {} ({}): all {} tensors of model.clip equal the official checkpoint, "
                "none missing; all CLIP parameters frozen (requires_grad=False)".format(
                    model.config.clip_backbone, os.path.relpath(os.path.realpath(clip_path), C.ROOT), len(shared)))
    return {k: v.detach().clone() for k, v in mine.items()}


def total_loss(config, criterion, preds, masks, loss_dict):
    """Train_one_epoch.py: loss_dict['loss_criterion'] = criterion(...); total = sum(loss_weight[k] * mean)."""
    loss_dict["loss_criterion"] = criterion(preds, masks.float())
    out = 0
    for k, v in loss_dict.items():
        loss_dict[k] = v.mean()
        out = out + config.loss_weight[k] * v.mean()
    return out


@torch.no_grad()
def evaluate(model, loader, config, criterion, out_dir=None, other_text=False):
    """Validation pass (official: forward with the default mode, eval mode): mean total loss, mean per-image Dice.
    other_text: each batch gets the tokens of the next batch (other samples' texts)."""
    model.eval()
    others = []
    if other_text:
        others = [(s["text_token"], s["text_mask"]) for s, _ in loader]
        others = others[1:] + others[:1]
    losses, scores = [], []
    for i, (sample, names) in enumerate(loader):
        images, masks = sample["image"].cuda(), sample["label"].cuda()
        tok, tmask = sample["text_token"], sample["text_mask"]
        if other_text:
            tok, tmask = others[i] if others[i][0].shape == tok.shape else (torch.roll(tok, 1, 0), torch.roll(tmask, 1, 0))
        preds, loss_dict = model(images, masks, tok.cuda(), tmask.cuda())
        losses.append(total_loss(config, criterion, preds, masks, loss_dict).item())
        for name, p, m in zip(names, preds[:, 0].cpu().numpy(), masks.cpu().numpy()):
            scores.append(C.dice(p >= C.THRESHOLD, m > 0))
            if out_dir is not None:
                C.save_mask(out_dir, name[:-4], p >= C.THRESHOLD)
    model.train()
    return float(np.mean(losses)), float(np.mean(scores))


def make_scheduler(utils, optimizer, cfg, dataset):
    sch = cfg["scheduler"][dataset]
    if dataset == "qata":  # config.lr == 'cosineLR'
        return utils.CosineAnnealingWarmRestarts(optimizer, T_0=sch["T_0"], T_mult=sch["T_mult"], eta_min=sch["eta_min"])
    # config.lr == 'exp' (train_model.py)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: max(0.99 ** epoch, 0.1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timing", action="store_true", help="one full-data epoch (compute estimate), runs/timing/")
    parser.add_argument("--quick", action="store_true", help="Task 1.3 checks on a small subset")
    args = parser.parse_args()

    cfg = C.load_config("reclmis")
    config, clip, Load_Dataset, utils, RecLMIS = import_official(cfg["official_config"][args.dataset])
    C.set_seed(args.seed)
    out = C.run_dir("reclmis", args.dataset, args.seed, args.quick, args.timing)
    logger = C.get_logger(os.path.join(out, "train.log"))
    logger.info("dataset={} seed={} quick={} GPU={} torch={}".format(args.dataset, args.seed, args.quick,
                                                                    os.environ["CUDA_VISIBLE_DEVICES"], torch.__version__))
    logger.info("git commit: {}".format(C.git_commit()))
    logger.info("config: {}".format(cfg))
    config_vit = check_official_config(logger, cfg, config, args.dataset)

    size = [cfg["input_size"], cfg["input_size"]]
    train_set = make_dataset(Load_Dataset, utils, config, args.dataset, "train",
                             Load_Dataset.RandomGenerator(output_size=size), "train")
    val_set = make_dataset(Load_Dataset, utils, config, args.dataset, "val",
                           Load_Dataset.ValGenerator(output_size=size), "val")
    if args.quick:
        for split in ("train", "val", "test"):
            check_pairing(logger, clip, make_dataset(Load_Dataset, utils, config, args.dataset, split,
                                                     Load_Dataset.ValGenerator(output_size=size), "val"),
                          args.dataset, split, config.token_len)
        q = cfg["quick_check"]
        train_set = subset(train_set, C.split_ids(args.dataset, "train")[:q["train_samples"]])
        val_set = subset(val_set, C.split_ids(args.dataset, "val")[:q["val_samples"]])
    else:
        assert len(train_set) == len(C.split_ids(args.dataset, "train"))
        assert len(val_set) == len(C.split_ids(args.dataset, "val"))
    loader_kwargs = dict(batch_size=config.batch_size, num_workers=cfg["num_workers"], pin_memory=True,
                         worker_init_fn=C.make_worker_init_fn(args.seed))
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    logger.info("train {} / val {} samples".format(len(train_set), len(val_set)))

    model = RecLMIS(config, config_vit, n_channels=config.n_channels, n_classes=config.n_labels)
    clip_path = os.path.join(RECLMIS, "nets", "ViT-B-32.pt")
    clip_before = check_clip(logger, model, clip_path) if args.quick else None
    model = model.cuda()
    total, trainable = C.count_params(model)
    logger.info("model RecLMIS: {:.2f}M parameters ({:.2f}M trainable)".format(total / 1e6, trainable / 1e6))
    criterion = utils.WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)
    lr_scheduler = make_scheduler(utils, optimizer, cfg, args.dataset)

    if args.quick:  # checks 3 and 4 on one training batch
        sample, _ = next(iter(train_loader))
        images, masks = sample["image"].cuda(), sample["label"].cuda()
        tok, tmask = sample["text_token"].cuda(), sample["text_mask"].cuda()
        logger.info("[check 3] batch: images {} {}, masks {} values {}, text tokens {}, text mask {}".format(
            tuple(images.shape), images.dtype, tuple(masks.shape), sorted(masks.unique().tolist()),
            tuple(tok.shape), tuple(tmask.shape)))
        model.train()
        preds, loss_dict = model(images, masks, tok, tmask)
        assert preds.shape == (images.shape[0], 1, C.SIZE, C.SIZE) and 0 <= preds.min() and preds.max() <= 1
        loss = total_loss(config, criterion, preds, masks, loss_dict)
        assert torch.isfinite(loss)
        logger.info("[check 3] forward pass ok: output {} (sigmoid, range [{:.3f}, {:.3f}]), total loss {:.4f}".format(
            tuple(preds.shape), preds.min().item(), preds.max().item(), loss.item()))
        assert model.aux and sorted(loss_dict) == sorted(LOSS_KEYS), (model.aux, sorted(loss_dict))
        assert all(torch.isfinite(v) and v.item() > 0 for v in loss_dict.values()), loss_dict
        logger.info("[check 4] conditioned reconstruction active (model.aux = True); losses: " +
                    ", ".join("{} {:.4f} (x{})".format(k, loss_dict[k].item(), config.loss_weight[k]) for k in LOSS_KEYS))
        optimizer.zero_grad()
        loss.backward()
        parts = {"text_module4 (CLIP text -> 512)": model.text_module4,
                 "interact (conditioned interaction / fusion)": model.interact,
                 "mlp_woi (text token weights)": model.mlp_woi, "mlp_visual (image token weights)": model.mlp_visual,
                 "rec_text_trans1 (conditioned language reconstruction)": model.rec_text_trans1,
                 "rec_img_trans1 (conditioned vision reconstruction)": model.rec_img_trans1}
        norms = {k: C.grad_norm(m) for k, m in parts.items()}
        assert all(v > 0 for v in norms.values()), norms
        logger.info("[check 4] non-zero gradients from the total loss in " +
                    ", ".join("{} {:.2e}".format(k, v) for k, v in norms.items()))
        optimizer.zero_grad()

    fields = ["epoch", "lr", "train_loss"] + ["train_" + k for k in LOSS_KEYS] + ["val_loss", "val_dice", "aux",
                                                                                  "seconds"]
    history = C.History(os.path.join(out, "history.csv"), fields)
    epochs = cfg["quick_check"]["epochs"] if args.quick else (1 if args.timing else config.epochs)
    patience = config.early_stopping_patience
    max_dice, best_epoch = 0.0, 1
    ckpt_path = os.path.join(out, "checkpoint_best.pth")
    for epoch in range(epochs):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        model.train()
        sums = {k: [] for k in ("total",) + LOSS_KEYS}
        for sample, _ in train_loader:
            images, masks = sample["image"].cuda(), sample["label"].cuda()
            preds, loss_dict = model(images, masks, sample["text_token"].cuda(), sample["text_mask"].cuda())
            loss = total_loss(config, criterion, preds, masks, loss_dict)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sums["total"].append(loss.item())
            for k in LOSS_KEYS:
                sums[k].append(loss_dict[k].item() if k in loss_dict else float("nan"))
        val_loss, val_dice = evaluate(model, val_loader, config, criterion)
        lr_scheduler.step()  # official: once per epoch, at the end of the validation pass

        seconds = time.time() - start
        row = {"epoch": epoch + 1, "lr": lr, "train_loss": np.mean(sums["total"]), "val_loss": val_loss,
               "val_dice": val_dice, "aux": model.aux, "seconds": round(seconds, 1)}
        row.update({"train_" + k: np.mean(sums[k]) for k in LOSS_KEYS})
        history.add(**row)
        logger.info("epoch {}/{} lr {:.2e} train_loss {:.4f} ({}) val_loss {:.4f} val_dice {:.4f} aux {} ({:.0f}s)".format(
            epoch + 1, epochs, lr, row["train_loss"],
            " ".join("{} {:.4f}".format(k[5:], row["train_" + k]) for k in LOSS_KEYS), val_loss, val_dice,
            model.aux, seconds))
        if not model.aux:
            logger.info("  WARNING: model.aux is False - the official forward disabled the reconstruction branch "
                        "after an exception")
        if val_dice > max_dice:  # official: saved from epoch 1 (epoch + 1 > 0)
            logger.info("  saving best model, val dice {:.4f} -> {:.4f}".format(max_dice, val_dice))
            max_dice, best_epoch = val_dice, epoch + 1
            torch.save({"epoch": epoch + 1, "val_dice": val_dice, "aux": model.aux, "state_dict": model.state_dict()},
                       ckpt_path)
        if epoch - best_epoch + 1 > patience:
            logger.info("early stopping at epoch {}".format(epoch + 1))
            break
    logger.info("done: best val dice {:.4f} at epoch {}; reconstruction branch active at end: {}".format(
        max_dice, best_epoch, model.aux))

    if args.quick:  # checks 5-6
        import pandas as pd
        h = pd.read_csv(os.path.join(out, "history.csv"))
        logger.info("[check 5] training loss: epoch 1 {:.4f} -> epoch {} {:.4f} (min {:.4f}); ".format(
            h.train_loss.iloc[0], len(h), h.train_loss.iloc[-1], h.train_loss.min()) +
            ", ".join("{} {:.4f} -> {:.4f}".format(k, h["train_" + k].iloc[0], h["train_" + k].iloc[-1]) for k in LOSS_KEYS))
        logger.info("[check 5] validation Dice: epoch 1 {:.4f} -> best {:.4f} (epoch {}); no NaN losses: {}; "
                    "aux active every epoch: {}".format(h.val_dice.iloc[0], h.val_dice.max(), int(h.val_dice.idxmax()) + 1,
                                                        not h[["train_loss", "val_loss"]].isna().any().any(),
                                                        bool(h.aux.all())))
        unchanged = all(torch.equal(v.cpu(), clip_before[k].cpu()) for k, v in model.clip.state_dict().items())
        logger.info("[check 2] CLIP weights unchanged after training (frozen): {}".format(unchanged))
        assert unchanged
        model.load_state_dict(torch.load(ckpt_path, map_location="cuda")["state_dict"])
        pred_dir = os.path.join(out, "val_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        _, own = evaluate(model, val_loader, config, criterion, out_dir=pred_dir)
        ids = C.split_ids(args.dataset, "val")[:cfg["quick_check"]["val_samples"]]
        logger.info("[check 6] validation predictions of the best checkpoint (epoch {}):".format(best_epoch))
        C.check_saved_predictions(logger, pred_dir, ids, args.dataset, "val", own)
        _, other = evaluate(model, val_loader, config, criterion, other_text=True)
        logger.info("[check 4] text changes the prediction (best checkpoint, validation subset): Dice with own "
                    "text {:.4f}, with other samples' texts {:.4f}".format(own, other))
        logger.info("QUICK CHECK DONE")


if __name__ == "__main__":
    main()
