"""Train DD-CMD with the official recipe (configs/ddcmd.yaml; starting point: official config/train.yaml).

Uses the official code unchanged: utils/dataset.py SegData (MONAI loading, augmentation, BiomedBERT tokenization),
net/creratemodel.py CreateModel (TextGuidedSegmentationModel, DiceCELoss, AdamW, CosineAnnealingLR) and the
Lightning trainer set up as in train.py (min/max epochs, EarlyStopping, ModelCheckpoint, single device).
Only change: ModelCheckpoint and EarlyStopping monitor the common validation Dice (per-image Dice at 224x224,
threshold 0.5, mean over the validation set) instead of the official val_MIoU. It is computed from the same
validation outputs (sigmoid probabilities) in a subclass of CreateModel and logged as val_common_dice.

The official loader (MONAI PILReader) transposes images and masks (H <-> W) consistently; Dice is unaffected.
Saved predictions are transposed back to the original orientation.

Data: data/views/ddcmd/<dataset>/ (symlinks to the common prepared 224x224 data + Task 1.1 text as CSV prompts).

Usage (env lgms_ddcmd):
  CUDA_VISIBLE_DEVICES=0 python src/phase1/train_ddcmd.py --dataset qata --seed 1001 [--quick]
--quick runs the Task 1.4 checks on a small subset and writes to runs/quick_check/ddcmd/.
"""
import argparse
import csv
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import common as C  # noqa: E402

DDCMD = os.path.join(C.EXTERNAL, "DD-CMD")
HF = os.path.join(C.PRETRAINED, "huggingface")


def import_official():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") is not None, "set CUDA_VISIBLE_DEVICES to the GPU to use"
    sys.path.insert(0, DDCMD)
    import pytorch_lightning as pl
    import torch.multiprocessing
    import utils.config as ddcmd_config
    from net.creratemodel import CreateModel
    from utils.dataset import SegData
    torch.multiprocessing.set_sharing_strategy("file_system")  # as in train.py
    return pl, ddcmd_config, CreateModel, SegData


def set_seed_official(seed):
    """train.py set_seed (no cudnn flags; Lightning deterministic=False by default)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(logger, cfg, ddcmd_config, dataset):
    """Official config/train.yaml + creratemodel.py/train.py constants, checked against configs/ddcmd.yaml;
    only data paths and the local pinned backbone copies are replaced."""
    args = ddcmd_config.load_cfg_from_cfg_file(os.path.join(DDCMD, "config", "train.yaml"))
    logger.info("official settings (external/DD-CMD/config/train.yaml, net/creratemodel.py) vs configs/ddcmd.yaml:")
    for key, frozen in (("vision_type", cfg["model"]["vision_type"]), ("bert_type", cfg["model"]["bert_type"]),
                        ("image_size", cfg["data"]["image_size"]), ("input_text_len", cfg["data"]["input_text_len"]),
                        ("train_batch_size", cfg["train_batch_size"]), ("valid_batch_size", cfg["valid_batch_size"]),
                        ("num_workers", cfg["num_workers"]), ("lr", cfg["optimizer"]["lr"]),
                        ("min_epochs", cfg["min_epochs"]), ("max_epochs", cfg["max_epochs"]),
                        ("patience", cfg["early_stopping_patience"]), ("device", cfg["devices"]),
                        ("ckpt_monitor", cfg["official_ckpt_monitor"]), ("ckpt_mode", cfg["monitor_mode"]),
                        ("save_top_k", 1)):
        C.check_matches(logger, args[key], frozen, key)
    src = open(os.path.join(DDCMD, "net", "creratemodel.py")).read()
    for snippet in ("DiceCELoss(sigmoid=True)", "torch.optim.AdamW(self.model.parameters(), lr=self.lr)",
                    "CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)"):
        assert snippet in src, snippet
        logger.info("  creratemodel.py: {} (official = frozen config)".format(snippet))
    view = os.path.join(C.VIEWS, "ddcmd", dataset)
    for split in ("train", "val", "test"):
        prefix = "val" if split == "val" else split
        args[prefix + "_csv_path"] = os.path.join(view, "prompt", "{}_text.csv".format(C.FOLDER[split]))
        args[prefix + "_root_path"] = os.path.join(view, "{}_Folder".format(C.FOLDER[split]))
    args["bert_type"] = os.path.join(HF, os.path.basename(cfg["model"]["bert_type"]))
    args["vision_type"] = os.path.join(HF, os.path.basename(cfg["model"]["vision_type"]))
    args["dataset_name"] = dataset
    return args


def make_dataset(SegData, tokenizer, args, split, mode):
    prefix = "val" if split == "val" else split
    return SegData(csv_path=args[prefix + "_csv_path"], root_path=args[prefix + "_root_path"], tokenizer=tokenizer,
                   mode=mode, image_size=args.image_size, input_text_len=args.input_text_len)


def subset(ds, ids):
    from torch.utils.data import Subset
    index = {name[:-4]: i for i, name in enumerate(ds.mask_list)}
    return Subset(ds, [index[s] for s in ids])


def check_pairing(logger, ds, tokenizer, dataset, split, text_len, n_full=16):
    """Task 1.4 check 1: image, mask and text pairing through the official loader (valid mode, no augmentation)."""
    texts = C.split_texts(dataset, split)
    assert sorted(n[:-4] for n in ds.mask_list) == sorted(texts), "{}/{}: GTs differ from split".format(dataset, split)
    for name in ds.mask_list:  # official lookup: image = img/<mask name>, caption = caption_map[<image name>]
        assert os.path.isfile(os.path.join(ds.root_path, "img", name)), name
        assert ds.caption_map[name] == texts[name[:-4]], name
    rng = np.random.RandomState(0)
    for i in rng.choice(len(ds), n_full, replace=False):
        (image, text), gt = ds[i]
        sid = ds.mask_list[i][:-4]
        ref = C.prepared_image(dataset, split, sid).astype(np.float32)
        z = (ref - ref.mean()) / ref.std()  # NormalizeIntensityd(channel_wise=True)
        assert image.shape == (1, C.SIZE, C.SIZE) and np.allclose(image[0].numpy(), z.T, atol=1e-4), sid
        assert np.array_equal(gt[0].numpy(), C.prepared_mask(dataset, split, sid).T.astype(np.float32)), sid
        tok = tokenizer(texts[sid], padding="max_length", max_length=text_len, truncation=True, return_tensors="pt")
        assert torch.equal(text["input_ids"], tok["input_ids"][0]) and torch.equal(text["attention_mask"],
                                                                                   tok["attention_mask"][0]), sid
    logger.info("[check 1] {} {}: all {} masks have their image and their own text (official loader keys); "
                "{} random samples fully loaded: image = z-scored prepared image, mask = prepared mask (both "
                "transposed by the official PILReader), tokens = BiomedBERT tokenization of the sample's Task 1.1 "
                "text".format(dataset, split, len(ds.mask_list), n_full))


def check_pretrained(logger, model, args):
    """Task 1.4 check 2: official BiomedBERT and ConvNeXt-Tiny weights are loaded (local pinned copies)."""
    from transformers import AutoModel
    for name, part, path in (("BiomedBERT (text)", model.text_encoder.model, args.bert_type),
                             ("ConvNeXt-Tiny (vision)", model.encoder.model, args.vision_type)):
        ref = AutoModel.from_pretrained(path).state_dict()
        mine = part.state_dict()
        shared = [k for k in ref if k in mine]
        assert shared and len(shared) == len(mine), (name, len(shared), len(mine))
        assert all(torch.equal(mine[k].cpu(), ref[k]) for k in shared), name
        logger.info("[check 2] {}: all {} tensors equal the pretrained checkpoint {}".format(
            name, len(shared), os.path.relpath(path, C.ROOT)))
    frozen = all(not p.requires_grad for p in model.text_encoder.parameters())
    trainable = all(p.requires_grad for p in model.encoder.parameters())
    assert frozen and trainable
    logger.info("[check 2] text encoder frozen: {}; vision encoder trainable: {}".format(frozen, trainable))


def build_lightning_model(pl, CreateModel):
    class Phase1DDCMD(CreateModel):
        """CreateModel + common validation Dice (per image, threshold 0.5) from the same validation outputs."""

        def __init__(self, args):
            super().__init__(args)
            self.val_scores, self.common_dice, self.epoch_lr = [], {}, {}

        def validation_step(self, batch, batch_idx):
            out = super().validation_step(batch, batch_idx)
            probs, y = out["preds"][:, 0].cpu().numpy(), out["y"][:, 0].cpu().numpy()
            self.val_scores += [C.dice(p >= C.THRESHOLD, t > 0) for p, t in zip(probs, y)]
            return out

        def validation_epoch_end(self, outputs):
            super().validation_epoch_end(outputs)
            value = float(np.mean(self.val_scores))
            self.val_scores = []
            if not self.trainer.sanity_checking:
                self.common_dice[self.trainer.current_epoch] = value
                self.print("epoch {} val_common_dice {:.4f}".format(self.trainer.current_epoch, value))
            self.log("val_common_dice", value, logger=True)

    return Phase1DDCMD


def history_callback(pl, path):
    fields = ["epoch", "lr", "train_loss", "val_loss", "val_common_dice", "val_dice_official", "val_MIoU_official",
              "seconds"]

    class History(pl.Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            self.start = time.time()
            pl_module.epoch_lr[trainer.current_epoch] = trainer.optimizers[0].param_groups[0]["lr"]

        def on_train_epoch_end(self, trainer, pl_module):
            e = trainer.current_epoch
            h = pl_module.history.get(e, {})
            pl_module.history[e]["seconds"] = round(time.time() - self.start, 1)
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(fields)
                for ep in sorted(pl_module.history):
                    if ep not in pl_module.common_dice:
                        continue
                    r = pl_module.history[ep]
                    w.writerow([ep + 1, pl_module.epoch_lr.get(ep), r.get("train_loss"), r.get("val_loss"),
                                pl_module.common_dice[ep], r.get("val_dice"), r.get("val_MIoU"), r.get("seconds")])
            print("epoch {} lr {:.2e} train_loss {:.4f} val_loss {:.4f} val_common_dice {:.4f} ({:.0f}s)".format(
                e + 1, pl_module.epoch_lr[e], h.get("train_loss", float("nan")), h.get("val_loss", float("nan")),
                pl_module.common_dice.get(e, float("nan")), pl_module.history[e]["seconds"]), flush=True)

    return History()


@torch.no_grad()
def evaluate(seg_model, ds, batch_size, out_dir=None, other_text=False):
    """Mean per-image Dice of the segmentation network on a dataset (valid mode); optional binary predictions saved
    in the original orientation (transposed back). other_text: each batch gets the next batch's text."""
    from torch.utils.data import DataLoader
    seg_model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    names = [ds.dataset.mask_list[i] for i in ds.indices] if hasattr(ds, "indices") else list(ds.mask_list)
    batches = list(loader)
    scores, k = [], 0
    for i, ((image, text), gt) in enumerate(batches):
        if other_text:
            nxt = batches[(i + 1) % len(batches)][0][1]
            text = nxt if nxt["input_ids"].shape == text["input_ids"].shape else \
                {key: torch.roll(v, 1, 0) for key, v in text.items()}
        logits = seg_model([image.cuda(), {key: v.cuda() for key, v in text.items()}])
        probs = torch.sigmoid(logits)[:, 0].cpu().numpy()
        for p, t in zip(probs, gt[:, 0].numpy()):
            scores.append(C.dice(p >= C.THRESHOLD, t > 0))
            if out_dir is not None:
                C.save_mask(out_dir, names[k][:-4], (p >= C.THRESHOLD).T)  # back to the original orientation
            k += 1
    assert k == len(names)
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timing", action="store_true", help="one full-data epoch (compute estimate), runs/timing/")
    parser.add_argument("--quick", action="store_true", help="Task 1.4 checks on a small subset")
    cli = parser.parse_args()

    pl, ddcmd_config, CreateModel, SegData = import_official()
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    cfg = C.load_config("ddcmd")
    out = C.run_dir("ddcmd", cli.dataset, cli.seed, cli.quick, cli.timing)
    logger = C.get_logger(os.path.join(out, "train.log"))
    logger.info("dataset={} seed={} quick={} GPU={} torch={} lightning={}".format(
        cli.dataset, cli.seed, cli.quick, os.environ["CUDA_VISIBLE_DEVICES"], torch.__version__, pl.__version__))
    logger.info("git commit: {}".format(C.git_commit()))
    logger.info("config: {}".format(cfg))
    args = load_args(logger, cfg, ddcmd_config, cli.dataset)
    set_seed_official(cli.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_type, trust_remote_code=True)
    ds_train = make_dataset(SegData, tokenizer, args, "train", "train")
    ds_valid = make_dataset(SegData, tokenizer, args, "val", "valid")
    if cli.quick:
        for split in ("train", "val", "test"):
            check_pairing(logger, make_dataset(SegData, tokenizer, args, split, "valid"), tokenizer, cli.dataset,
                          split, args.input_text_len)
        q = cfg["quick_check"]
        ds_train = subset(ds_train, C.split_ids(cli.dataset, "train")[:q["train_samples"]])
        ds_valid = subset(ds_valid, C.split_ids(cli.dataset, "val")[:q["val_samples"]])
    else:
        assert len(ds_train) == len(C.split_ids(cli.dataset, "train"))
        assert len(ds_valid) == len(C.split_ids(cli.dataset, "val"))
    dl_train = DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    dl_valid = DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.info("train {} / val {} samples".format(len(ds_train), len(ds_valid)))

    model = build_lightning_model(pl, CreateModel)(args)
    seg = model.model
    total, trainable = C.count_params(seg)
    logger.info("model TextGuidedSegmentationModel: {:.2f}M parameters ({:.2f}M trainable)".format(
        total / 1e6, trainable / 1e6))

    if cli.quick:  # checks 2-4 on one training batch
        check_pretrained(logger, seg, args)
        seg.cuda().train()
        (image, text), gt = next(iter(dl_train))
        image, gt = image.cuda(), gt.cuda()
        text = {k: v.cuda() for k, v in text.items()}
        logger.info("[check 3] batch: images {} {}, masks {} values {}, input_ids {}, attention_mask {}".format(
            tuple(image.shape), image.dtype, tuple(gt.shape), sorted(gt.unique().tolist()),
            tuple(text["input_ids"].shape), tuple(text["attention_mask"].shape)))
        gates, spectral_calls = [], []
        hooks = []
        for name in ("decoder16", "decoder8", "decoder4"):
            dec = getattr(seg, name)
            hooks.append(dec.fuse.gate_mlp.register_forward_hook(
                lambda m, i, o, n=name: gates.append((n, torch.sigmoid(o).detach().mean().item()))))
            hooks.append(dec.spectral.register_forward_hook(lambda m, i, o, n=name: spectral_calls.append(n)))
        logits = seg([image, text])
        for h in hooks:
            h.remove()
        assert logits.shape == (image.shape[0], 1, C.SIZE, C.SIZE)
        loss = model.loss_fn(logits, (gt > 0.5).float())
        assert torch.isfinite(loss)
        logger.info("[check 3] forward pass ok: output {} (logits), DiceCELoss {:.4f}".format(tuple(logits.shape),
                                                                                           loss.item()))
        assert len(gates) == 3 and len(spectral_calls) == 3
        logger.info("[check 4] cross-modal decoder called at all 3 decoder stages: text-guided attention gate "
                    "g = sigmoid(gate_mlp) mean " + ", ".join("{} {:.3f}".format(n, g) for n, g in gates) +
                    "; frequency (DCT) text modulation called in " + ", ".join(spectral_calls))
        loss.backward()
        parts = {}
        for name in ("decoder16", "decoder8", "decoder4"):
            dec = getattr(seg, name)
            parts[name + ".fuse.txt_proj"] = dec.fuse.txt_proj
            parts[name + ".fuse.cross"] = dec.fuse.cross
            parts[name + ".fuse.gate_mlp"] = dec.fuse.gate_mlp
            parts[name + ".spectral.txt_proj"] = dec.spectral.txt_proj
        parts["refine1.film (text FiLM)"] = seg.refine1.film
        norms = {k: C.grad_norm(m) for k, m in parts.items()}
        assert all(v > 0 for v in norms.values()), norms
        logger.info("[check 4] non-zero gradients from the segmentation loss in " +
                    ", ".join("{} {:.2e}".format(k, v) for k, v in norms.items()))
        assert all(p.grad is None for p in seg.text_encoder.parameters())
        seg.zero_grad()
        seg.cpu()

    epochs = cfg["quick_check"]["epochs"] if cli.quick else (1 if cli.timing else args.max_epochs)
    checkpoint = ModelCheckpoint(dirpath=out, filename="checkpoint_best", monitor=cfg["monitor"], mode=cfg["monitor_mode"],
                                 save_top_k=1, save_last=False, verbose=True)
    callbacks = [checkpoint, history_callback(pl, os.path.join(out, "history.csv"))]
    if not cli.quick:
        callbacks.append(EarlyStopping(monitor=cfg["monitor"], patience=args.patience, mode=cfg["monitor_mode"]))
    trainer = pl.Trainer(logger=[CSVLogger(save_dir=out, name="csv_logs")], min_epochs=1 if cli.timing else args.min_epochs,
                         max_epochs=epochs, accelerator="gpu", devices=args.device, callbacks=callbacks,
                         enable_progress_bar=False, log_every_n_steps=10, enable_model_summary=False)
    trainer.fit(model, dl_train, dl_valid)
    best = float(checkpoint.best_model_score)
    logger.info("done: best val_common_dice {:.4f}, checkpoint {}".format(best, checkpoint.best_model_path))

    if cli.quick:  # checks 5-6 and text sensitivity
        import pandas as pd
        h = pd.read_csv(os.path.join(out, "history.csv"))
        logger.info("[check 5] training loss: epoch 1 {:.4f} -> epoch {} {:.4f} (min {:.4f})".format(
            h.train_loss.iloc[0], len(h), h.train_loss.iloc[-1], h.train_loss.min()))
        logger.info("[check 5] validation Dice: epoch 1 {:.4f} -> best {:.4f} (epoch {})".format(
            h.val_common_dice.iloc[0], h.val_common_dice.max(), int(h.val_common_dice.idxmax()) + 1))
        state = torch.load(checkpoint.best_model_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state)
        seg = model.model.cuda()
        pred_dir = os.path.join(out, "val_predictions")
        os.makedirs(pred_dir, exist_ok=True)
        own = evaluate(seg, ds_valid, args.valid_batch_size, out_dir=pred_dir)
        assert abs(own - best) < 1e-6, (own, best)
        ids = C.split_ids(cli.dataset, "val")[:cfg["quick_check"]["val_samples"]]
        logger.info("[check 6] validation predictions of the best checkpoint (val_common_dice {:.4f}), transposed "
                    "back to the original orientation:".format(best))
        C.check_saved_predictions(logger, pred_dir, ids, cli.dataset, "val", own)
        other = evaluate(seg, ds_valid, args.valid_batch_size, other_text=True)
        logger.info("[check 4] text changes the prediction (best checkpoint, validation subset): Dice with own "
                    "text {:.4f}, with other samples' texts {:.4f}".format(own, other))
        logger.info("QUICK CHECK DONE")


if __name__ == "__main__":
    main()
