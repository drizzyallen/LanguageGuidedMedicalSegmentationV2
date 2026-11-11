"""Task 0.4: predict the LViT test set with a run's best checkpoint and save binary 224x224 masks.

  U-Net:     LViT ValGenerator input, sigmoid output
  TransUNet: uint8 / 255 input, softmax lesion probability
  nnU-Net:   nnUNetPredictor with checkpoint_best_valdice.pth and nnU-Net's default inference
             (sliding window, Gaussian weighting, test-time mirroring)
Lesion = probability >= 0.5. Output: runs/<method>/<dataset>/seed_<seed>/test/predictions/<sample_id>.png
(0 = background, 255 = lesion).

Usage: python src/predict.py --method unet --dataset qata --seed 1001
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream  # noqa: E402
from data.dataset import PreparedDataset, image_path, sample_ids  # noqa: E402
from training_common import ROOT, THRESHOLD, load_config  # noqa: E402


def run_dir(method, dataset, seed):
    return os.path.join(ROOT, "runs", method, dataset, f"seed_{seed}")


def save_mask(out_dir, sample_id, pred):
    assert pred.shape == (224, 224), pred.shape
    assert cv2.imwrite(os.path.join(out_dir, f"{sample_id}.png"), pred.astype(np.uint8) * 255)


@torch.no_grad()
def predict_torch(model, dataset, lesion_prob, out_dir, batch_size):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    n = 0
    for images, _, ids in loader:
        probs = lesion_prob(model, images.cuda()).cpu().numpy()
        for sample_id, p in zip(ids, probs):
            save_mask(out_dir, sample_id, p >= THRESHOLD)
            n += 1
    return n


def predict_unet(dataset, ckpt_path, out_dir):
    import train_unet
    cfg = load_config("unet")
    model = upstream.lvit_unet().UNet(n_channels=cfg["model"]["n_channels"],
                                      n_classes=cfg["model"]["n_classes"]).cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)  # own checkpoint
    model.load_state_dict(ckpt["state_dict"])
    size = [cfg["input_size"]] * 2
    data = PreparedDataset(dataset, "test", train_unet.LViTTransform(upstream.lvit_load_dataset().ValGenerator(size)))
    return predict_torch(model, data, train_unet.lesion_prob, out_dir, cfg["batch_size"]), ckpt


def predict_transunet(dataset, ckpt_path, out_dir):
    import train_transunet
    cfg = load_config("transunet")
    modeling = upstream.transunet_modeling()
    mcfg = cfg["model"]
    config_vit = modeling.CONFIGS[mcfg["vit_name"]]
    config_vit.n_classes = mcfg["n_classes"]
    config_vit.n_skip = mcfg["n_skip"]
    grid = cfg["input_size"] // mcfg["vit_patches_size"]
    config_vit.patches.grid = (grid, grid)
    model = modeling.VisionTransformer(config_vit, img_size=cfg["input_size"], num_classes=config_vit.n_classes).cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)  # own checkpoint
    model.load_state_dict(ckpt["state_dict"])
    data = PreparedDataset(dataset, "test", train_transunet.val_transform)
    return predict_torch(model, data, train_transunet.lesion_prob, out_dir, cfg["batch_size"]), ckpt


def predict_nnunet(dataset, seed, out_dir):
    from nnunet.paths import results_dir, set_env
    set_env(results_dir(dataset, seed))
    os.environ["nnUNet_extTrainer"] = os.path.join(ROOT, "src", "nnunet")  # official hook for external trainers
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    cfg = load_config("nnunet")
    model_dir = glob.glob(os.path.join(results_dir(dataset, seed), "Dataset*", f"{cfg['trainer']}__*__2d"))
    assert len(model_dir) == 1, model_dir
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=torch.device("cuda"),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_dir[0], use_folds=(cfg["fold"],),
                                                   checkpoint_name="checkpoint_best_valdice.pth")
    reader = NaturalImage2DIO()
    n = 0
    for sample_id in sample_ids(dataset, "test"):
        image, props = reader.read_images([image_path(dataset, "test", sample_id)])
        _, probs = predictor.predict_single_npy_array(image, props, None, None, True)
        save_mask(out_dir, sample_id, np.squeeze(probs[1]) >= THRESHOLD)
        n += 1
    ckpt = torch.load(os.path.join(model_dir[0], f"fold_{cfg['fold']}", "checkpoint_best_valdice.pth"),
                      map_location="cpu", weights_only=False)
    return n, {"epoch": ckpt["current_epoch"], "path": os.path.join(model_dir[0], "fold_0", "checkpoint_best_valdice.pth")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["unet", "transunet", "nnunet"])
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    base = run_dir(args.method, args.dataset, args.seed)
    out_dir = os.path.join(base, "test", "predictions")
    os.makedirs(out_dir, exist_ok=True)
    if args.method == "nnunet":
        n, info = predict_nnunet(args.dataset, args.seed, out_dir)
        checkpoint = {"path": os.path.relpath(info["path"], ROOT), "epoch": info["epoch"]}
    else:
        ckpt_path = os.path.join(base, "checkpoint_best.pth")
        fn = predict_unet if args.method == "unet" else predict_transunet
        n, ckpt = fn(args.dataset, ckpt_path, out_dir)
        checkpoint = {"path": os.path.relpath(ckpt_path, ROOT), "epoch": ckpt["epoch"], "val_dice": ckpt["val_dice"]}
    expected = len(sample_ids(args.dataset, "test"))
    assert n == expected == len(os.listdir(out_dir)), (n, expected)
    with open(os.path.join(base, "test", "prediction_info.json"), "w") as f:
        json.dump({"method": args.method, "dataset": args.dataset, "seed": args.seed, "n_predictions": n,
                   "threshold": THRESHOLD, "checkpoint": checkpoint}, f, indent=2)
    print(f"[{args.method} {args.dataset} {args.seed}] {n} test predictions -> {out_dir}")


if __name__ == "__main__":
    main()
