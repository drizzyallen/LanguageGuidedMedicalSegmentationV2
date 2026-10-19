"""Check that the pinned official implementations build and run (no real data).

U-Net:     LViT nets/UNet.py, built as in LViT Config.py (n_channels=3, n_labels=1, sigmoid).
TransUNet: official R50-ViT-B_16, built as in TransUNet train.py (n_skip=3, grid=img/16),
           2 classes, ImageNet-21k weights loaded with the official load_from().

Usage: python src/checks/check_public_implementations.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = os.path.join(ROOT, "external")
WEIGHTS = os.path.join(EXT, "pretrained", "imagenet21k", "R50+ViT-B_16.npz")
IMG_SIZE = 224


def check_unet(device):
    sys.path.insert(0, os.path.join(EXT, "LViT"))
    from nets.UNet import UNet
    sys.path.pop(0)

    model = UNet(n_channels=3, n_classes=1).to(device).eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 1, IMG_SIZE, IMG_SIZE), y.shape
    assert 0.0 <= y.min().item() and y.max().item() <= 1.0, "expected sigmoid output"
    n = sum(p.numel() for p in model.parameters())
    print(f"[U-Net]     ok  output {tuple(y.shape)} in [0,1] (sigmoid), params {n / 1e6:.2f}M")


def check_transunet(device):
    sys.path.insert(0, os.path.join(EXT, "TransUNet"))
    from networks.vit_seg_modeling import CONFIGS, VisionTransformer
    sys.path.pop(0)

    config = CONFIGS["R50-ViT-B_16"]
    config.n_classes = 2
    config.n_skip = 3
    config.patches.grid = (IMG_SIZE // 16, IMG_SIZE // 16)
    config.pretrained_path = WEIGHTS

    model = VisionTransformer(config, img_size=IMG_SIZE, num_classes=config.n_classes)
    before = model.transformer.embeddings.patch_embeddings.weight.detach().clone()
    weights = np.load(config.pretrained_path, allow_pickle=False)
    model.load_from(weights=weights)
    after = model.transformer.embeddings.patch_embeddings.weight.detach()
    expected = torch.from_numpy(weights["embedding/kernel"]).permute(3, 2, 0, 1)
    assert not torch.equal(before, after), "pretrained weights were not loaded"
    assert torch.allclose(after, expected), "patch embedding does not match checkpoint"

    model = model.to(device).eval()
    x = torch.randn(2, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, 2, IMG_SIZE, IMG_SIZE), y.shape
    n = sum(p.numel() for p in model.parameters())
    print(f"[TransUNet] ok  R50-ViT-B_16, output {tuple(y.shape)} (2-class logits), "
          f"ImageNet-21k weights loaded, params {n / 1e6:.2f}M")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device {device}")
    check_unet(device)
    check_transunet(device)


if __name__ == "__main__":
    main()
