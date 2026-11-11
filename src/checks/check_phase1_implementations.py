"""Check that the pinned official Phase 1 implementations build and run (no QaTa/MosMed data).

Each method runs in its own environment (envs/README.md):
  python src/checks/check_phase1_implementations.py --method lvit      # env lgms_lvit
  python src/checks/check_phase1_implementations.py --method reclmis   # env lgms_reclmis
  python src/checks/check_phase1_implementations.py --method ddcmd     # env lgms_ddcmd

For each: build the official model as its training script does, encode an example sentence with the method's
official text pipeline, confirm the pretrained text/vision component is loaded, and run one forward pass on a
random 224x224 image.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = os.path.join(ROOT, "external")
PRETRAINED = os.path.join(EXT, "pretrained")
SENTENCE = "Bilateral pulmonary infection, two infected areas, all in the lower left lung and lower right lung."


def import_repo(name):
    path = os.path.join(EXT, name)
    sys.path.insert(0, path)
    return path


def report(method, out, extra):
    print(f"[{method}] ok  output {tuple(out.shape)}, {extra}")


def check_lvit(device):
    import_repo("LViT")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    import Config  # LViT Config.py sets CUDA_VISIBLE_DEVICES="0" on import; restore ours
    if visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    from bert_embedding import BertEmbedding
    from nets.LViT import LViT

    # Text pipeline of LViT Load_Dataset.ImageToImage2D: BertEmbedding on the sentence list, first sentence,
    # at most 10 token embeddings.
    embedding = BertEmbedding()
    tokens = embedding(SENTENCE.split("\n"))
    text = np.array(tokens[0][1])[:10, :]
    text = torch.from_numpy(text).float().unsqueeze(0).to(device)

    model = LViT(Config.get_CTranS_config(), n_channels=Config.n_channels, n_classes=Config.n_labels).to(device).eval()
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x, text)
    assert out.shape == (1, 1, 224, 224) and 0 <= out.min() and out.max() <= 1
    n = sum(p.numel() for p in model.parameters())
    report("LViT-T", out, f"sigmoid output; text input {tuple(text.shape)} from BertEmbedding "
                          f"(bert_12_768_12 / book_corpus_wiki_en_uncased, default); params {n / 1e6:.2f}M")


def check_reclmis(device):
    repo = import_repo("RecLMIS")
    import clip
    import Config_covid19 as config
    from nets.RecLMIS import RecLMIS

    ckpt_path = os.path.join(repo, "nets", "ViT-B-32.pt")
    assert os.path.realpath(ckpt_path) == os.path.join(PRETRAINED, "clip", "ViT-B-32.pt")
    model = RecLMIS(config, config.get_ViT_config(), n_channels=config.n_channels, n_classes=config.n_labels)
    reference = torch.jit.load(ckpt_path, map_location="cpu").state_dict()["token_embedding.weight"].float()
    assert torch.equal(model.clip.token_embedding.weight.detach().cpu(), reference), "CLIP weights not loaded"
    model = model.to(device).eval()

    # Text pipeline of RecLMIS Load_Dataset: clip.tokenize(context_length=config.token_len, truncate=True)
    text_token = clip.tokenize(SENTENCE.split("\n"), context_length=config.token_len, truncate=True).to(device)
    text_mask = (text_token != 0).int()
    x = torch.randn(1, 3, 224, 224, device=device)
    masks = torch.zeros(1, 224, 224, device=device)
    with torch.no_grad():
        out, _, _ = model(x, masks, text_token, text_mask, mode="test")
    n = sum(p.numel() for p in model.parameters())
    report("RecLMIS", out, f"CLIP {config.get_ViT_config().clip_backbone} loaded (frozen={config.get_ViT_config().frozen_clip}); "
                           f"text tokens {tuple(text_token.shape)}; params {n / 1e6:.2f}M")


def check_ddcmd(device):
    import_repo("DD-CMD")
    import yaml
    from net.model import TextGuidedSegmentationModel
    from transformers import AutoModel, AutoTokenizer

    with open(os.path.join(EXT, "DD-CMD", "config", "train.yaml")) as f:
        cfg = yaml.safe_load(f)
    hf = os.path.join(PRETRAINED, "huggingface")
    bert = os.path.join(hf, os.path.basename(cfg["MODEL"]["bert_type"]))
    vision = os.path.join(hf, os.path.basename(cfg["MODEL"]["vision_type"]))
    model = TextGuidedSegmentationModel(bert_type=bert, vision_type=vision)
    for name, part, path in (("BiomedBERT", model.text_encoder.model, bert), ("ConvNeXt-Tiny", model.encoder.model, vision)):
        ref = AutoModel.from_pretrained(path).state_dict()
        mine = part.state_dict()
        same = all(torch.equal(mine[k], ref[k]) for k in ref if k in mine)
        assert same and len(set(ref) & set(mine)) > 0, f"{name} weights not loaded"
    model = model.to(device).eval()

    # Text pipeline of DD-CMD utils/dataset.py: tokenizer(padding="max_length", max_length=input_text_len)
    tokenizer = AutoTokenizer.from_pretrained(bert, trust_remote_code=True)
    tok = tokenizer(SENTENCE, padding="max_length", max_length=cfg["DATA"]["input_text_len"], truncation=True,
                    return_tensors="pt")
    text = {k: v.to(device) for k, v in tok.items()}
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model([x, text])
    n = sum(p.numel() for p in model.parameters())
    report("DD-CMD", out, f"BiomedBERT + ConvNeXt-Tiny loaded from local pinned copies; text tokens "
                          f"{tuple(text['input_ids'].shape)}; params {n / 1e6:.2f}M")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["lvit", "reclmis", "ddcmd"])
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}, device {device}")
    {"lvit": check_lvit, "reclmis": check_reclmis, "ddcmd": check_ddcmd}[args.method](device)


if __name__ == "__main__":
    main()
