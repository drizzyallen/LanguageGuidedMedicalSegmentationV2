"""Import code from the pinned official repositories without modifying them.

LViT and TransUNet both have a top-level `utils.py`, so modules are loaded by file path under unique names.
"""
import importlib.util
import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXTERNAL = os.path.join(ROOT, "external")
LVIT = os.path.join(EXTERNAL, "LViT")
TRANSUNET = os.path.join(EXTERNAL, "TransUNet")
TRANSUNET_WEIGHTS = os.path.join(EXTERNAL, "pretrained", "imagenet21k", "R50+ViT-B_16.npz")


def _load(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def lvit_unet():
    return _load("lvit_unet", os.path.join(LVIT, "nets", "UNet.py"))


def lvit_utils():
    return _load("lvit_utils", os.path.join(LVIT, "utils.py"))


def lvit_load_dataset():
    # Load_Dataset.py imports bert_embedding for its text datasets. Only the image transforms
    # (RandomGenerator, ValGenerator) are used here, so a placeholder module satisfies the import.
    if "bert_embedding" not in sys.modules:
        placeholder = types.ModuleType("bert_embedding")
        placeholder.BertEmbedding = None
        sys.modules["bert_embedding"] = placeholder
    return _load("lvit_load_dataset", os.path.join(LVIT, "Load_Dataset.py"))


def transunet_modeling():
    if TRANSUNET not in sys.path:
        sys.path.insert(0, TRANSUNET)
    import networks.vit_seg_modeling as modeling  # package with relative imports
    return modeling


def transunet_utils():
    return _load("transunet_utils", os.path.join(TRANSUNET, "utils.py"))


def transunet_dataset():
    return _load("transunet_dataset_synapse", os.path.join(TRANSUNET, "datasets", "dataset_synapse.py"))
