"""nnU-Net folders for this project. Must be applied before importing nnunetv2."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NNUNET_RAW = os.path.join(ROOT, "data", "nnunet", "raw")
NNUNET_PREPROCESSED = os.path.join(ROOT, "data", "nnunet", "preprocessed")
DATASET_NAMES = {"qata": "Dataset501_QaTaCOV19", "mosmed": "Dataset502_MosMedDataPlus"}
DATASET_IDS = {"qata": 501, "mosmed": 502}


def results_dir(dataset, seed, quick=False):
    return os.path.join(ROOT, "runs", "quick_check" if quick else "", "nnunet", dataset, f"seed_{seed}")


def set_env(results):
    os.environ["nnUNet_raw"] = NNUNET_RAW
    os.environ["nnUNet_preprocessed"] = NNUNET_PREPROCESSED
    os.environ["nnUNet_results"] = results
