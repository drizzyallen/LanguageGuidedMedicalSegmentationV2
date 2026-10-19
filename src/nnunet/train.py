"""Train nnU-Net v2 2d (fold 0 = LViT train/val) with nnUNetTrainerValDice (configs/nnunet.yaml).

Mirrors nnunetv2.run.run_training.get_trainer_from_args, but instantiates our trainer subclass directly so the
pinned nnU-Net code is not modified. Each seed writes to its own nnUNet_results folder.

Usage: python src/nnunet/train.py --dataset qata --seed 1001 [--quick]
"""
import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nnunet.paths import DATASET_NAMES, NNUNET_PREPROCESSED, ROOT, results_dir, set_env  # noqa: E402

QUICK = {"num_epochs": 5, "num_iterations_per_epoch": 50, "num_val_iterations_per_epoch": 5, "val_case_limit": 32}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["qata", "mosmed"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--quick", action="store_true", help="Task 0.3 quick check")
    args = parser.parse_args()

    set_env(results_dir(args.dataset, args.seed, args.quick))
    import torch
    import yaml
    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunet.trainer import nnUNetTrainerValDice

    with open(os.path.join(ROOT, "configs", "nnunet.yaml")) as f:
        cfg = yaml.safe_load(f)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    folder = os.path.join(NNUNET_PREPROCESSED, DATASET_NAMES[args.dataset])
    plans = load_json(os.path.join(folder, cfg["plans"] + ".json"))
    plans["continue_training"] = False  # as set by nnunetv2.run.run_training.get_trainer_from_args
    dataset_json = load_json(os.path.join(folder, "dataset.json"))
    nnUNetTrainerValDice.project_dataset = args.dataset
    trainer = nnUNetTrainerValDice(plans=plans, configuration=cfg["configuration"], fold=cfg["fold"],
                                   dataset_json=dataset_json, device=torch.device("cuda"))

    # The config records the default nnUNetTrainer values; fail if they differ.
    expected = {"initial_lr": cfg["optimizer"]["lr"], "weight_decay": cfg["optimizer"]["weight_decay"],
                "num_epochs": cfg["max_epochs"], "num_iterations_per_epoch": cfg["iterations_per_epoch"],
                "num_val_iterations_per_epoch": cfg["val_iterations_per_epoch"]}
    for attr, value in expected.items():
        assert getattr(trainer, attr) == value, f"{attr}: trainer {getattr(trainer, attr)} != config {value}"
    trainer.print_to_log_file(f"seed {args.seed}; dataset {args.dataset}; config {cfg}")

    if args.quick:
        nnUNetTrainerValDice.quick_check = True
        nnUNetTrainerValDice.val_case_limit = QUICK["val_case_limit"]
        for attr in ("num_epochs", "num_iterations_per_epoch", "num_val_iterations_per_epoch"):
            setattr(trainer, attr, QUICK[attr])

    trainer.run_training()
    trainer.print_to_log_file(f"done: best validation Dice {trainer.best_val_dice:.4f} "
                              f"at epoch {trainer.best_val_dice_epoch}")


if __name__ == "__main__":
    main()
