"""nnUNetTrainer + checkpoint selection by validation Dice on full 224x224 predictions.

Training is the unchanged default nnUNetTrainer. After every epoch, all LViT validation cases are predicted with
nnU-Net's own predictor (sliding window, no test-time mirroring), mapped back to the 224x224 image with nnU-Net's
export logic, thresholded at lesion probability 0.5, and scored against the prepared masks (data/prepared/).
The best epoch is saved as checkpoint_best_valdice.pth. nnU-Net's own checkpoint_best.pth (EMA pseudo Dice)
is still written but not used.
"""
import os
import warnings

import cv2
import numpy as np
import torch
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from data.dataset import mask_path
from evaluation.metrics import dice

THRESHOLD = 0.5


class nnUNetTrainerValDice(nnUNetTrainer):
    project_dataset = None        # "qata" / "mosmed", set by src/nnunet/train.py
    val_case_limit = None         # only for the quick check
    quick_check = False

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.best_val_dice = None
        self.best_val_dice_epoch = None

    def on_train_start(self):
        super().on_train_start()
        if self.quick_check:
            self._quick_sanity_checks()

    def _quick_sanity_checks(self):
        """Doc Task 0.3 quick check items 1-4 on one training batch."""
        batch = next(self.dataloader_train)
        data, target = batch["data"], batch["target"]
        target0 = target[0] if isinstance(target, list) else target
        self.print_to_log_file(f"[check 1] loaded one batch: data {tuple(data.shape)}, target {tuple(target0.shape)}, "
                               f"target values {sorted(torch.unique(target0).tolist())}")
        data = data.to(self.device, non_blocking=True)
        target = [t.to(self.device, non_blocking=True) for t in target] if isinstance(target, list) \
            else target.to(self.device, non_blocking=True)
        self.network.train()
        output = self.network(data)
        out0 = output[0] if isinstance(output, (list, tuple)) else output
        self.print_to_log_file(f"[check 2] forward pass ok: output {tuple(out0.shape)} "
                               f"(+{len(output) - 1} deep supervision outputs)")
        assert out0.shape[1] == 2 and tuple(out0.shape[-2:]) == tuple(target0.shape[-2:]) == (224, 224)
        self.print_to_log_file(f"[check 3] shapes ok: output spatial {tuple(out0.shape[-2:])} == "
                               f"target spatial {tuple(target0.shape[-2:])}")
        loss = self.loss(output, target)
        assert torch.isfinite(loss)
        self.print_to_log_file(f"[check 4] loss computed: {loss.item():.4f}")

    @torch.no_grad()
    def compute_validation_dice(self):
        self.set_deep_supervision_enabled(False)
        self.network.eval()
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                    perform_everything_on_device=True, device=self.device, verbose=False,
                                    verbose_preprocessing=False, allow_tqdm=False)
        predictor.manual_initialization(self.network, self.plans_manager, self.configuration_manager, None,
                                        self.dataset_json, self.__class__.__name__,
                                        self.inference_allowed_mirroring_axes)
        _, val_keys = self.do_split()
        if self.val_case_limit is not None:
            val_keys = val_keys[:self.val_case_limit]
        dataset_val = self.dataset_class(self.preprocessed_dataset_folder, val_keys)
        scores = []
        for key in dataset_val.identifiers:
            data, _, _, properties = dataset_val.load_case(key)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = torch.from_numpy(data[:])
            logits = predictor.predict_sliding_window_return_logits(data).cpu()
            _, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
                logits, self.plans_manager, self.configuration_manager, self.label_manager, properties,
                return_probabilities=True)
            pred = np.squeeze(probs[1]) >= THRESHOLD
            gt = cv2.imread(mask_path(self.project_dataset, "val", key), cv2.IMREAD_GRAYSCALE) > 0
            assert pred.shape == gt.shape == (224, 224), (pred.shape, gt.shape)
            scores.append(dice(pred, gt))
        self.set_deep_supervision_enabled(self.enable_deep_supervision)
        self.network.train()
        return float(np.mean(scores)), len(scores)

    def on_epoch_end(self):
        val_dice, n = self.compute_validation_dice()
        self.print_to_log_file(f"Validation Dice (full 224x224, {n} cases): {np.round(val_dice, 4)}")
        if self.best_val_dice is None or val_dice > self.best_val_dice:
            self.best_val_dice, self.best_val_dice_epoch = val_dice, self.current_epoch
            self.print_to_log_file(f"New best validation Dice: {np.round(val_dice, 4)} -> checkpoint_best_valdice.pth")
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_best_valdice.pth"))
        super().on_epoch_end()
