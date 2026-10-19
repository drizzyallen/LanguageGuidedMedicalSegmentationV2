#!/usr/bin/env bash
# Task 0.3 quick check: for each method and dataset, load one batch, run one forward pass, check output/mask
# shapes, compute the loss, and run a short training on a small subset. Logs go to runs/quick_check/.
#
# Usage: bash src/checks/run_quick_checks.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${1:-0}"
rm -rf runs/quick_check
for dataset in qata mosmed; do
  CUDA_VISIBLE_DEVICES="$GPU" python src/train_unet.py --dataset "$dataset" --seed 1001 --quick
  CUDA_VISIBLE_DEVICES="$GPU" python src/train_transunet.py --dataset "$dataset" --seed 1001 --quick
  CUDA_VISIBLE_DEVICES="$GPU" python src/nnunet/train.py --dataset "$dataset" --seed 1001 --quick
done
