#!/usr/bin/env bash
# Tasks 1.2-1.4 quick checks: for each method and dataset, run the method's training wrapper with --quick
# (pairing check through the official loader, pretrained-component check, forward pass, text/cross-modal path
# check, short training on a small subset, validation predictions). Logs: runs/quick_check/logs/,
# per-run outputs: runs/quick_check/<method>/<dataset>/seed_1001/.
#
# Usage: bash src/checks/run_phase1_quick_checks.sh [GPU_ID]
set -euo pipefail
cd "$(dirname "$0")/../.."
GPU="${1:-0}"
ENVS=/data/ramialle/miniconda3/envs
mkdir -p runs/quick_check/logs
for dataset in qata mosmed; do
  CUDA_VISIBLE_DEVICES="$GPU" "$ENVS/lgms_lvit/bin/python" -W ignore src/phase1/train_lvit.py \
    --dataset "$dataset" --seed 1001 --quick > "runs/quick_check/logs/lvit_${dataset}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" "$ENVS/lgms_reclmis/bin/python" -W ignore src/phase1/train_reclmis.py \
    --dataset "$dataset" --seed 1001 --quick > "runs/quick_check/logs/reclmis_${dataset}.log" 2>&1
  CUDA_VISIBLE_DEVICES="$GPU" "$ENVS/lgms_ddcmd/bin/python" -W ignore src/phase1/train_ddcmd.py \
    --dataset "$dataset" --seed 1001 --quick > "runs/quick_check/logs/ddcmd_${dataset}.log" 2>&1
done
grep -h "QUICK CHECK DONE" runs/quick_check/logs/*.log | wc -l
