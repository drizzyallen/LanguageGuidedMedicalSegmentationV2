#!/usr/bin/env bash
# Task 0.4: test predictions + common evaluation for one method, both datasets, all three seeds.
# Usage: bash src/test_phase0.sh <method> <gpu>     (logs: runs/logs/test_<method>.log)
set -euo pipefail
cd "$(dirname "$0")/.."
METHOD="$1"
GPU="$2"
for dataset in qata mosmed; do
  for seed in 1001 1002 1003; do
    CUDA_VISIBLE_DEVICES="$GPU" python src/predict.py --method "$METHOD" --dataset "$dataset" --seed "$seed"
    python src/evaluation/evaluate.py --method "$METHOD" --dataset "$dataset" --seed "$seed"
  done
done
echo "ALL DONE $METHOD"
