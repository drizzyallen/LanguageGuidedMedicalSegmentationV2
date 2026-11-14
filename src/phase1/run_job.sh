#!/usr/bin/env bash
# Task 1.6: one Phase 1 run = training -> test predictions of the best checkpoint -> common evaluation.
# Usage: CUDA_VISIBLE_DEVICES=<gpu> bash src/phase1/run_job.sh <lvit|reclmis|ddcmd> <qata|mosmed> <seed>
# Outputs: runs/<method>/<dataset>/seed_<seed>/{train.log, history.csv, checkpoint_best.*, test.log,
#          test/predictions/*.png, test/prediction_info.json, test/per_case_metrics.csv, test/test_summary.json}
set -euo pipefail
cd "$(dirname "$0")/../.."
METHOD="$1"
DATASET="$2"
SEED="$3"
ENVS=/data/ramialle/miniconda3/envs
PY="$ENVS/lgms_${METHOD}/bin/python"
echo "=== $(date '+%F %T') train $METHOD $DATASET $SEED (GPU $CUDA_VISIBLE_DEVICES)"
"$PY" -W ignore "src/phase1/train_${METHOD}.py" --dataset "$DATASET" --seed "$SEED"
echo "=== $(date '+%F %T') test predictions"
"$PY" -W ignore "src/phase1/train_${METHOD}.py" --dataset "$DATASET" --seed "$SEED" --test
echo "=== $(date '+%F %T') common evaluation (lgms_phase0)"
"$ENVS/lgms_phase0/bin/python" src/evaluation/evaluate.py --method "$METHOD" --dataset "$DATASET" --seed "$SEED"
echo "=== $(date '+%F %T') JOB DONE"
