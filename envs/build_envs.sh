#!/usr/bin/env bash
# Build the three Phase 1 conda environments (one per official repository).
# Usage: bash envs/build_envs.sh {lvit|reclmis|ddcmd}
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
TORCH_CU111=https://download.pytorch.org/whl/cu111/torch_stable.html
TORCH_CU113=https://download.pytorch.org/whl/cu113
case "$1" in
  lvit)     # LViT README: Python 3.7; install bert-embedding first, then requirements (numpy 1.17.5)
    conda create -y -q -n lgms_lvit python=3.7
    conda activate lgms_lvit
    pip install -q bert-embedding==1.0.1   # unpinned in LViT requirements.txt; version used here
    pip install -q -r envs/lvit_requirements.txt -f "$TORCH_CU111" ;;
  reclmis)  # RecLMIS README: Python 3.9, pip install -r requirements.txt
    conda create -y -q -n lgms_reclmis python=3.9
    conda activate lgms_reclmis
    pip install -q --upgrade pip
    pip install -q -r envs/reclmis_requirements.txt -f "$TORCH_CU111"
    # fairseq 0.10.2 has no Python 3.9 wheel and its setup.py fails under recent setuptools:
    # build it against the installed torch with an older setuptools and no build isolation
    pip install -q "setuptools<60" cython
    # the PyPI sdist of 0.10.2 ships fairseq/examples as a directory, which makes setup.py fail;
    # install the same release from its official git tag (v0.10.2 = 83e615d)
    pip install -q "fairseq @ git+https://github.com/facebookresearch/fairseq.git@83e615d66905b8ca7483122a37da1a85f13f4b8e" --no-build-isolation ;;
  ddcmd)    # DD-CMD README: Python 3.8, pip install -r requirements.txt
    conda create -y -q -n lgms_ddcmd python=3.8
    conda activate lgms_ddcmd
    pip install -q -r envs/ddcmd_requirements.txt --extra-index-url "$TORCH_CU113" ;;
esac
pip check || true
pip freeze > "envs/${1}_freeze.txt"
echo "BUILD DONE $1"
