#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
ENV_NAME="arap"
PYTHON_VERSION="3.10"

REQUIREMENTS_FILE="requirements.txt"
SCRIPT_PATH="arap_main.py"
WEIGHT_TYPE="cot" # Options: cot, uniform
# ==================

echo "==> [1/5] Check conda"
if ! command -v conda &> /dev/null; then
  echo "ERROR: conda not found. Please install Anaconda or Miniconda."
  exit 1
fi

# Enable conda inside script
eval "$(conda shell.bash hook)"

echo "==> [2/5] Create conda environment: $ENV_NAME"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

echo "==> [3/5] Activate environment"
conda activate "$ENV_NAME"

echo "==> [4/5] Install dependencies from $REQUIREMENTS_FILE"
if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
  echo "ERROR: $REQUIREMENTS_FILE not found."
  exit 1
fi

pip install --upgrade pip
pip install -r "$REQUIREMENTS_FILE"

echo "==> [5/5] Run ARAP"
if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "ERROR: $SCRIPT_PATH not found."
  exit 1
fi

python "$SCRIPT_PATH" --weight "$WEIGHT_TYPE"

echo "==> Done."
