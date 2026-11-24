#!/usr/bin/env bash
# Environment setup for Ubuntu / WSL2 with an RTX 4090 (CUDA 12.x).
#
#   bash setup_env.sh
#   source .venv/bin/activate
#
# Check your driver first:  nvidia-smi
# Under WSL the driver comes from the Windows host - do NOT install a Linux
# driver inside WSL, only the CUDA runtime that ships with the torch wheels.

set -euo pipefail

PYTHON="${PYTHON:-python3.11}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "installing $PYTHON"
  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-venv python3.11-dev build-essential
fi

"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

echo "installing torch (${TORCH_CUDA})"
pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

echo "installing torch-geometric"
pip install torch_geometric
# Optional compiled kernels. PyG runs without them; they speed up sparse ops.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install pyg_lib torch_scatter torch_sparse \
  -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}+${TORCH_CUDA}.html" || \
  echo "optional PyG kernels unavailable for this torch build - continuing without them"

echo "installing the rest"
pip install -e .

python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
import torch_geometric, rdkit, lightgbm
print("torch_geometric", torch_geometric.__version__, "| rdkit", rdkit.__version__)
PY

echo
echo "done. next:  source .venv/bin/activate && python -m xaifaith.smoke_test"
