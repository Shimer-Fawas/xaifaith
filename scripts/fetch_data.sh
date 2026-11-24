#!/usr/bin/env bash
# Fetch every dataset the paper uses. Run from the repository root.
set -euo pipefail

echo "[1/2] MoleculeNet (downloaded by PyTorch Geometric into data/moleculenet/)"
python - <<'PY'
from torch_geometric.datasets import MoleculeNet
for name in ("BACE", "BBBP", "ClinTox", "ESOL", "Lipo", "SIDER", "Tox21"):
    MoleculeNet(root="data/moleculenet", name=name)
    print("  ok:", name)
PY

echo "[2/2] B-XAIC benchmark (Hugging Face: mproszewska/B-XAIC)"
command -v hf >/dev/null || pip install -q "huggingface_hub[cli]"
hf download mproszewska/B-XAIC --repo-type dataset --local-dir data/bxaic
