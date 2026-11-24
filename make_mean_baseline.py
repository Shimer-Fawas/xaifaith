"""Mean atom-feature vector over drug-like chemical space.

Computed from B-XAIC's ChEMBL-35 molecules. No train/test split is involved, so
this cannot leak: it is a fixed reference input, not a fitted quantity.
"""
import argparse
import numpy as np
import torch
from rdkit import Chem
from xaifaith.featurize import atom_features, ATOM_DIM

ap = argparse.ArgumentParser()
ap.add_argument("--sdf", required=True)
ap.add_argument("--out", default="runs/baselines/chembl_mean.pt")
ap.add_argument("--max-mols", type=int, default=0)
a = ap.parse_args()

total = np.zeros(ATOM_DIM, dtype=np.float64)
n_atoms = n_mols = 0
for mol in Chem.ForwardSDMolSupplier(a.sdf, removeHs=True, sanitize=True):
    if mol is None:
        continue
    for atom in mol.GetAtoms():
        total += np.asarray(atom_features(atom), dtype=np.float64)
        n_atoms += 1
    n_mols += 1
    if a.max_mols and n_mols >= a.max_mols:
        break

if n_atoms == 0:
    raise SystemExit("no atoms read - check --sdf")
vec = torch.from_numpy(total / n_atoms).float()
torch.save({"vector": vec, "n_atoms": n_atoms, "n_mols": n_mols, "source": a.sdf}, a.out)
print(f"{n_mols} molecules, {n_atoms} atoms -> {a.out}")
print(f"dim={vec.numel()} min={vec.min():.4f} max={vec.max():.4f} sum={vec.sum():.4f}")
