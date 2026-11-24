"""Float-valued molecular featurisation.

We deliberately do NOT use PyG's integer-encoded MoleculeNet features. Gradient
attribution and feature-masking both need a continuous input space with a
well-defined zero baseline, so every atom and bond is one-hot encoded into a
float vector. The all-zeros vector then means "no information about this atom",
which is exactly the baseline the deletion curves in faithfulness.py assume.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ATOM_SYMBOLS: Sequence[str] = ("C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se")
DEGREES: Sequence[int] = (0, 1, 2, 3, 4, 5)
FORMAL_CHARGES: Sequence[int] = (-2, -1, 0, 1, 2)
HYBRIDISATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
NUM_HS: Sequence[int] = (0, 1, 2, 3, 4)
CHIRALITIES = (
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
)
BOND_TYPES = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
BOND_STEREO = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
)


def _one_hot(value, choices: Sequence) -> List[float]:
    """One-hot with a trailing catch-all slot, so unseen values never crash."""
    vec = [0.0] * (len(choices) + 1)
    try:
        vec[list(choices).index(value)] = 1.0
    except ValueError:
        vec[-1] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> List[float]:
    return (
        _one_hot(atom.GetSymbol(), ATOM_SYMBOLS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDISATIONS)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
        + _one_hot(atom.GetChiralTag(), CHIRALITIES)
    )


def bond_features(bond: Chem.Bond) -> List[float]:
    return (
        _one_hot(bond.GetBondType(), BOND_TYPES)
        + [float(bond.GetIsConjugated()), float(bond.IsInRing())]
        + _one_hot(bond.GetStereo(), BOND_STEREO)
    )


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))


def mol_to_data(smiles: str, y: Optional[float] = None) -> Optional[Data]:
    """SMILES -> PyG Data with float x and edge_attr. Returns None if unparseable."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    src, dst, eattr = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        src += [i, j]
        dst += [j, i]
        eattr += [feats, feats]

    if len(src) == 0:  # single-atom molecule: add a self loop so convs have an edge
        src, dst = [0], [0]
        eattr = [[0.0] * BOND_DIM]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(eattr, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = smiles
    if y is not None:
        data.y = torch.tensor([[float(y)]], dtype=torch.float)
    return data


def featurize_many(smiles: Sequence[str], labels: Sequence[float]) -> List[Data]:
    out = []
    for smi, y in zip(smiles, labels):
        d = mol_to_data(smi, y)
        if d is not None:
            out.append(d)
    return out


def heavy_atom_counts(dataset: Sequence[Data]) -> np.ndarray:
    return np.array([int(d.num_nodes) for d in dataset])
