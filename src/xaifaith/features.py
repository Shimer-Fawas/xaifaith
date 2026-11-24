"""Cheap features for the faithfulness predictor.

Hard constraint: every feature here must be computable WITHOUT the faithfulness
label, i.e. without any deletion curves. If a feature needs masked forward
passes you have simply reinvented the metric, and the predictor is worthless in
deployment because computing the feature costs as much as computing the answer.

Three groups:
  1. molecule   - RDKit descriptors, no model involved
  2. model      - confidence, uncertainty, distance to training manifold
  3. attribution- shape statistics of the attribution vector itself
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .explain import _single_graph_inputs
from .models import enable_mc_dropout


# --------------------------------------------------------------------------- #
# 1. molecule-level
# --------------------------------------------------------------------------- #

def molecule_features(smiles: str) -> Dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    ring_info = mol.GetRingInfo()
    n_atoms = mol.GetNumAtoms()
    return {
        "mol_n_atoms": float(n_atoms),
        "mol_n_bonds": float(mol.GetNumBonds()),
        "mol_mw": float(Descriptors.MolWt(mol)),
        "mol_logp": float(Crippen.MolLogP(mol)),
        "mol_tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "mol_n_rings": float(ring_info.NumRings()),
        "mol_aromatic_frac": float(sum(a.GetIsAromatic() for a in mol.GetAtoms()) / max(n_atoms, 1)),
        "mol_frac_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "mol_rotatable": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "mol_hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "mol_hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "mol_heteroatom_frac": float(
            sum(a.GetSymbol() not in ("C", "H") for a in mol.GetAtoms()) / max(n_atoms, 1)
        ),
        "mol_bertz": float(Descriptors.BertzCT(mol)),
        "mol_n_stereocentres": float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
        # how branched the graph is - a proxy for how many distinct substructures
        # an explainer has to choose between
        "mol_mean_degree": float(np.mean([a.GetDegree() for a in mol.GetAtoms()])),
        "mol_max_degree": float(max(a.GetDegree() for a in mol.GetAtoms())),
    }


# --------------------------------------------------------------------------- #
# 2. model-state
# --------------------------------------------------------------------------- #

@torch.no_grad()
def embed_dataset(model, dataset: Sequence[Data], device, batch_size: int = 256) -> np.ndarray:
    """Graph embeddings for a reference set (used for the distance feature)."""
    if not hasattr(model, "node_embeddings"):
        return np.zeros((len(dataset), 1), dtype=np.float32)
    from torch_geometric.nn import global_mean_pool

    model.eval()
    out = []
    for batch in DataLoader(list(dataset), batch_size=batch_size):
        batch = batch.to(device)
        h = model.node_embeddings(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        out.append(global_mean_pool(h, batch.batch).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), dtype=np.float32)


def model_features(
    model,
    data: Data,
    device,
    task_type: str,
    train_embeddings: Optional[np.ndarray] = None,
    mc_samples: int = 20,
) -> Dict[str, float]:
    x, edge_index, edge_attr, batch = _single_graph_inputs(data, device)

    model.eval()
    with torch.no_grad():
        logit = float(model(x, edge_index, edge_attr, batch).view(-1)[0])

    feats: Dict[str, float] = {"model_output": logit}
    if task_type == "classification":
        prob = 1.0 / (1.0 + np.exp(-logit))
        feats["model_prob"] = float(prob)
        feats["model_margin"] = float(abs(prob - 0.5))
        feats["model_entropy"] = float(
            -(prob * np.log(prob + 1e-9) + (1 - prob) * np.log(1 - prob + 1e-9))
        )
    else:
        feats["model_prob"] = float("nan")
        feats["model_margin"] = float(abs(logit))
        feats["model_entropy"] = float("nan")

    # MC-dropout spread: how stable is this prediction under the model's own noise
    enable_mc_dropout(model)
    with torch.no_grad():
        samples = [float(model(x, edge_index, edge_attr, batch).view(-1)[0]) for _ in range(mc_samples)]
    model.eval()
    feats["model_mc_std"] = float(np.std(samples))

    # input-gradient norm: local sensitivity, a cheap proxy for how sharply the
    # decision depends on this particular molecule
    x_grad = x.detach().clone().requires_grad_(True)
    out = model(x_grad, edge_index, edge_attr, batch)
    out.sum().backward()
    feats["model_grad_norm"] = float(x_grad.grad.norm())
    model.zero_grad(set_to_none=True)

    # distance to the training manifold - applicability domain in embedding space
    if train_embeddings is not None and train_embeddings.shape[1] > 1 and hasattr(model, "node_embeddings"):
        from torch_geometric.nn import global_mean_pool

        with torch.no_grad():
            h = model.node_embeddings(x, edge_index, edge_attr, batch)
            emb = global_mean_pool(h, batch).cpu().numpy()
        dists = np.linalg.norm(train_embeddings - emb, axis=1)
        nearest = np.sort(dists)[:10]
        feats["model_knn_dist_1"] = float(nearest[0])
        feats["model_knn_dist_10"] = float(nearest.mean())
        feats["model_embedding_norm"] = float(np.linalg.norm(emb))
    return feats


# --------------------------------------------------------------------------- #
# 3. attribution shape
# --------------------------------------------------------------------------- #

def attribution_features(attribution: np.ndarray) -> Dict[str, float]:
    a = np.asarray(attribution, dtype=np.float64).ravel()
    if a.size == 0 or not np.all(np.isfinite(a)):
        return {}
    mag = np.abs(a)
    total = mag.sum() + 1e-12
    p = mag / total
    order = np.sort(mag)[::-1]
    n = a.size
    k = max(1, int(round(0.2 * n)))

    # Gini over attribution magnitude: is the explanation concentrated on a few
    # atoms or smeared across the molecule? Smeared explanations are the ones
    # that score well on null-explanation benchmarks while saying nothing.
    sorted_mag = np.sort(mag)
    index = np.arange(1, n + 1)
    gini = float((2 * index - n - 1).dot(sorted_mag) / (n * sorted_mag.sum() + 1e-12))

    return {
        "attr_entropy": float(-(p * np.log(p + 1e-12)).sum() / np.log(n + 1e-12)) if n > 1 else 0.0,
        "attr_gini": gini,
        "attr_top1_share": float(order[0] / total),
        "attr_topk_share": float(order[:k].sum() / total),
        "attr_std": float(a.std()),
        "attr_mean_abs": float(mag.mean()),
        "attr_max_abs": float(mag.max()),
        "attr_positive_frac": float((a > 0).mean()),
        "attr_sign_balance": float(abs(a[a > 0].sum() + a[a < 0].sum()) / total),
        "attr_range_ratio": float((order[0] - order[-1]) / (order[0] + 1e-12)),
    }


def assemble_feature_row(
    smiles: str,
    attribution: np.ndarray,
    model_feats: Dict[str, float],
    explainer: str,
    architecture: str,
    dataset: str,
) -> Dict[str, float | str]:
    row: Dict[str, float | str] = {
        "smiles": smiles,
        "explainer": explainer,
        "architecture": architecture,
        "dataset": dataset,
    }
    row.update(molecule_features(smiles))
    row.update(model_feats)
    row.update(attribution_features(attribution))
    return row


FEATURE_PREFIXES = ("mol_", "model_", "attr_")


def feature_columns(columns: Sequence[str]) -> List[str]:
    return [c for c in columns if c.startswith(FEATURE_PREFIXES)]
