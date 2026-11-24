"""B-XAIC: ground-truth explanations, for measuring plausibility beside faithfulness.

B-XAIC (Proszewska, Danel & Rymarczyk, NeurIPS 2026) is 50K ChEMBL-35 molecules
with 7 binary tasks and, for every positive, the atoms that determine the label.
Negatives carry a *null explanation*: nothing should stand out.

Why it matters here: our faithfulness metric asks whether an attribution matches
the MODEL. B-XAIC asks whether it matches the CHEMISTRY. The premise of the
project is that those come apart, and this module measures both on the same
molecule, model and attribution.

Two things about the release decide how this is written.

ATOM ORDER. The explanation atoms live in `explanations.sdf` as index lists
("pains = '19,20,21'"), and those indices refer to the SDF molecule's own atom
ordering. Featurising from the CSV SMILES instead would apply them to a
different ordering and produce plausible-looking, silently wrong numbers. So the
graph is built from the SDF molecule, and the indices align by construction.

RECONSTRUCTION IS A CHECK, NOT A SOURCE. Every task can also be re-derived with
RDKit, and comparing the two caught real drift: against the released labels, the
rules below agree exactly for B, P, X, rings-count and rings-max, but only 0.854
for indole (the SMARTS demands [nH] and misses N-substituted indoles) and 0.889
for PAINS (RDKit's catalog is not the Baell-Holloway variant they used). The
released labels and masks are therefore authoritative; `ground_truth` stays as a
cross-check and for molecules outside the release.

Download: https://huggingface.co/datasets/mproszewska/B-XAIC
    hf download mproszewska/B-XAIC --repo-type dataset --local-dir data/bxaic
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams
from torch_geometric.data import Data

from .data import SplitIndices
from .featurize import BOND_DIM, atom_features, bond_features

RDLogger.DisableLog("rdApp.*")

HF_URL = "https://huggingface.co/datasets/mproszewska/B-XAIC"
CSV_NAME = "data.csv"
SDF_NAME = "explanations.sdf"

# our task key -> (SDF property holding the atom indices, CSV label column)
TASK_SOURCES: Dict[str, Tuple[str, str]] = {
    "b": ("B", "B"),
    "p": ("P", "P"),
    "x": ("X", "X"),
    "indole": ("indole", "indole"),
    "pains": ("pains", "PAINS"),
    "rings_count": ("rings", "rings-count"),
    "rings_max": ("largest_rings", "rings-max"),
}

# Tasks whose ground truth is a scattered handful of atoms rather than a
# connected motif; their plausibility behaves differently and the analysis
# reports the two groups apart.
ATOM_LEVEL_TASKS = frozenset({"b", "p", "x"})

INDOLE = Chem.MolFromSmarts("c1ccc2[nH]ccc2c1")
HALOGENS = frozenset({"F", "Cl", "Br", "I"})
_PAINS_CATALOG: Optional[FilterCatalog.FilterCatalog] = None


# --------------------------------------------------------------------------- #
# reconstruction (cross-check only - see module docstring)
# --------------------------------------------------------------------------- #

def _pains_catalog() -> FilterCatalog.FilterCatalog:
    global _PAINS_CATALOG
    if _PAINS_CATALOG is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(params)
    return _PAINS_CATALOG


def _by_element(symbols: frozenset) -> Callable:
    def fn(mol: Chem.Mol) -> Tuple[int, np.ndarray]:
        mask = np.array([a.GetSymbol() in symbols for a in mol.GetAtoms()], dtype=bool)
        return int(mask.any()), mask
    return fn


def _indole(mol: Chem.Mol) -> Tuple[int, np.ndarray]:
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for match in mol.GetSubstructMatches(INDOLE):
        mask[list(match)] = True
    return int(mask.any()), mask


def _pains(mol: Chem.Mol) -> Tuple[int, np.ndarray]:
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    catalog = _pains_catalog()
    if not catalog.HasMatch(mol):
        return 0, mask
    for entry in catalog.GetMatches(mol):
        for filter_match in entry.GetFilterMatches(mol):
            mask[[mol_atom for _, mol_atom in filter_match.atomPairs]] = True
    return int(mask.any()), mask


def _rings_count(mol: Chem.Mol, threshold: int = 4) -> Tuple[int, np.ndarray]:
    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]  # keep mol referenced
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    if len(rings) <= threshold:
        return 0, mask
    for ring in rings:
        mask[ring] = True
    return 1, mask


def _rings_max(mol: Chem.Mol, threshold: int = 6) -> Tuple[int, np.ndarray]:
    rings = [list(r) for r in Chem.GetSymmSSSR(mol)]
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    big = [r for r in rings if len(r) > threshold]
    if not big:
        return 0, mask
    for ring in big:
        mask[ring] = True
    return 1, mask


TASKS: Dict[str, Callable[[Chem.Mol], Tuple[int, np.ndarray]]] = {
    "b": _by_element(frozenset({"B"})),
    "p": _by_element(frozenset({"P"})),
    "x": _by_element(HALOGENS),
    "indole": _indole,
    "pains": _pains,
    "rings_count": _rings_count,
    "rings_max": _rings_max,
}

# Agreement with the released labels, measured on 3000 molecules. Anything below
# 1.0 means the rule here is not theirs; those tasks must use the release.
RECONSTRUCTION_AGREEMENT = {
    "b": 1.000, "p": 1.000, "x": 1.000,
    "rings_count": 1.000, "rings_max": 0.99998,
    "indole": 0.854, "pains": 0.889,
}
TRUSTWORTHY_RECONSTRUCTION = frozenset(
    t for t, a in RECONSTRUCTION_AGREEMENT.items() if a > 0.999
)


def ground_truth(smiles: str, task: str) -> Optional[Tuple[int, np.ndarray]]:
    """Re-derive a label and atom mask from SMILES. Cross-check; see the docstring."""
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return TASKS[task](mol)


# --------------------------------------------------------------------------- #
# loading the release
# --------------------------------------------------------------------------- #

@dataclass
class BXAICTask:
    name: str
    dataset: List[Data]
    labels: np.ndarray
    masks: List[np.ndarray]
    split: SplitIndices
    alignment: float = float("nan")   # CSV/SDF row correspondence, sampled
    agreement: float = float("nan")   # released labels vs reconstruction
    skipped: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if self.labels.size else float("nan")


def _mol_to_data(mol: Chem.Mol, y: float) -> Optional[Data]:
    """PyG graph from an RDKit mol, keeping its atom order (the masks depend on it)."""
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
    if not src:
        src, dst, eattr = [0], [0], [[0.0] * BOND_DIM]

    data = Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(eattr, dtype=torch.float),
    )
    data.smiles = Chem.MolToSmiles(mol)
    data.y = torch.tensor([[float(y)]], dtype=torch.float)
    return data


def _parse_indices(raw: Optional[str], n_atoms: int) -> np.ndarray:
    mask = np.zeros(n_atoms, dtype=bool)
    if not raw:
        return mask
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        if 0 <= idx < n_atoms:
            mask[idx] = True
    return mask


def _paths(data_dir: str) -> Tuple[str, str]:
    csv_path = os.path.join(data_dir, CSV_NAME)
    sdf_path = os.path.join(data_dir, SDF_NAME)
    for path in (csv_path, sdf_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Download B-XAIC into {data_dir!r}:\n"
                f"  hf download mproszewska/B-XAIC --repo-type dataset --local-dir {data_dir}"
            )
    return csv_path, sdf_path


def _select_rows(
    labels: np.ndarray, max_molecules: Optional[int], seed: int
) -> np.ndarray:
    """Class-balanced subsample, so a rare task keeps its positives."""
    if not max_molecules or len(labels) <= max_molecules:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(labels == 1)
    neg = np.flatnonzero(labels == 0)
    n_pos = min(len(pos), max(1, max_molecules // 2))
    n_neg = min(len(neg), max_molecules - n_pos)
    return np.sort(np.concatenate([
        rng.choice(pos, n_pos, replace=False),
        rng.choice(neg, n_neg, replace=False),
    ]))


def load_bxaic(
    data_dir: str,
    task: str,
    split_index: int = 0,
    max_molecules: Optional[int] = None,
    seed: int = 0,
    alignment_sample: int = 300,
    cross_check: bool = True,
) -> BXAICTask:
    """Load one task: graphs from the SDF, labels and splits from the CSV.

    The graph comes from the SDF molecule because the ground-truth atom indices
    are in that molecule's atom order. The CSV supplies the label and the
    predefined split. Row i of the CSV is assumed to be record i of the SDF, and
    that assumption is verified on a sample of canonical SMILES rather than
    trusted - `alignment` on the result reports the agreement.
    """
    if task not in TASK_SOURCES:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASK_SOURCES)}")
    sdf_prop, csv_column = TASK_SOURCES[task]
    csv_path, sdf_path = _paths(data_dir)

    df = pd.read_csv(csv_path)
    split_column = f"split_{split_index}"
    for column in (csv_column, split_column, "smiles"):
        if column not in df.columns:
            raise KeyError(f"{csv_path} has no column {column!r}; found {list(df.columns)}")

    supplier = Chem.SDMolSupplier(sdf_path, sanitize=True, removeHs=True)
    notes: List[str] = []
    if len(supplier) != len(df):
        notes.append(f"CSV has {len(df)} rows, SDF has {len(supplier)} records")
        warnings.warn(notes[-1], RuntimeWarning)

    labels_all = pd.to_numeric(df[csv_column], errors="coerce").to_numpy()
    keep = _select_rows(np.nan_to_num(labels_all, nan=-1), max_molecules, seed)

    # Verify the row correspondence before relying on it.
    rng = np.random.default_rng(seed)
    probe = keep if len(keep) <= alignment_sample else rng.choice(keep, alignment_sample, replace=False)
    matches, compared = 0, 0
    for i in probe:
        mol = supplier[int(i)]
        if mol is None:
            continue
        reference = Chem.MolFromSmiles(str(df["smiles"].iloc[int(i)]))
        if reference is None:
            continue
        compared += 1
        matches += int(Chem.MolToSmiles(mol) == Chem.MolToSmiles(reference))
    alignment = matches / compared if compared else float("nan")
    if compared and alignment < 0.99:
        notes.append(f"CSV/SDF alignment only {alignment:.3f} - indices may not correspond")
        warnings.warn(notes[-1], RuntimeWarning)

    dataset: List[Data] = []
    labels: List[float] = []
    masks: List[np.ndarray] = []
    assignments: List[str] = []
    agree, agree_n, skipped = 0, 0, 0

    for i in keep:
        i = int(i)
        label = labels_all[i]
        if np.isnan(label):
            skipped += 1
            continue
        mol = supplier[i]
        if mol is None:
            skipped += 1
            continue
        data = _mol_to_data(mol, float(label))
        if data is None:
            skipped += 1
            continue

        mask = _parse_indices(
            mol.GetProp(sdf_prop) if mol.HasProp(sdf_prop) else "", mol.GetNumAtoms()
        )
        if cross_check and task in TRUSTWORTHY_RECONSTRUCTION:
            rebuilt = TASKS[task](mol)
            agree_n += 1
            agree += int(rebuilt[0] == int(label))

        dataset.append(data)
        labels.append(float(label))
        masks.append(mask)
        assignments.append(str(df[split_column].iloc[i]))

    assignments_arr = np.asarray(assignments)
    split = SplitIndices(
        train=sorted(np.flatnonzero(assignments_arr == "train").tolist()),
        val=sorted(np.flatnonzero(np.isin(assignments_arr, ("valid", "val"))).tolist()),
        test=sorted(np.flatnonzero(assignments_arr == "test").tolist()),
    )
    if not split.val or not split.test:
        notes.append(f"{split_column} gave train/val/test "
                     f"{len(split.train)}/{len(split.val)}/{len(split.test)}")
        warnings.warn(notes[-1], RuntimeWarning)

    return BXAICTask(
        name=task,
        dataset=dataset,
        labels=np.asarray(labels, dtype=float),
        masks=masks,
        split=split,
        alignment=alignment,
        agreement=(agree / agree_n) if agree_n else float("nan"),
        skipped=skipped,
        notes=notes,
    )


def summarise_tasks(
    data_dir: str, max_molecules: Optional[int] = 2000, split_index: int = 0
) -> pd.DataFrame:
    """One row per task: size, balance, CSV/SDF alignment, mask statistics."""
    rows = []
    for task in TASK_SOURCES:
        try:
            loaded = load_bxaic(data_dir, task, split_index=split_index,
                                max_molecules=max_molecules)
        except Exception as exc:  # noqa: BLE001 - reported in the table
            rows.append({"task": task, "error": f"{type(exc).__name__}: {exc}"[:60]})
            continue
        positive_masks = [m for m, y in zip(loaded.masks, loaded.labels) if y > 0.5]
        empty_positive = sum(1 for m in positive_masks if not m.any())
        rows.append({
            "task": task,
            "n": len(loaded.dataset),
            "pos_rate": round(loaded.positive_rate, 3),
            "train/val/test": f"{len(loaded.split.train)}/{len(loaded.split.val)}/{len(loaded.split.test)}",
            "mean_truth_atoms": round(float(np.mean([m.sum() for m in positive_masks])), 2)
            if positive_masks else 0.0,
            "positives_wo_mask": empty_positive,
            "csv_sdf_alignment": round(loaded.alignment, 4),
            "recon_agreement": round(loaded.agreement, 4) if not np.isnan(loaded.agreement) else None,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# plausibility metrics (B-XAIC's own, per molecule)
# --------------------------------------------------------------------------- #

def subgraph_auroc(attribution: np.ndarray, mask: np.ndarray, signed: bool = False) -> float:
    """SE: do the ground-truth atoms get higher importance than the rest?

    Undefined when every atom is relevant or none is.
    """
    if mask.size != attribution.size or mask.all() or not mask.any():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    score = attribution if signed else np.abs(attribution)
    if not np.all(np.isfinite(score)):
        return float("nan")
    try:
        return float(roc_auc_score(mask.astype(int), score))
    except ValueError:
        return float("nan")


def null_explanation_score(attribution: np.ndarray, whisker: float = 1.5) -> float:
    """NE: for a negative, nothing should stand out. 1 if no IQR outliers.

    This is the metric a uniform attribution games - saying nothing scores
    perfectly - so it is reported beside SE, never averaged into it.
    """
    values = np.abs(np.asarray(attribution, dtype=float))
    if values.size < 4 or not np.all(np.isfinite(values)):
        return float("nan")
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        # A flat bulk with one spike also has q1 == q3, and the interval test
        # would pass it. Score 1 only when the attribution really is uniform.
        return float(bool(np.allclose(values, values[0])))
    outliers = (values > q3 + whisker * iqr) | (values < q1 - whisker * iqr)
    return float(not outliers.any())


def topk_recovery(attribution: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of ground-truth atoms inside the top-k attributions, k = |truth|."""
    if mask.size != attribution.size or not mask.any():
        return float("nan")
    k = int(mask.sum())
    top = set(np.argsort(-np.abs(attribution))[:k].tolist())
    return float(len(top & set(np.flatnonzero(mask).tolist())) / k)


def plausibility(attribution: np.ndarray, mask: np.ndarray, label: float) -> Dict[str, float]:
    """Every plausibility measure for one molecule."""
    if label > 0.5:
        return {
            "se_auroc": subgraph_auroc(attribution, mask),
            "se_auroc_signed": subgraph_auroc(attribution, mask, signed=True),
            "topk_recovery": topk_recovery(attribution, mask),
            "ne_score": float("nan"),
            "n_truth_atoms": float(mask.sum()),
        }
    return {
        "se_auroc": float("nan"),
        "se_auroc_signed": float("nan"),
        "topk_recovery": float("nan"),
        "ne_score": null_explanation_score(attribution),
        "n_truth_atoms": 0.0,
    }
