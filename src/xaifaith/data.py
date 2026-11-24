"""Dataset loading and scaffold splitting.

MoleculeNet is downloaded via PyG purely for the SMILES + labels; the graphs
themselves are rebuilt by featurize.py so that x is float-valued.
"""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from torch_geometric.datasets import MoleculeNet

from .featurize import featurize_many

# name -> (PyG MoleculeNet key, task index within y, task type)
DATASETS: Dict[str, Tuple[str, int, str]] = {
    "bace": ("BACE", 0, "classification"),
    "bbbp": ("BBBP", 0, "classification"),
    "clintox": ("ClinTox", 0, "classification"),
    # Tox21 task 0 = NR-AR; SIDER task 0 = hepatobiliary disorders
    "tox21_nr_ar": ("Tox21", 0, "classification"),
    "tox21_sr_mmp": ("Tox21", 9, "classification"),
    "sider_hepato": ("SIDER", 0, "classification"),
    "hiv": ("HIV", 0, "classification"),
    "esol": ("ESOL", 0, "regression"),
    "lipo": ("Lipo", 0, "regression"),
}


@dataclass
class SplitIndices:
    train: List[int]
    val: List[int]
    test: List[int]

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(asdict(self), fh)

    @staticmethod
    def load(path: str) -> "SplitIndices":
        with open(path) as fh:
            return SplitIndices(**json.load(fh))


def _scaffold(smiles: str) -> str:
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
    except Exception:
        return ""


def _is_classification(labels: Optional[Sequence[float]]) -> bool:
    return labels is not None and set(float(v) for v in labels) <= {0.0, 1.0}


def _all_splits_have_both_classes(split: "SplitIndices", labels: Sequence[float]) -> bool:
    for part in (split.train, split.val, split.test):
        if not part:
            return False
        values = {float(labels[i]) for i in part}
        if len(values) < 2:
            return False
    return True


def _assign(index_sets: List[List[int]], n: int, frac: Tuple[float, float, float]) -> SplitIndices:
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    train: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for group in index_sets:
        if len(train) + len(group) <= n_train:
            train += group
        elif len(val) + len(group) <= n_val:
            val += group
        else:
            test += group
    return SplitIndices(sorted(train), sorted(val), sorted(test))


def scaffold_split(
    smiles: Sequence[str],
    labels: Optional[Sequence[float]] = None,
    frac: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    max_retries: int = 20,
) -> SplitIndices:
    """Balanced Bemis-Murcko scaffold split.

    Scaffold splitting matters here beyond the usual reason. A random split lets
    near-duplicate analogues straddle train and test, which inflates both accuracy
    and apparent explanation quality (the AVE-bias problem). Any faithfulness
    number computed on a random split is measuring memorisation.

    Balanced, though, not size-ordered. Sorting groups by size sends the rarest
    scaffolds to the test set, and on a congeneric series (BACE is the obvious
    case) those tail scaffolds are heavily skewed to one class, which produces
    single-class splits and undefined AUROC. So oversized groups still go to
    train - that is what prevents leakage - and the rest are shuffled. If a split
    still ends up single-class we retry on another seed before giving up.
    """
    groups: Dict[str, List[int]] = {}
    for i, smi in enumerate(smiles):
        groups.setdefault(_scaffold(smi), []).append(i)

    n = len(smiles)
    n_val = int(frac[1] * n)
    n_test = n - int(frac[0] * n) - n_val
    classification = _is_classification(labels)

    big = [g for g in groups.values() if len(g) > max(n_val, n_test) / 2]
    small = [g for g in groups.values() if len(g) <= max(n_val, n_test) / 2]

    split = None
    for attempt in range(max_retries):
        rng = np.random.default_rng(seed + attempt)
        shuffled = [small[i] for i in rng.permutation(len(small))]
        # Oversized groups normally go to train first, which keeps the held-out
        # sets scaffold-diverse. But when there are few small groups the shuffle
        # has nothing to act on and every seed returns the same split - a silent
        # non-replication. There, shuffle the big groups too: whole groups still
        # move together, so no scaffold straddles and nothing leaks.
        ordered_big = big if len(small) >= 10 else [big[i] for i in rng.permutation(len(big))]
        split = _assign(ordered_big + shuffled, n, frac)
        if not split.val or not split.test:
            continue
        if not classification or _all_splits_have_both_classes(split, labels):
            if attempt:
                warnings.warn(
                    f"scaffold split needed {attempt + 1} seeds to give every split both classes",
                    RuntimeWarning,
                )
            return split

    # Very few distinct scaffolds (synthetic sets, tight congeneric series).
    # Fall back loudly rather than returning a split that silently breaks
    # early stopping or AUROC.
    warnings.warn(
        f"could not build a usable scaffold split ({len(groups)} scaffolds for {n} molecules) "
        "after {max_retries} attempts; falling back to a random split. Faithfulness numbers "
        "from this split are NOT scaffold-held-out.".format(max_retries=max_retries),
        RuntimeWarning,
    )
    return random_split(n, frac, seed=seed)


def random_split(
    n: int, frac: Tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 0
) -> SplitIndices:
    """Deterministic random split. Smoke tests only - see scaffold_split."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train, n_val = int(frac[0] * n), int(frac[1] * n)
    return SplitIndices(
        sorted(int(i) for i in order[:n_train]),
        sorted(int(i) for i in order[n_train : n_train + n_val]),
        sorted(int(i) for i in order[n_train + n_val :]),
    )


def load_dataset(name: str, root: str = "data") -> Tuple[List[Data], str]:
    """Returns (list of PyG Data with float features, task type)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(DATASETS)}")
    pyg_name, task_idx, task_type = DATASETS[name]

    raw = MoleculeNet(root=os.path.join(root, "moleculenet"), name=pyg_name)

    smiles: List[str] = []
    labels: List[float] = []
    for item in raw:
        y = item.y.view(-1)
        if task_idx >= y.numel():
            continue
        value = float(y[task_idx])
        if np.isnan(value):
            continue
        if Chem.MolFromSmiles(item.smiles) is None:
            continue
        smiles.append(item.smiles)
        labels.append(value)

    dataset = featurize_many(smiles, labels)
    return dataset, task_type


def get_split(
    dataset: Sequence[Data], cache_path: str | None = None, seed: int = 0
) -> SplitIndices:
    """Scaffold split for a dataset, cached.

    `seed` reaches the splitter, so a different seed gives a genuinely different
    split rather than only a different model initialisation. Replication runs
    depend on that: without it, repeated "seeds" re-use one split and the spread
    they report is model-init noise, not split noise.
    """
    if cache_path and os.path.exists(cache_path):
        return SplitIndices.load(cache_path)
    labels = [float(d.y.view(-1)[0]) for d in dataset] if getattr(dataset[0], "y", None) is not None else None
    split = scaffold_split([d.smiles for d in dataset], labels=labels, seed=seed)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        split.save(cache_path)
    return split


def subset(dataset: Sequence[Data], idx: Sequence[int]) -> List[Data]:
    return [dataset[i] for i in idx]
