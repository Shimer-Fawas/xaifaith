"""Smoke test on a tiny synthetic task with a known ground-truth substructure.

Molecules containing a nitro group are labelled positive. A trained model should
learn that, and a good explainer should put its mass on the nitro atoms. This
runs in about a minute on CPU and checks the whole pipeline end to end before
you spend GPU hours on it.

    python -m xaifaith.smoke_test
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem

from .data import random_split
from .explain import compute_attribution
from .faithfulness import FaithfulnessConfig, compute_references, faithfulness_for_attribution
from .featurize import featurize_many
from .models import build_model
from .train import TrainConfig, train_model

# {0} is the discriminative position, {1} a noise substituent.
CORES = [
    "{1}c1ccc(cc1){0}",
    "{1}c1ccc(cn1){0}",
    "{1}c1ccc(s1){0}",
    "{1}c1ccc(o1){0}",
    "{1}C1CCC(CC1){0}",
    "{1}C1CCC(NC1){0}",
    "{1}c1ccc2ccccc2c1{0}",
    "{1}c1ccc(cc1)C{0}",
    "{1}c1cccc(c1){0}",
    "{1}C1CCCC(C1){0}",
    "{1}c1ccc(cc1)CC{0}",
    "{1}C1CCOC(C1){0}",
]
POSITIVE_GROUP = "[N+](=O)[O-]"
NEGATIVE_GROUPS = ["C", "CC", "O", "OC", "F", "Cl", "CCC", "N", "C(C)C", "S(C)"]
NOISE_GROUPS = ["", "C", "CC", "O", "F", "Cl", "N", "CO", "CCO", "OC", "CN", "CCC"]


def _unique(raw_smiles, seen):
    out = []
    for raw in raw_smiles:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def build_synthetic(seed: int = 0):
    """Molecules containing a nitro group are positive. Everything else is noise.

    Enumerated rather than sampled, then class-balanced: the positive class has a
    bounded number of distinct members (cores x noise groups), so random sampling
    with dedupe silently produces a 10%-positive set and a tiny test-set positive
    count, which makes every number downstream noisy.
    """
    rng = np.random.default_rng(seed)
    seen: set[str] = set()

    positives = _unique(
        [core.format(POSITIVE_GROUP, noise) for core in CORES for noise in NOISE_GROUPS], seen
    )
    negatives = _unique(
        [
            core.format(group, noise)
            for core in CORES
            for noise in NOISE_GROUPS
            for group in NEGATIVE_GROUPS
        ],
        seen,
    )
    # balance so the test split has a usable number of each class
    keep = min(len(positives), len(negatives))
    negatives = [negatives[i] for i in rng.permutation(len(negatives))[:keep]]
    positives = positives[:keep]

    smiles = positives + negatives
    labels = [1.0] * len(positives) + [0.0] * len(negatives)
    return smiles, labels


def nitro_atom_mask(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
    mask = np.zeros(mol.GetNumAtoms(), dtype=bool)
    for match in mol.GetSubstructMatches(pattern):
        mask[list(match)] = True
    return mask


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles, labels = build_synthetic()
    dataset = featurize_many(smiles, labels)
    # Random split on purpose: this synthetic set has only a handful of distinct
    # Murcko scaffolds, so a scaffold split would be degenerate. Real runs use
    # scaffold splits throughout (see data.py).
    split = random_split(len(dataset))
    print(f"{len(dataset)} molecules, train/val/test "
          f"{len(split.train)}/{len(split.val)}/{len(split.test)}")

    cfg = TrainConfig(architecture="gin", epochs=40, patience=10)
    metrics = train_model(dataset, split, "classification", cfg, "runs/smoke/model", device=str(device))
    print(f"test AUROC {metrics['test_score']:.3f}  (should be well above 0.9)")
    print(f"split class balance: "
          f"train {np.mean([float(dataset[i].y) for i in split.train]):.2f} / "
          f"val {np.mean([float(dataset[i].y) for i in split.val]):.2f} / "
          f"test {np.mean([float(dataset[i].y) for i in split.test]):.2f}")

    from .train import load_model

    model, task_type = load_model("runs/smoke/model", device=str(device))
    fcfg = FaithfulnessConfig(n_random_orders=8, max_curve_points=16)

    results = {}
    # faithfulness over the whole test split; nitro-recovery only where there
    # is a nitro group to recover
    held_out = [dataset[i] for i in split.test]
    for name in ("integrated_gradients", "gradient_input", "saliency", "random"):
        faiths, plausibilities = [], []
        for data in held_out:
            refs = compute_references(model, data, device, fcfg)
            if refs.degenerate:
                continue
            attribution = compute_attribution(name, model, data, device)
            scores = faithfulness_for_attribution(model, data, attribution, refs, device, fcfg)
            faiths.append(scores["faithfulness"])

            truth = nitro_atom_mask(data.smiles)
            if truth.any() and truth.size == attribution.size:
                k = int(truth.sum())
                top = set(np.argsort(-attribution)[:k])
                plausibilities.append(len(top & set(np.where(truth)[0])) / k)
        stderr = float(np.nanstd(faiths) / max(np.sqrt(len(faiths)), 1))
        results[name] = (float(np.nanmean(faiths)), stderr, float(np.mean(plausibilities)), len(faiths))

    print(f"\n explainer                faithfulness      nitro-recovery   (n={results['random'][3]})")
    for name, (faith, stderr, plaus, _) in results.items():
        print(f"  {name:>22}   {faith:+.3f} ± {stderr:.3f}      {plaus:.3f}")
    print("\nSanity checks: integrated_gradients should clearly beat random on both")
    print("columns. The random row is high-variance on this toy task - judge it by")
    print("whether 0.0 sits inside its error bar, not by its point estimate.")


if __name__ == "__main__":
    main()
