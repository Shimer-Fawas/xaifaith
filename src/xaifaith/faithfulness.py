"""Per-molecule faithfulness.

The label this project predicts.

Faithfulness here is faithfulness TO THE MODEL, not agreement with a chemist or
with a structural alert. An attribution is faithful for a given molecule if
masking the atoms it ranks highest collapses the model's output faster than
masking atoms in a random order.

The normalisation is the part that makes it per-molecule rather than per-method.
A raw deletion AUC is not comparable across molecules: a 12-atom fragment and a
40-atom molecule have different achievable ranges, and some molecules collapse
under almost any masking. So each molecule gets its own two reference points:

    random order   -> the floor (what you get by knowing nothing)
    greedy oracle  -> the ceiling (the best ordering that exists for this model
                      and this molecule, found by greedy search)

and faithfulness is where the attribution sits between them:

    F = (AUC_random - AUC_attribution) / (AUC_random - AUC_greedy)

F = 1 means the attribution is as good as the best ordering that exists for this
molecule. F = 0 means it is no better than random. F < 0 means it is worse than
random, which happens more often than the literature admits.
"""
from __future__ import annotations

import hashlib

import os

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from .batching import forward_masked

EPS = 1e-6

# numpy 2.0 renamed trapz -> trapezoid
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class FaithfulnessConfig:
    n_random_orders: int = 8
    max_curve_points: int = 20
    greedy_max_atoms: int = 60
    # Width of the beam used to search for the ceiling ordering. 1 is plain
    # greedy and reproduces every number in runs/v1 exactly; wider beams find
    # better orderings, which RAISES the ceiling and so LOWERS every F.
    beam_width: int = field(default_factory=lambda: int(os.environ.get("XAIFAITH_BEAM_WIDTH", "1")))
    comprehensiveness_k: float = 0.2
    min_dynamic_range: float = 1e-3
    chunk: int = 512


@dataclass
class MoleculeReferences:
    """Per-molecule quantities that do not depend on which explainer is used."""

    base_output: float
    empty_output: float
    denom: float
    auc_random: float
    auc_random_std: float
    auc_greedy: float
    greedy_exact: bool
    degenerate: bool
    n_atoms: int
    greedy_order: List[int] = field(default_factory=list)


def _masks_from_order(order: Sequence[int], n: int, n_points: int):
    """Boolean keep-masks for progressively deleting atoms in `order`."""
    cut_points = np.unique(np.linspace(0, n, min(n_points, n) + 1).round().astype(int))
    masks = torch.ones(len(cut_points), n, dtype=torch.bool)
    for row, cut in enumerate(cut_points):
        if cut > 0:
            masks[row, list(order[:cut])] = False
    return masks, cut_points / float(n)


def _normalised_auc(values: np.ndarray, fractions: np.ndarray, empty: float, denom: float) -> float:
    curve = (values - empty) / denom
    return float(_trapezoid(curve, fractions))


def _deletion_auc(
    model, data: Data, order: Sequence[int], device, refs_empty: float, denom: float, cfg: FaithfulnessConfig
) -> float:
    n = int(data.num_nodes)
    masks, fractions = _masks_from_order(order, n, cfg.max_curve_points)
    values = forward_masked(model, data, masks, device, chunk=cfg.chunk).cpu().numpy()
    return _normalised_auc(values, fractions, refs_empty, denom)


def _greedy_order(model, data: Data, device, denom: float, cfg: FaithfulnessConfig) -> tuple[List[int], bool]:
    """Greedy search for the atom ordering that collapses the output fastest.

    Exact for molecules up to `greedy_max_atoms`; beyond that we return the best
    of several random orders as a (weaker) ceiling and flag it.
    """
    n = int(data.num_nodes)
    if n > cfg.greedy_max_atoms:
        empty = _empty_value(model, data, device, cfg)
        best, best_auc = None, np.inf
        rng = np.random.default_rng(0)
        for _ in range(16):
            order = list(rng.permutation(n))
            auc = _deletion_auc(model, data, order, device, empty, denom, cfg)
            if auc < best_auc:
                best, best_auc = order, auc
        return list(best), False

    sign = 1.0 if denom > 0 else -1.0
    width = max(1, int(cfg.beam_width))

    # Beam search over deletion orderings. Each entry is (cumulative score, order).
    # The score accumulates the direction-adjusted output after each deletion, so
    # a low total means the output collapsed early. Width 1 is exactly the former
    # greedy loop.
    beam: List[tuple] = [(0.0, [])]
    greedy_order: List[int] = []
    for _ in range(n):
        plans, meta = [], []
        for bi, (score, chosen) in enumerate(beam):
            taken = set(chosen)
            for cand in range(n):
                if cand in taken:
                    continue
                plans.append(chosen + [cand])
                meta.append((bi, score))
        masks = torch.ones(len(plans), n, dtype=torch.bool)
        for row, sel in enumerate(plans):
            masks[row, sel] = False
        values = forward_masked(model, data, masks, device, chunk=cfg.chunk).cpu().numpy()
        scored = [(meta[r][1] + sign * float(values[r]), plans[r]) for r in range(len(plans))]
        scored.sort(key=lambda t: t[0])
        beam = scored[:width]
        if width > 1:
            # Sorting globally across every beam entry's extensions can prune the
            # width-1 path, which lets a wider beam return a WORSE ceiling than
            # plain greedy. Pinning the greedy continuation makes the ceiling
            # monotone non-increasing in beam width.
            cands = [s for s in scored if s[1][:-1] == greedy_order]
            if cands:
                g_best = min(cands, key=lambda t: t[0])
                if all(e[1] != g_best[1] for e in beam):
                    beam[-1] = g_best
                greedy_order = g_best[1]

    empty = _empty_value(model, data, device, cfg)
    best, best_auc = None, np.inf
    for _, order in beam:
        auc = _deletion_auc(model, data, order, device, empty, denom, cfg)
        if auc < best_auc:
            best, best_auc = order, auc
    return list(map(int, best)), True


def _empty_value(model, data: Data, device, cfg: FaithfulnessConfig) -> float:
    n = int(data.num_nodes)
    masks = torch.zeros(1, n, dtype=torch.bool)
    return float(forward_masked(model, data, masks, device, chunk=cfg.chunk)[0])


def compute_references(
    model, data: Data, device, cfg: Optional[FaithfulnessConfig] = None
) -> MoleculeReferences:
    cfg = cfg or FaithfulnessConfig()
    n = int(data.num_nodes)

    full = torch.ones(1, n, dtype=torch.bool)
    base = float(forward_masked(model, data, full, device, chunk=cfg.chunk)[0])
    empty = _empty_value(model, data, device, cfg)
    denom = base - empty

    degenerate = abs(denom) < cfg.min_dynamic_range
    if degenerate:
        # The model's output barely moves between the full molecule and nothing,
        # so no ordering can be distinguished. These molecules are dropped, and
        # the fraction dropped is itself a reportable number.
        return MoleculeReferences(base, empty, denom, np.nan, np.nan, np.nan, False, True, n)

    # hashlib, not builtin hash(): Python randomises string hashing per
    # process unless PYTHONHASHSEED is set, which made auc_random - and so
    # every per-molecule F - irreproducible across separate invocations.
    _seed = int.from_bytes(hashlib.sha256(data.smiles.encode()).digest()[:4], "little")
    rng = np.random.default_rng(_seed)
    random_aucs = [
        _deletion_auc(model, data, list(rng.permutation(n)), device, empty, denom, cfg)
        for _ in range(cfg.n_random_orders)
    ]
    greedy_order, exact = _greedy_order(model, data, device, denom, cfg)
    auc_greedy = _deletion_auc(model, data, greedy_order, device, empty, denom, cfg)

    return MoleculeReferences(
        base_output=base,
        empty_output=empty,
        denom=denom,
        auc_random=float(np.mean(random_aucs)),
        auc_random_std=float(np.std(random_aucs)),
        auc_greedy=float(auc_greedy),
        greedy_exact=exact,
        degenerate=False,
        n_atoms=n,
        greedy_order=list(map(int, greedy_order)),
    )


def faithfulness_for_attribution(
    model,
    data: Data,
    attribution: np.ndarray,
    refs: MoleculeReferences,
    device,
    cfg: Optional[FaithfulnessConfig] = None,
) -> Dict[str, float]:
    cfg = cfg or FaithfulnessConfig()
    n = int(data.num_nodes)

    if refs.degenerate or np.any(~np.isfinite(attribution)):
        return {
            "faithfulness": np.nan,
            "auc_attribution": np.nan,
            "comprehensiveness": np.nan,
            "sufficiency": np.nan,
            "beats_random": np.nan,
        }

    # Rank atoms in the direction of the model's own prediction. If the output
    # moves downward when the molecule is removed, positive attributions are the
    # drivers; otherwise negative ones are.
    direction = 1.0 if refs.denom > 0 else -1.0
    order = list(np.argsort(-direction * attribution))

    auc_attr = _deletion_auc(model, data, order, device, refs.empty_output, refs.denom, cfg)
    spread = refs.auc_random - refs.auc_greedy
    faith = (refs.auc_random - auc_attr) / (spread + EPS)
    faith = float(np.clip(faith, -1.0, 1.0))

    k = max(1, int(round(cfg.comprehensiveness_k * n)))
    comp_mask = torch.ones(1, n, dtype=torch.bool)
    comp_mask[0, order[:k]] = False
    suff_mask = torch.zeros(1, n, dtype=torch.bool)
    suff_mask[0, order[:k]] = True
    comp_value = float(forward_masked(model, data, comp_mask, device, chunk=cfg.chunk)[0])
    suff_value = float(forward_masked(model, data, suff_mask, device, chunk=cfg.chunk)[0])

    return {
        "faithfulness": faith,
        "auc_attribution": float(auc_attr),
        # fraction of the full-molecule signal destroyed by removing the top-k atoms
        "comprehensiveness": float((refs.base_output - comp_value) / refs.denom),
        # fraction of the signal the top-k atoms carry on their own
        "sufficiency": float((suff_value - refs.empty_output) / refs.denom),
        "beats_random": float(auc_attr < refs.auc_random),
    }
