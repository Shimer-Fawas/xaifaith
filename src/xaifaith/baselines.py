"""Masking and attribution baselines, chosen per process by environment variable.

    XAIFAITH_MASK_BASELINE   zeros | mean    what the deletion metric masks atoms TO
    XAIFAITH_IG_BASELINE     zeros | mean    what integrated gradients integrates FROM
    XAIFAITH_MEAN_VECTOR     path            required when either is "mean"

Both default to "zeros", reproducing every number in runs/v1 and runs/seed*.

They are deliberately independent. When both are "zeros" the metric's masking
path and IG's integration path are the SAME geometric path: features moving
between the real vector and the origin. That alignment plausibly explains why IG
outranks the near-circular occlusion control. Varying the two separately is how
we find out.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_VALID = ("zeros", "mean")
_mean: Optional[torch.Tensor] = None
_loaded_from = ""
_announced = False


def _name(var: str) -> str:
    v = os.environ.get(var, "zeros").strip().lower()
    if v not in _VALID:
        raise ValueError(f"{var} must be one of {_VALID}, got {v!r}")
    return v


def mask_baseline_name() -> str:
    return _name("XAIFAITH_MASK_BASELINE")


def ig_baseline_name() -> str:
    return _name("XAIFAITH_IG_BASELINE")


def _load_mean() -> torch.Tensor:
    global _mean, _loaded_from
    if _mean is not None:
        return _mean
    path = os.environ.get("XAIFAITH_MEAN_VECTOR", "").strip()
    if not path:
        raise RuntimeError(
            "a 'mean' baseline was requested but XAIFAITH_MEAN_VECTOR is not set. "
            "Build the vector with make_mean_baseline.py and point this at the .pt file."
        )
    blob = torch.load(path, map_location="cpu", weights_only=False)
    _mean = blob["vector"].float()
    _loaded_from = f"{path} ({blob['n_mols']} mols, {blob['n_atoms']} atoms)"
    return _mean


def _announce() -> None:
    global _announced
    if _announced:
        return
    _announced = True
    extra = f" mean_vector={_loaded_from}" if _loaded_from else ""
    print(
        f"[baselines] mask={mask_baseline_name()} ig={ig_baseline_name()}{extra}",
        file=sys.stderr, flush=True,
    )


def _expand(vec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if vec.numel() != x.shape[1]:
        raise RuntimeError(
            f"mean baseline has {vec.numel()} features but inputs have {x.shape[1]}"
        )
    return vec.to(x.device, x.dtype).unsqueeze(0).expand(x.shape[0], -1)


def mask_baseline_for(x: torch.Tensor) -> Optional[torch.Tensor]:
    """None means all-zeros, so the default path is unchanged."""
    name = mask_baseline_name()
    if name == "zeros":
        _announce()
        return None
    vec = _load_mean()
    _announce()
    return _expand(vec, x)


def ig_baseline_for(x: torch.Tensor) -> torch.Tensor:
    name = ig_baseline_name()
    if name == "zeros":
        _announce()
        return torch.zeros_like(x)
    vec = _load_mean()
    _announce()
    return _expand(vec, x)
