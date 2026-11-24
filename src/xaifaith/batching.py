"""Batched graph replication helpers.

Everything downstream (integrated gradients, deletion curves, the greedy oracle)
needs K perturbed copies of ONE molecule evaluated in a single forward pass.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch_geometric.data import Data

from .baselines import mask_baseline_for


def replicate(data: Data, k: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (edge_index, edge_attr, batch) for k disjoint copies of `data`."""
    n = int(data.num_nodes)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)

    offsets = (torch.arange(k, device=device) * n).view(k, 1, 1)
    ei = (edge_index.unsqueeze(0) + offsets).permute(1, 0, 2).reshape(2, -1)
    ea = edge_attr.repeat(k, 1)
    batch = torch.arange(k, device=device).repeat_interleave(n)
    return ei, ea, batch


def masked_node_features(
    x: torch.Tensor, keep_masks: torch.Tensor, baseline: torch.Tensor | None = None
) -> torch.Tensor:
    """Build stacked node features for K masked copies.

    keep_masks: bool tensor [K, n]. True = keep the real atom features,
    False = replace with the baseline (all-zeros by default, which in our
    featurisation means "no information about this atom").
    """
    k, n = keep_masks.shape
    if baseline is None:
        # None now means "use the process-wide masking baseline". That is
        # all-zeros unless XAIFAITH_MASK_BASELINE says otherwise, so every
        # existing call site keeps its exact previous behaviour.
        baseline = mask_baseline_for(x)
    base = torch.zeros_like(x) if baseline is None else baseline
    x_rep = x.unsqueeze(0).expand(k, n, x.shape[1])
    base_rep = base.unsqueeze(0).expand(k, n, x.shape[1])
    mask = keep_masks.unsqueeze(-1).to(x.dtype)
    return (mask * x_rep + (1.0 - mask) * base_rep).reshape(k * n, x.shape[1])


@torch.no_grad()
def forward_masked(
    model,
    data: Data,
    keep_masks: torch.Tensor,
    device: torch.device,
    chunk: int = 512,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar model output for each masked copy. Returns [K]."""
    x = data.x.to(device)
    keep_masks = keep_masks.to(device)
    outs = []
    for start in range(0, keep_masks.shape[0], chunk):
        block = keep_masks[start : start + chunk]
        k = block.shape[0]
        ei, ea, batch = replicate(data, k, device)
        x_rep = masked_node_features(x, block, baseline)
        out = model(x_rep, ei, ea, batch)
        outs.append(out.view(-1))
    return torch.cat(outs)
