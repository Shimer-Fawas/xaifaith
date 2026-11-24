"""GNN backbones.

Every model exposes the same forward signature

    forward(x, edge_index, edge_attr, batch) -> Tensor [num_graphs, 1]

with x as the FIRST positional argument. That is what lets gradient-based
attribution differentiate the output w.r.t. atom features without wrappers, and
what lets faithfulness.py swap in masked copies of x.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AttentiveFP, GATv2Conv, GCNConv, GINEConv, global_mean_pool

from .featurize import ATOM_DIM, BOND_DIM

ARCHITECTURES = ("gcn", "gin", "gat", "attentivefp")


class MessagePassingNet(nn.Module):
    """GCN / GIN / GAT with a shared skeleton and an exposed node embedding."""

    def __init__(
        self,
        kind: str = "gin",
        in_dim: int = ATOM_DIM,
        edge_dim: int = BOND_DIM,
        hidden: int = 128,
        layers: int = 3,
        out_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if kind not in ("gcn", "gin", "gat"):
            raise ValueError(kind)
        self.kind = kind
        self.dropout = dropout
        self.hidden = hidden

        self.input_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(layers):
            if kind == "gcn":
                conv = GCNConv(hidden, hidden)
            elif kind == "gin":
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                conv = GINEConv(mlp, edge_dim=edge_dim)
            else:
                conv = GATv2Conv(hidden, hidden // 4, heads=4, edge_dim=edge_dim)
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden))

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def node_embeddings(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index) if self.kind == "gcn" else conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def readout(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return self.head(global_mean_pool(h, batch))

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        return self.readout(self.node_embeddings(x, edge_index, edge_attr, batch), batch)


class AttentiveFPNet(nn.Module):
    """AttentiveFP wrapper. No exposed node embedding, so Grad-CAM is skipped for it."""

    def __init__(
        self,
        in_dim: int = ATOM_DIM,
        edge_dim: int = BOND_DIM,
        hidden: int = 128,
        layers: int = 3,
        timesteps: int = 2,
        out_dim: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.kind = "attentivefp"
        self.net = AttentiveFP(
            in_channels=in_dim,
            hidden_channels=hidden,
            out_channels=out_dim,
            edge_dim=edge_dim,
            num_layers=layers,
            num_timesteps=timesteps,
            dropout=dropout,
        )

    def forward(self, x, edge_index, edge_attr, batch) -> torch.Tensor:
        return self.net(x, edge_index, edge_attr, batch)


def build_model(kind: str, **kwargs) -> nn.Module:
    if kind == "attentivefp":
        return AttentiveFPNet(**kwargs)
    kwargs.pop("timesteps", None)
    return MessagePassingNet(kind=kind, **kwargs)


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode everywhere except dropout — used for the uncertainty feature."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
