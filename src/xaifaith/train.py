"""Training loop for the backbone predictors."""
from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data import SplitIndices, subset
from .models import build_model


@dataclass
class TrainConfig:
    architecture: str = "gin"
    hidden: int = 128
    layers: int = 3
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 64
    epochs: int = 120
    patience: int = 20
    seed: int = 0


def _run_epoch(model, loader, task_type, device, optimiser=None):
    training = optimiser is not None
    model.train() if training else model.eval()
    loss_fn = nn.BCEWithLogitsLoss() if task_type == "classification" else nn.MSELoss()

    losses, preds, targets = [], [], []
    for batch in loader:
        batch = batch.to(device)
        with torch.set_grad_enabled(training):
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = loss_fn(out, batch.y)
        if training:
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
        losses.append(float(loss.detach()))
        preds.append(out.detach().cpu().numpy())
        targets.append(batch.y.detach().cpu().numpy())

    if not preds:
        return float("nan"), float("nan")

    preds = np.concatenate(preds).ravel()
    targets = np.concatenate(targets).ravel()
    if task_type == "classification":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                score = roc_auc_score(targets, preds)
            except ValueError:  # single class present in this split
                score = float("nan")
    else:
        score = -mean_absolute_error(targets, preds)
    return float(np.mean(losses)), float(score)


def train_model(
    dataset: Sequence[Data],
    split: SplitIndices,
    task_type: str,
    config: TrainConfig,
    out_dir: str,
    device: str = "cuda",
) -> Dict[str, float]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(subset(dataset, split.train), batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(subset(dataset, split.val), batch_size=256)
    test_loader = DataLoader(subset(dataset, split.test), batch_size=256)

    model = build_model(
        config.architecture, hidden=config.hidden, layers=config.layers, dropout=config.dropout
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="max", factor=0.5, patience=8)

    best_score, best_epoch = -np.inf, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    used_loss_fallback = False

    for epoch in range(config.epochs):
        _run_epoch(model, train_loader, task_type, device, optimiser)
        val_loss, val_score = _run_epoch(model, val_loader, task_type, device)

        # Selection criterion: validation AUROC when it is defined, otherwise
        # negative validation loss. Without the fallback, a single-class or empty
        # validation split makes every epoch nan, no checkpoint is ever taken,
        # and the model silently saved is whatever the final epoch produced.
        if not np.isnan(val_score):
            criterion = val_score
        elif not np.isnan(val_loss):
            criterion = -val_loss
            used_loss_fallback = True
        else:
            continue

        scheduler.step(criterion)
        if criterion > best_score:
            best_score, best_epoch = criterion, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch - best_epoch >= config.patience:
            break

    if used_loss_fallback:
        warnings.warn(
            "validation AUROC was undefined for at least one epoch (single-class validation "
            "split); model selection fell back to validation loss. Check the split.",
            RuntimeWarning,
        )

    model.load_state_dict(best_state)
    _, test_score = _run_epoch(model, test_loader, task_type, device)

    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "config": config.__dict__, "task_type": task_type},
        os.path.join(out_dir, "model.pt"),
    )
    metrics = {"val_score": best_score, "test_score": test_score, "best_epoch": best_epoch}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def load_model(checkpoint_dir: str, device: str = "cuda") -> tuple[nn.Module, str]:
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    blob = torch.load(os.path.join(checkpoint_dir, "model.pt"), map_location=device, weights_only=False)
    cfg = blob["config"]
    model = build_model(
        cfg["architecture"], hidden=cfg["hidden"], layers=cfg["layers"], dropout=cfg["dropout"]
    )
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, blob["task_type"]
