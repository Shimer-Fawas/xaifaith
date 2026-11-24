"""Atom-level attribution methods.

Every explainer has the signature

    explain(model, data, device) -> np.ndarray of shape [num_atoms]

Signed attributions are returned where the method produces them; faithfulness.py
ranks atoms by the signed value in the direction of the model's own prediction,
so sign is information, not noise.
"""
from __future__ import annotations

from typing import Callable, Dict

import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .baselines import ig_baseline_for
from .batching import forward_masked, masked_node_features, replicate


def _single_graph_inputs(data: Data, device: torch.device):
    x = data.x.to(device).clone()
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    batch = torch.zeros(int(data.num_nodes), dtype=torch.long, device=device)
    return x, edge_index, edge_attr, batch


def _grad_wrt_x(model, data, device, x_override=None, create_batch=None):
    x, edge_index, edge_attr, batch = _single_graph_inputs(data, device)
    if x_override is not None:
        x = x_override
    x = x.detach().requires_grad_(True)
    if create_batch is not None:
        edge_index, edge_attr, batch = create_batch
    out = model(x, edge_index, edge_attr, batch)
    out.sum().backward()
    return x, x.grad.detach()


# --------------------------------------------------------------------------- #
# gradient family
# --------------------------------------------------------------------------- #

def saliency(model, data: Data, device: torch.device) -> np.ndarray:
    x, grad = _grad_wrt_x(model, data, device)
    return grad.norm(dim=-1).cpu().numpy()


def saliency_signed(model, data: Data, device: torch.device) -> np.ndarray:
    """Signed counterpart of `saliency`.

    `saliency` returns the gradient NORM, which discards direction entirely.
    faithfulness.py ranks atoms in the direction of the model's own
    prediction, so an unsigned score cannot use that ranking. This variant
    tests whether saliency's null result comes from throwing away the sign
    rather than from the gradient being uninformative.
    """
    x, grad = _grad_wrt_x(model, data, device)
    return grad.sum(dim=-1).cpu().numpy()


def gradient_input(model, data: Data, device: torch.device) -> np.ndarray:
    x, grad = _grad_wrt_x(model, data, device)
    return (grad * x.detach()).sum(dim=-1).cpu().numpy()


def integrated_gradients(model, data: Data, device: torch.device, steps: int = 32) -> np.ndarray:
    """Riemann-trapezoid IG against the baseline named by XAIFAITH_IG_BASELINE.

    Defaults to all-zeros, the no-information point of our featurisation.
    """
    x, edge_index, edge_attr, batch = _single_graph_inputs(data, device)
    n, d = x.shape
    baseline = ig_baseline_for(x)

    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device).view(steps, 1, 1)
    path = baseline.unsqueeze(0) + alphas * (x - baseline).unsqueeze(0)
    path = path.reshape(steps * n, d).detach().requires_grad_(True)

    ei, ea, bt = replicate(data, steps, device)
    out = model(path, ei, ea, bt)
    out.sum().backward()

    grads = path.grad.detach().view(steps, n, d).mean(dim=0)
    return (grads * (x - baseline)).sum(dim=-1).cpu().numpy()


def _mol_seed(data: Data, salt: str = "") -> int:
    """Deterministic per-molecule seed for stochastic explainers.

    Salted per consumer so that no explainer shares a random stream with the
    random reference orderings in faithfulness.py, which are seeded from the
    unsalted SMILES digest. An unsalted collision would correlate the random
    control with the reference it is scored against.
    """
    return int.from_bytes(
        hashlib.sha256((str(data.smiles) + salt).encode()).digest()[:4], "little"
    )


def smoothgrad(
    model, data: Data, device: torch.device, samples: int = 25, noise: float = 0.15
) -> np.ndarray:
    x, edge_index, edge_attr, batch = _single_graph_inputs(data, device)
    n, d = x.shape
    sigma = noise * float(x.max() - x.min() + 1e-8)

    noisy = x.unsqueeze(0).repeat(samples, 1, 1)
    _g = torch.Generator(device=noisy.device)
    _g.manual_seed(_mol_seed(data, "smoothgrad"))
    noisy = noisy + torch.randn(
        noisy.shape, generator=_g, device=noisy.device, dtype=noisy.dtype
    ) * sigma
    noisy = noisy.reshape(samples * n, d).detach().requires_grad_(True)

    ei, ea, bt = replicate(data, samples, device)
    out = model(noisy, ei, ea, bt)
    out.sum().backward()

    grads = noisy.grad.detach().view(samples, n, d).mean(dim=0)
    return (grads * x).sum(dim=-1).cpu().numpy()


def grad_cam(model, data: Data, device: torch.device) -> np.ndarray:
    """Grad-CAM over the final message-passing layer. Needs node_embeddings()."""
    if not hasattr(model, "node_embeddings"):
        return np.full(int(data.num_nodes), np.nan)
    x, edge_index, edge_attr, batch = _single_graph_inputs(data, device)
    h = model.node_embeddings(x, edge_index, edge_attr, batch)
    h.retain_grad()
    out = model.readout(h, batch)
    out.sum().backward()
    weights = h.grad.detach().mean(dim=0)
    cam = F.relu((h.detach() * weights).sum(dim=-1))
    return cam.cpu().numpy()


# --------------------------------------------------------------------------- #
# perturbation family
# --------------------------------------------------------------------------- #

def occlusion(model, data: Data, device: torch.device) -> np.ndarray:
    """Leave-one-atom-out. Included as a strong reference, with a caveat.

    Occlusion is near-circular with respect to our deletion-based faithfulness
    label: both mask atom features and watch the output move. Report it, but do
    not treat a high occlusion faithfulness score as evidence about the other
    explainers. The meta-model experiments exclude it by default.
    """
    n = int(data.num_nodes)
    full = torch.ones(1, n, dtype=torch.bool)
    leave_one_out = ~torch.eye(n, dtype=torch.bool)
    masks = torch.cat([full, leave_one_out], dim=0)
    outs = forward_masked(model, data, masks, device)
    return (outs[0] - outs[1:]).cpu().numpy()


def gnn_explainer(model, data: Data, device: torch.device, epochs: int = 100) -> np.ndarray:
    """PyG GNNExplainer node-attribute mask, summed per atom."""
    try:
        from torch_geometric.explain import Explainer, GNNExplainer
        from torch_geometric.explain.config import ModelConfig
    except ImportError:
        return np.full(int(data.num_nodes), np.nan)

    x, edge_index, edge_attr, batch = _single_graph_inputs(data, device)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=ModelConfig(mode="regression", task_level="graph", return_type="raw"),
    )
    _s = _mol_seed(data, "gnn_explainer")
    torch.manual_seed(_s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_s)
    try:
        explanation = explainer(x, edge_index, edge_attr=edge_attr, batch=batch)
    except Exception:
        return np.full(int(data.num_nodes), np.nan)
    return explanation.node_mask.detach().sum(dim=-1).cpu().numpy()


def random_attribution(model, data: Data, device: torch.device) -> np.ndarray:
    return np.random.default_rng(
        _mol_seed(data, "random_attribution")
    ).normal(size=int(data.num_nodes))


EXPLAINERS: Dict[str, Callable] = {
    "saliency_signed": saliency_signed,
    "saliency": saliency,
    "gradient_input": gradient_input,
    "integrated_gradients": integrated_gradients,
    "smoothgrad": smoothgrad,
    "grad_cam": grad_cam,
    "gnn_explainer": gnn_explainer,
    "occlusion": occlusion,
    "random": random_attribution,
}

# Explainers whose faithfulness is partly definitional given our metric.
CIRCULAR_EXPLAINERS = {"occlusion"}


def compute_attribution(name: str, model, data: Data, device: torch.device) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    attribution = EXPLAINERS[name](model, data, device)
    model.zero_grad(set_to_none=True)
    return np.asarray(attribution, dtype=np.float64).ravel()
