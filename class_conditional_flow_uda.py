"""Minimal class-conditional transport prototype for UDA.

This file is intentionally standalone: it operates on already extracted source
and target features and does not depend on the MLUDA / SceneShift codebase.
Run it directly to execute the synthetic smoke test at the bottom.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def compute_source_prototypes(
    source_features: Tensor, source_labels: Tensor, num_classes: int
) -> Tuple[Tensor, Tensor]:
    """Return class means and a boolean mask indicating classes with examples."""
    if source_features.ndim != 2 or source_labels.ndim != 1:
        raise ValueError("features must be [N,D] and labels must be [N]")
    d = source_features.shape[1]
    prototypes = source_features.new_zeros(num_classes, d)
    valid = torch.zeros(num_classes, dtype=torch.bool, device=source_features.device)
    for c in range(num_classes):
        mask = source_labels == c
        if mask.any():
            prototypes[c] = source_features[mask].mean(dim=0)
            valid[c] = True
    return prototypes, valid


def compute_soft_membership(
    target_features: Tensor,
    target_probabilities: Tensor,
    source_prototypes: Tensor,
    beta: float = 1.0,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> Tensor:
    """Compute detached r_j^c from classifier probabilities and cosine similarity."""
    if target_probabilities.shape != (target_features.shape[0], source_prototypes.shape[0]):
        raise ValueError("target probabilities must have shape [Bt, num_classes]")
    cosine = F.normalize(target_features, dim=-1) @ F.normalize(source_prototypes, dim=-1).t()
    logits = torch.log(target_probabilities.clamp_min(eps)) + beta * cosine
    return F.softmax(logits / temperature, dim=-1).detach()


def uncertainty_filter(soft_membership: Tensor, tau_r: float = 0.6) -> Tensor:
    """Keep samples whose most likely soft class exceeds tau_r."""
    return soft_membership.max(dim=1).values > tau_r


def _sinkhorn_kernel(cost: Tensor, source_mass: Tensor, target_mass: Tensor,
                     reg: float, iterations: int) -> Tensor:
    # Inputs are nonnegative, normalized marginals. Small clamps avoid division by zero.
    kernel = torch.exp(-cost / reg).clamp_min(torch.finfo(cost.dtype).tiny)
    u = torch.ones_like(source_mass)
    v = torch.ones_like(target_mass)
    for _ in range(iterations):
        u = source_mass / (kernel @ v).clamp_min(1e-12)
        v = target_mass / (kernel.t() @ u).clamp_min(1e-12)
    return u[:, None] * kernel * v[None, :]


def classwise_sinkhorn_ot(
    source_features: Tensor,
    source_labels: Tensor,
    target_features: Tensor,
    soft_membership: Tensor,
    num_classes: int,
    reg: float = 0.05,
    iterations: int = 100,
    filtered_mask: Optional[Tensor] = None,
) -> Dict[int, Dict[str, Tensor]]:
    """Compute one cosine-cost entropic OT coupling per class.

    Source mass is uniform within a class; target mass is normalized r_j^c over
    the uncertainty-filtered target samples. Classes without mass are skipped.
    """
    if filtered_mask is None:
        filtered_mask = torch.ones(target_features.shape[0], dtype=torch.bool, device=target_features.device)
    result: Dict[int, Dict[str, Tensor]] = {}
    for c in range(num_classes):
        sm = source_labels == c
        tm = filtered_mask & (soft_membership[:, c] > 0)
        if not sm.any() or not tm.any():
            continue
        zs, zt = source_features[sm], target_features[tm]
        cost = 1.0 - F.normalize(zs, dim=-1) @ F.normalize(zt, dim=-1).t()
        source_mass = zs.new_full((zs.shape[0],), 1.0 / zs.shape[0])
        target_mass = soft_membership[tm, c]
        target_mass = target_mass / target_mass.sum().clamp_min(1e-12)
        coupling = _sinkhorn_kernel(cost, source_mass, target_mass, reg, iterations)
        result[c] = {"coupling": coupling, "source_indices": sm.nonzero(as_tuple=False).flatten(),
                     "target_indices": tm.nonzero(as_tuple=False).flatten(), "cost": cost}
    return result


def sample_ot_pairs(
    ot_result: Dict[int, Dict[str, Tensor]],
    source_features: Tensor,
    target_features: Tensor,
    max_pairs_per_class: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Construct weighted source-target pairs by taking all or top coupling entries."""
    src, tgt, labels, weights = [], [], [], []
    for c, item in ot_result.items():
        coupling = item["coupling"]
        flat = coupling.flatten()
        count = flat.numel() if max_pairs_per_class is None else min(max_pairs_per_class, flat.numel())
        chosen = torch.topk(flat, count).indices
        si, ti = torch.unravel_index(chosen, coupling.shape)
        src.append(source_features[item["source_indices"][si]])
        tgt.append(target_features[item["target_indices"][ti]])
        labels.append(torch.full((count,), c, dtype=torch.long, device=source_features.device))
        weights.append(flat[chosen])
    if not src:
        d = source_features.shape[1]
        empty = source_features.new_empty(0, d)
        return empty, empty, torch.empty(0, dtype=torch.long, device=source_features.device)
    return torch.cat(src), torch.cat(tgt), torch.cat(labels)


class ConditionalFlowMLP(nn.Module):
    """Small MLP predicting class-conditioned flow velocity."""

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 128,
                 class_dim: int = 32, time_dim: int = 32):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.class_embedding = nn.Embedding(num_classes, class_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + time_dim + class_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, z_tau: Tensor, tau: Tensor, class_labels: Tensor) -> Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        h = torch.cat([z_tau, self.time_mlp(tau.float()), self.class_embedding(class_labels)], dim=-1)
        return self.net(h)


def flow_matching_loss(model: ConditionalFlowMLP, source: Tensor, target: Tensor,
                       class_labels: Tensor, tau_max: float = 0.3) -> Tensor:
    tau = torch.rand(source.shape[0], device=source.device) * tau_max
    z_tau = (1.0 - tau[:, None]) * source + tau[:, None] * target
    velocity = target - source
    return F.mse_loss(model(z_tau, tau, class_labels), velocity)


def bridge_classification_loss(classifier: nn.Module, source: Tensor, target: Tensor,
                               class_labels: Tensor, tau_max: float = 0.3) -> Tensor:
    tau = torch.rand(source.shape[0], device=source.device) * tau_max
    z_tau = (1.0 - tau[:, None]) * source + tau[:, None] * target
    return F.cross_entropy(classifier(z_tau), class_labels)


def _synthetic_test(seed: int = 7) -> None:
    torch.manual_seed(seed)
    num_classes, d = 3, 8
    source_per_class, target_per_class = 12, 10
    centers = F.normalize(torch.randn(num_classes, d), dim=-1) * 3.0
    source = torch.cat([centers[c] + 0.25 * torch.randn(source_per_class, d) for c in range(num_classes)])
    labels = torch.arange(num_classes).repeat_interleave(source_per_class)
    target = torch.cat([centers[c] + 0.35 * torch.randn(target_per_class, d) + 0.15 for c in range(num_classes)])
    prototypes, valid = compute_source_prototypes(source, labels, num_classes)
    classifier = nn.Linear(d, num_classes)
    with torch.no_grad():
        classifier.weight.copy_(centers)
        classifier.bias.zero_()
    probs = classifier(target).softmax(dim=-1)
    membership = compute_soft_membership(target, probs, prototypes, beta=2.0, temperature=0.5)
    keep = uncertainty_filter(membership, tau_r=0.45)
    ot = classwise_sinkhorn_ot(source, labels, target, membership, num_classes, filtered_mask=keep)
    src_pair, tgt_pair, pair_labels = sample_ot_pairs(ot, source, target, max_pairs_per_class=24)
    flow = ConditionalFlowMLP(d, num_classes)
    bridge = nn.Linear(d, num_classes)
    if src_pair.numel():
        fm = flow_matching_loss(flow, src_pair, tgt_pair, pair_labels, tau_max=0.3)
        br = bridge_classification_loss(bridge, src_pair, tgt_pair, pair_labels, tau_max=0.3)
        total = fm + br
        total.backward()
    else:
        fm = br = source.sum() * 0.0
    print("Synthetic class-conditional transport test")
    print("valid source classes:", int(valid.sum()))
    print("filtered target samples:", int(keep.sum()), "/", target.shape[0])
    for c in range(num_classes):
        if c in ot:
            item = ot[c]
            print(f"class {c}: effective target={item['target_indices'].numel()}, coupling shape={tuple(item['coupling'].shape)}")
        else:
            print(f"class {c}: effective target=0, coupling shape=None")
    print(f"FM loss={fm.item():.6f}, bridge loss={br.item():.6f}")
    print("backward: OK" if src_pair.numel() else "backward: skipped (no OT pairs)")


if __name__ == "__main__":
    _synthetic_test()
