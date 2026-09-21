"""Post-hoc reliability/mass/diversity audit for epoch-100 transport models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
LEGACY = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path[:0] = [str(ROOT), str(LEGACY)]
import utils
from net2 import DSANSS
from class_conditional_flow_uda import compute_source_prototypes
from train_houston_agreement_transport import (
    agreement_membership,
    classwise_sinkhorn_ot_weighted,
)

K = 7
BATCH = 32


@torch.no_grad()
def extract(model, sx, sy, tx, ty, ids, device):
    model.eval()
    target_context = tx[ids[:BATCH]].to(device)
    source_features = []
    for start in range(0, len(sx), BATCH):
        source = sx[start:start+BATCH].to(device)
        source_features.append(model(source, target_context[:len(source)])[0])
    source_features = torch.cat(source_features)
    prototypes, valid = compute_source_prototypes(source_features, sy.to(device), K)

    target_features, probabilities = [], []
    source_context = sx[:BATCH].to(device)
    for start in range(0, len(ids), BATCH):
        target = tx[ids[start:start+BATCH]].to(device)
        output = model(source_context[:len(target)], target)
        target_features.append(output[5])
        probabilities.append(output[8].softmax(1))
    target_features = torch.cat(target_features)
    q = torch.cat(probabilities)
    s, r, agreement = agreement_membership(target_features, q, prototypes, valid)
    margin = r.topk(2, dim=1).values
    margin = margin[:, 0] - margin[:, 1]
    return source_features, prototypes, target_features, q, s, r, agreement, margin, ty[ids].to(device)


def weighted_diversity(features, weights):
    weights = weights.double()
    total = weights.sum().clamp_min(1e-12)
    normalized = weights / total
    x = F.normalize(features.double(), dim=1)
    mean = (normalized[:, None] * x).sum(0)
    covariance_trace = float((normalized * ((x - mean) ** 2).sum(1)).sum())
    pairwise_cosine_distance = float(1.0 - mean.square().sum())
    ess = float(total.square() / weights.square().sum().clamp_min(1e-12))
    return ess, covariance_trace, pairwise_cosine_distance


@torch.no_grad()
def analyze_model(name, checkpoint, sx, sy, tx, ty, ids, device):
    model = DSANSS(48, 7, K).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    fs, prototypes, ft, q, s, r, agreement, margin, labels = extract(
        model, sx, sy, tx, ty, ids, device
    )
    pred = r.argmax(1)
    pred_distance = 1.0 - F.cosine_similarity(ft, prototypes[pred])
    gt_distance = 1.0 - F.cosine_similarity(ft, prototypes[labels])

    # Expected per-target cost under each model's own training-time weighting.
    own_weight = agreement if name == "js" else margin if name == "margin" else torch.ones_like(margin)
    ot = classwise_sinkhorn_ot_weighted(fs, sy.to(device), ft, r, own_weight, K)
    target_cost_num = torch.zeros(len(ft), device=device)
    target_cost_den = torch.zeros(len(ft), device=device)
    for item in ot.values():
        coupling = item["coupling"]
        target_cost_num += (coupling * item["cost"]).sum(0)
        target_cost_den += coupling.sum(0)
    target_ot_cost = target_cost_num / target_cost_den.clamp_min(1e-12)

    order = agreement.argsort()
    bins = []
    for bin_id, index in enumerate(torch.tensor_split(order, 5)):
        bins.append({
            "bin": bin_id,
            "n": len(index),
            "agreement_min": float(agreement[index].min()),
            "agreement_max": float(agreement[index].max()),
            "agreement_mean": float(agreement[index].mean()),
            "target_accuracy": float((pred[index] == labels[index]).float().mean()),
            "ot_cost": float(target_ot_cost[index].mean()),
            "predicted_prototype_distance": float(pred_distance[index].mean()),
            "gt_prototype_distance": float(gt_distance[index].mean()),
            "margin_mean": float(margin[index].mean()),
        })

    classes = []
    for class_id in range(K):
        row = {"class": class_id}
        for weight_name, reliability in (("js", agreement), ("margin", margin)):
            weights = r[:, class_id] * reliability
            ess, covariance_trace, pairwise = weighted_diversity(ft, weights)
            row[weight_name] = {
                "mass": float(weights.sum()),
                "ess": ess,
                "covariance_trace": covariance_trace,
                "average_pairwise_cosine_distance": pairwise,
                "max_normalized_weight": float(weights.max() / weights.sum().clamp_min(1e-12)),
            }
        classes.append(row)
    return {
        "model": name,
        "checkpoint": str(checkpoint),
        "n": len(ids),
        "q_accuracy": float((q.argmax(1) == labels).float().mean()),
        "s_accuracy": float((s.argmax(1) == labels).float().mean()),
        "r_accuracy": float((pred == labels).float().mean()),
        "agreement_bins": bins,
        "per_class_weight_audit": classes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs_transport_ablation_100_1341")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs_strict_mluda_1341/ilda.npz")
    parser.add_argument("--samples", type=int, default=1216)
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--out", type=Path, default=ROOT / "runs_transport_ablation_100_1341/reliability_audit.json")
    args = parser.parse_args()
    device = torch.device("cuda")
    cached = np.load(args.cache)
    _, source_gt = utils.load_data_houston(
        str(LEGACY / "datasets/Houston/Houston13.mat"),
        str(LEGACY / "datasets/Houston/Houston13_7gt.mat"),
    )
    _, target_gt = utils.load_data_houston(
        str(LEGACY / "datasets/Houston/Houston18.mat"),
        str(LEGACY / "datasets/Houston/Houston18_7gt.mat"),
    )
    utils.set_seed(args.seed)
    source_x, source_y = utils.get_sample_data(cached["s"], source_gt, 3, 180)
    _, target_x, target_y, *_ = utils.get_all_data(cached["t"], target_gt, 3)
    sx = torch.as_tensor(source_x)
    sy = torch.as_tensor(source_y)
    tx = torch.as_tensor(target_x)
    ty = torch.as_tensor(target_y)
    generator = torch.Generator().manual_seed(args.seed + 9000)
    ids = torch.randperm(len(tx), generator=generator)[:args.samples]
    output = {
        "protocol": "fixed post-hoc subset; GT is diagnostic only",
        "seed": args.seed,
        "target_indices_sha256": __import__("hashlib").sha256(ids.numpy().tobytes()).hexdigest(),
        "models": [],
    }
    for name in ("js", "soft", "margin"):
        checkpoint = args.run_root / f"{name}_transport/epoch100.pth"
        output["models"].append(analyze_model(name, checkpoint, sx, sy, tx, ty, ids, device))
    args.out.write_text(json.dumps(output, indent=2))
    print(json.dumps(output))


if __name__ == "__main__":
    main()
