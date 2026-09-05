"""Clean four-group DCRN / GuidedPGC(Scene Shift) audit.

The only train-time choices are:

  A: raw DCRN + CE
  B: raw DCRN + handcrafted Scene Shift
  C: GuidedPGC(ILDA) + DCRN + CE
  D: GuidedPGC(ILDA) + DCRN + handcrafted Scene Shift

No foundation, semantic, orthogonal, prototype, modulation, LMMD, SCL or
contrastive branch is imported or enabled here. Target GT is never loaded.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from config_Houston import HalfWidth  # noqa: E402
from model import DCRNClassifier  # noqa: E402
import utils  # noqa: E402

# Registered clean Step-1 source splits.  The optimization seed remains fixed
# at 1174 for every split so A/B are strictly matched.
SPLITS = (1174, 1370, 1417, 1418, 1421, 1535, 1546, 1599, 1610, 1631, 1703, 2141)
GROUPS = ("A", "B", "C", "D")
GROUP_DESCRIPTIONS = {
    "A": "DCRN + CE",
    "B": "DCRN + Scene Shift",
    "C": "GuidedPGC(ILDA) + DCRN + CE",
    "D": "GuidedPGC(ILDA) + DCRN + Scene Shift",
}


def center_patches(cube, centers, width=7):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    output = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        output[i] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return output


def source_split(gt, seed):
    """Keep the established 180-per-class source split exactly matched."""
    rng = np.random.RandomState(seed)
    padded = np.pad(gt, HalfWidth)
    rows, cols = np.nonzero(padded)
    train_indices, val_indices = [], []
    for cls in range(int(padded.max())):
        indices = [i for i in range(len(rows)) if padded[rows[i], cols[i]] == cls + 1]
        rng.shuffle(indices)
        train_indices.extend(indices[:180])
        val_indices.extend(indices[180:])
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_centers = np.asarray(
        [(rows[i] - HalfWidth, cols[i] - HalfWidth) for i in train_indices], dtype=np.int64
    )
    val_centers = np.asarray(
        [(rows[i] - HalfWidth, cols[i] - HalfWidth) for i in val_indices], dtype=np.int64
    )
    train_y = gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1
    val_y = gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    return train_centers, train_y, val_centers, val_y


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def augment(x):
    if torch.rand(()) < 0.5:
        x = x.flip(-1)
    if torch.rand(()) < 0.5:
        x = x.flip(-2)
    return x


def scene_shift(x, source_mean, source_std, target_mean, target_std, strength=0.7):
    """The validated handcrafted Target-Guided Scene Shift, unchanged."""
    source_mean = torch.as_tensor(source_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    source_std = torch.as_tensor(source_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    target_mean = torch.as_tensor(target_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    target_std = torch.as_tensor(target_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    shifted = (x - source_mean) / (source_std + 1e-5)
    shifted = shifted * (strength * target_std + (1.0 - strength) * source_std)
    shifted = shifted + strength * target_mean + (1.0 - strength) * source_mean
    scale = 1.0 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=x.device)
    low_frequency_noise = F.avg_pool2d(
        torch.randn_like(shifted), kernel_size=5, stride=1, padding=2
    )
    return (shifted * scale + 0.015 * low_frequency_noise).clamp(0, 1)


def build_band_adaptive_strength(source_mean, source_std, target_mean, target_std,
                                 alpha_min=0.4, alpha_max=0.8):
    """Deterministic source/target global-statistics band adaptation."""
    def normalize(values):
        values = np.asarray(values, dtype=np.float32)
        lo, hi = float(values.min()), float(values.max())
        return (values - lo) / max(hi - lo, 1e-8)
    delta_mu = normalize(np.abs(target_mean - source_mean))
    delta_std = normalize(np.abs(target_std - source_std))
    score = normalize(delta_mu + delta_std)
    alpha = alpha_min + score * (alpha_max - alpha_min)
    return score.astype(np.float32), alpha.astype(np.float32)


def band_adaptive_scene_shift(x, source_mean, source_std, target_mean, target_std,
                              alpha, strength_noise=True):
    sm = torch.as_tensor(source_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ss = torch.as_tensor(source_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    tm = torch.as_tensor(target_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ts = torch.as_tensor(target_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    a = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)[None, :, None, None]
    mu_mix = (1.0 - a) * sm + a * tm
    std_mix = (1.0 - a) * ss + a * ts
    shifted = (x - sm) / (ss + 1e-5) * std_mix + mu_mix
    scale = 1.0 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=x.device)
    low_frequency_noise = F.avg_pool2d(
        torch.randn_like(shifted), kernel_size=5, stride=1, padding=2
    )
    return (shifted * scale + 0.015 * low_frequency_noise).clamp(0, 1)


def build_target_modes(target, n_modes=4, seed=1174):
    """Unsupervised target spectral modes; target labels are never consulted."""
    pixels = target.reshape(-1, target.shape[-1]).astype(np.float32)
    scaler = StandardScaler().fit(pixels)
    z = scaler.transform(pixels)
    # Fit on a deterministic subsample for bounded memory/time, then retain
    # centers in the original spectral domain for per-band statistics.
    rng = np.random.RandomState(seed)
    take = min(len(z), 50000)
    ids = rng.choice(len(z), take, replace=False)
    km = KMeans(n_clusters=n_modes, random_state=seed, n_init=10).fit(z[ids])
    labels = km.predict(z)
    modes = []
    for mode in range(n_modes):
        members = pixels[labels == mode]
        if len(members) == 0:
            members = pixels
        modes.append({"mean": members.mean(0), "std": members.std(0) + 1e-5})
    mode_signatures = np.stack([m["mean"] for m in modes])
    return modes, mode_signatures


def conditional_scene_shift(x, source_mean, source_std, modes, mode_signatures, strength=0.7):
    """Scene Shift with source-sample-specific soft target spectral modes."""
    b = x.shape[0]
    source_signature = x.mean(dim=(-1, -2))
    signatures = torch.as_tensor(mode_signatures, device=x.device, dtype=x.dtype)
    source_signature = F.normalize(source_signature, dim=1)
    signatures = F.normalize(signatures, dim=1)
    weights = torch.softmax(source_signature @ signatures.t() / 0.1, dim=1)
    target_mean = torch.as_tensor(np.stack([m["mean"] for m in modes]), device=x.device, dtype=x.dtype)
    target_std = torch.as_tensor(np.stack([m["std"] for m in modes]), device=x.device, dtype=x.dtype)
    target_mean = weights @ target_mean
    target_std = weights @ target_std
    sm = torch.as_tensor(source_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ss = torch.as_tensor(source_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    tm = target_mean[:, :, None, None]
    ts = target_std[:, :, None, None]
    shifted = (x - sm) / (ss + 1e-5)
    shifted = shifted * (strength * ts + (1.0 - strength) * ss)
    shifted = shifted + strength * tm + (1.0 - strength) * sm
    scale = 1.0 + 0.04 * torch.randn(b, 1, 1, 1, device=x.device)
    low_frequency_noise = F.avg_pool2d(torch.randn_like(shifted), kernel_size=5, stride=1, padding=2)
    return (shifted * scale + 0.015 * low_frequency_noise).clamp(0, 1)


def build_target_neighbors(target, radius=2, top_k=4, threshold=0.85):
    """Build local target-only spectral neighbors without reading target GT."""
    height, width, bands = target.shape
    spectra = target.reshape(-1, bands).astype(np.float32)
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-8)
    neighbor_ids = np.full((height * width, top_k), -1, dtype=np.int64)
    neighbor_sims = np.zeros((height * width, top_k), dtype=np.float32)
    for row in range(height):
        for col in range(width):
            center = row * width + col
            candidates = []
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = row + dr, col + dc
                    if 0 <= rr < height and 0 <= cc < width:
                        idx = rr * width + cc
                        sim = float(np.dot(spectra[center], spectra[idx]))
                        if sim >= threshold:
                            candidates.append((sim, idx))
            candidates.sort(reverse=True)
            for slot, (sim, idx) in enumerate(candidates[:top_k]):
                neighbor_ids[center, slot] = idx
                neighbor_sims[center, slot] = sim
    return neighbor_ids, neighbor_sims


def neighborhood_consistency(model, center_x, neighbor_x, similarity,
                             confidence_threshold=0.7, active_fraction=1.0):
    """One-way high-confidence-to-low-confidence target neighborhood KL."""
    logits_i = model(center_x)
    logits_j = model(neighbor_x)
    p_i, p_j = logits_i.softmax(1), logits_j.softmax(1)
    conf_i, conf_j = p_i.max(1).values, p_j.max(1).values
    teacher_i = conf_i >= conf_j
    teacher_p = torch.where(teacher_i[:, None], p_i.detach(), p_j.detach())
    student_p = torch.where(teacher_i[:, None], p_j, p_i)
    high_conf = torch.maximum(conf_i, conf_j)
    valid = high_conf >= confidence_threshold
    # Select only the most reliable fraction among gated pairs. This keeps the
    # consistency signal sparse without introducing pseudo-label CE.
    if active_fraction < 1.0 and valid.any():
        scores = (similarity * high_conf).detach()
        valid_scores = scores[valid]
        cutoff = torch.quantile(valid_scores, 1.0 - active_fraction)
        valid = valid & (scores >= cutoff)
    kl = F.kl_div(student_p.clamp_min(1e-8).log(), teacher_p, reduction="none").sum(1)
    weights = similarity * high_conf * valid.float()
    denom = weights.sum()
    if denom.item() == 0:
        return logits_i.sum() * 0.0, 0.0, 0.0
    agreement = (p_i.argmax(1) == p_j.argmax(1)).float()
    return (weights * kl).sum() / denom, float(valid.float().mean()), float(agreement.mean())


def load_cubes(use_ilda):
    source, source_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    # Target imagery only. Houston18 GT is deliberately not opened in training.
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    if use_ilda:
        from UtilsCMS import ILDA
        source, target = ILDA(source, target, 2, 0.009)
    return source.astype(np.float32), source_gt, target.astype(np.float32)


def train_one(args):
    set_seed(args.optimization_seed)
    device = torch.device(args.device)
    use_ilda = args.group in ("C", "D")
    use_scene_shift = args.group in ("B", "D")
    source, source_gt, target = load_cubes(use_ilda)
    train_centers, train_y, val_centers, val_y = source_split(source_gt, args.split_seed)
    train_x = center_patches(source, train_centers)
    val_x = center_patches(source, val_centers)

    source_flat = source.reshape(-1, source.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    source_mean, source_std = source_flat.mean(0), source_flat.std(0)
    target_mean, target_std = target_flat.mean(0), target_flat.std(0)
    band_score = band_alpha = None
    if use_scene_shift and args.shift_mode == "band_adaptive":
        band_score, band_alpha = build_band_adaptive_strength(
            source_mean, source_std, target_mean, target_std,
            alpha_min=args.alpha_min, alpha_max=args.alpha_max,
        )
        print(json.dumps({
            "band_score": band_score.tolist(),
            "band_alpha": band_alpha.tolist(),
            "band_alpha_min": float(band_alpha.min()),
            "band_alpha_max": float(band_alpha.max()),
            "band_alpha_mean": float(band_alpha.mean()),
        }), flush=True)
    modes = mode_signatures = None
    if use_scene_shift and args.shift_mode == "conditional":
        modes, mode_signatures = build_target_modes(target, n_modes=args.num_modes, seed=args.optimization_seed)
    target_x = target_neighbors = target_similarity = None
    if args.use_target_neighborhood:
        target_x = center_patches(
            target,
            np.stack(np.meshgrid(np.arange(target.shape[0]), np.arange(target.shape[1]), indexing="ij"), -1).reshape(-1, 2),
        )
        target_neighbors, target_similarity = build_target_neighbors(
            target, radius=args.neighbor_radius, top_k=args.neighbor_top_k,
            threshold=args.neighbor_similarity_threshold,
        )
        valid_pair_count = int((target_neighbors >= 0).sum())
        print(json.dumps({"target_neighbor_pairs": valid_pair_count,
                          "target_neighbor_pair_ratio": valid_pair_count / target_neighbors.size,
                          "target_neighbor_nodes": int(target_neighbors.shape[0])}), flush=True)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=args.batch_size,
        shuffle=False,
    )
    model = DCRNClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    history = []
    best = {"val_acc": -1.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = seen = 0
        nbr_loss_sum = nbr_valid_sum = nbr_agreement_sum = nbr_seen = 0.0
        neighborhood_active = args.use_target_neighborhood and (
            epoch > args.neighborhood_warmup_epochs
        )
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(augment(x))
            loss = cross_entropy(logits, y)
            if use_scene_shift:
                if args.shift_mode == "conditional":
                    shifted = conditional_scene_shift(x, source_mean, source_std, modes, mode_signatures)
                elif args.shift_mode == "band_adaptive":
                    shifted = band_adaptive_scene_shift(
                        x, source_mean, source_std, target_mean, target_std, band_alpha
                    )
                else:
                    shifted = scene_shift(x, source_mean, source_std, target_mean, target_std)
                loss = loss + 0.5 * cross_entropy(model(augment(shifted)), y)
            if neighborhood_active:
                centers = torch.randint(0, len(target_x), (len(y),))
                slots = torch.randint(0, target_neighbors.shape[1], (len(y),))
                nids = target_neighbors[centers.numpy(), slots.numpy()]
                valid = nids >= 0
                if valid.any():
                    centers_v = centers[valid]
                    neigh_v = torch.from_numpy(nids[valid])
                    sim_v = torch.from_numpy(target_similarity[centers.numpy()[valid], slots.numpy()[valid]])
                    target_center = torch.from_numpy(target_x[centers_v.numpy()]).to(device)
                    target_neighbor = torch.from_numpy(target_x[neigh_v.numpy()]).to(device)
                    nbr_loss, valid_ratio, agreement = neighborhood_consistency(
                        model, target_center, target_neighbor, sim_v.to(device),
                        confidence_threshold=args.neighbor_confidence_threshold,
                        active_fraction=args.neighbor_active_fraction,
                    )
                    loss = loss + args.lambda_nbr * nbr_loss
                    nbr_loss_sum += float(nbr_loss.detach()) * len(centers_v)
                    nbr_valid_sum += valid_ratio * len(centers_v)
                    nbr_agreement_sum += agreement * len(centers_v)
                    nbr_seen += len(centers_v)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)

        model.eval()
        val_loss = val_correct = val_seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                y = y.to(device)
                val_loss += cross_entropy(logits, y).item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_seen += len(y)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss / val_seen,
            "val_acc": val_correct / val_seen,
        }
        if neighborhood_active:
            row.update({
                "neighbor_loss": nbr_loss_sum / max(nbr_seen, 1),
                "neighbor_effective_pair_ratio": nbr_valid_sum / max(nbr_seen, 1),
                "neighbor_prediction_agreement": nbr_agreement_sum / max(nbr_seen, 1),
            })
        history.append(row)
        print(json.dumps(row), flush=True)
        if epoch >= args.checkpoint_selection_start_epoch and row["val_acc"] > best["val_acc"]:
            best = row.copy()
            torch.save(
                {
                    "model": model.state_dict(),
                    "group": args.group,
                    "group_description": GROUP_DESCRIPTIONS[args.group],
                    "split_seed": args.split_seed,
                    "optimization_seed": args.optimization_seed,
                    "use_ilda": use_ilda,
                    "use_scene_shift": use_scene_shift,
                    "shift_mode": args.shift_mode,
                    "num_modes": args.num_modes,
                    "alpha_min": args.alpha_min,
                    "alpha_max": args.alpha_max,
                    "band_score": None if band_score is None else band_score.tolist(),
                    "band_alpha": None if band_alpha is None else band_alpha.tolist(),
                    "use_target_neighborhood": args.use_target_neighborhood,
                    "lambda_nbr": args.lambda_nbr,
                    "neighbor_radius": args.neighbor_radius,
                    "neighbor_top_k": args.neighbor_top_k,
                    "neighbor_similarity_threshold": args.neighbor_similarity_threshold,
                    "neighbor_confidence_threshold": args.neighbor_confidence_threshold,
                    "neighbor_active_fraction": args.neighbor_active_fraction,
                    "neighborhood_warmup_epochs": args.neighborhood_warmup_epochs,
                    "preprocessing": {
                        "source_target_ilda": use_ilda,
                        "patch_width": 7,
                        "data_range": "native Houston ori_data / ILDA output",
                    },
                    "backbone": {
                        "name": "DCRN_02",
                        "call": "DCRN_02(x, x)",
                        "cross_attention_source_target_interaction": False,
                    },
                    "disabled_losses": [
                        "foundation", "LMMD", "SCL", "prototype", "semantic", "orth", "modulation"
                    ],
                    "best": best,
                    "target_gt_used_for_training_or_selection": False,
                },
                args.output / "best.pth",
            )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    config.update({
        "group_description": GROUP_DESCRIPTIONS[args.group],
        "use_ilda": use_ilda,
        "use_scene_shift": use_scene_shift,
        "shift_mode": args.shift_mode,
        "num_modes": args.num_modes,
        "alpha_min": args.alpha_min,
        "alpha_max": args.alpha_max,
        "band_score": None if band_score is None else band_score.tolist(),
        "band_alpha": None if band_alpha is None else band_alpha.tolist(),
        "use_target_neighborhood": args.use_target_neighborhood,
        "lambda_nbr": args.lambda_nbr,
        "neighbor_radius": args.neighbor_radius,
        "neighbor_top_k": args.neighbor_top_k,
        "neighbor_similarity_threshold": args.neighbor_similarity_threshold,
        "neighbor_confidence_threshold": args.neighbor_confidence_threshold,
        "neighbor_active_fraction": args.neighbor_active_fraction,
        "neighborhood_warmup_epochs": args.neighborhood_warmup_epochs,
        "backbone": "DCRN_02(x, x)",
        "cross_attention_source_target_interaction": False,
        "target_gt_used_for_training_or_selection": False,
    })
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "summary.json").write_text(json.dumps({
        "config": config,
        "best": best,
        "target_gt_used_for_training_or_selection": False,
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--split-seed", type=int, choices=SPLITS, required=True)
    parser.add_argument("--optimization-seed", type=int, default=1174)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shift-mode", choices=("global", "conditional", "band_adaptive"), default="global")
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--alpha-min", type=float, default=0.4)
    parser.add_argument("--alpha-max", type=float, default=0.8)
    parser.add_argument("--use-target-neighborhood", action="store_true")
    parser.add_argument("--lambda-nbr", type=float, default=0.05)
    parser.add_argument("--neighbor-radius", type=int, default=2)
    parser.add_argument("--neighbor-top-k", type=int, default=4)
    parser.add_argument("--neighbor-similarity-threshold", type=float, default=0.85)
    parser.add_argument("--neighbor-confidence-threshold", type=float, default=0.7)
    parser.add_argument("--neighbor-active-fraction", type=float, default=1.0)
    parser.add_argument("--neighborhood-warmup-epochs", type=int, default=0)
    parser.add_argument("--checkpoint-selection-start-epoch", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    train_one(parse_args())
