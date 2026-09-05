"""Houston Target-Guided Scene Shift and Foundation Reliability training.

Target ground truth is never loaded here. HyperSIGMA features are read from an
offline cache whose target sample universe is the complete, unlabeled image.
Target ground truth is consumed only by the separate post-hoc evaluator.
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

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from config_Houston import HalfWidth  # noqa: E402
from model import SemanticSceneUDA  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
import utils  # noqa: E402

CLASSES = 7
FOUNDATION_CACHE = HERE / "cache" / "hypersigma_fspec_full48_all_target.npz"


def center_patches(cube, centers, width):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    output = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        output[i] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return output


def paired_source_samples(adapted, raw, gt, seed):
    rng = np.random.RandomState(seed)
    padded_gt = np.pad(gt, HalfWidth)
    rows, cols = np.nonzero(padded_gt)
    train_indices, val_indices = [], []
    for cls in range(int(padded_gt.max())):
        indices = [i for i in range(len(rows)) if padded_gt[rows[i], cols[i]] == cls + 1]
        rng.shuffle(indices)
        train_indices += indices[:180]
        val_indices += indices[180:]
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    train_centers = np.asarray(
        [(rows[i] - HalfWidth, cols[i] - HalfWidth) for i in train_indices], dtype=np.int64
    )
    val_centers = np.asarray(
        [(rows[i] - HalfWidth, cols[i] - HalfWidth) for i in val_indices], dtype=np.int64
    )
    train_labels = gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1
    val_labels = gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    return (
        train_centers,
        center_patches(adapted, train_centers, 7),
        center_patches(raw, train_centers, 33),
        train_labels,
        val_centers,
        center_patches(adapted, val_centers, 7),
        center_patches(raw, val_centers, 33),
        val_labels,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def shift_view(x, src_mean, src_std, tgt_mean, tgt_std, strength=0.7):
    """Validated handcrafted, label-preserving target-guided Scene Shift."""
    src_mean = torch.as_tensor(src_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    src_std = torch.as_tensor(src_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    tgt_mean = torch.as_tensor(tgt_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    tgt_std = torch.as_tensor(tgt_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    shifted = (x - src_mean) / (src_std + 1e-5)
    shifted = shifted * (strength * tgt_std + (1.0 - strength) * src_std)
    shifted = shifted + strength * tgt_mean + (1.0 - strength) * src_mean
    scale = 1.0 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=x.device)
    low_frequency_noise = F.avg_pool2d(
        torch.randn_like(shifted), kernel_size=5, stride=1, padding=2
    )
    return (shifted * scale + 0.015 * low_frequency_noise).clamp(0, 1)


def augment(x):
    if torch.rand(()) < 0.5:
        x = x.flip(-1)
    if torch.rand(()) < 0.5:
        x = x.flip(-2)
    return x


def _full_image_centers(shape):
    return np.argwhere(np.ones(shape, dtype=bool)).astype(np.int64)


def load_foundation_cache(cache_path, train_centers, train_labels, target_centers):
    """Load an exact-coordinate cache and build fixed source-train centers."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Foundation cache not found: {cache_path}. Run prepare_foundation_cache.py first."
        )
    required = {"source_centers", "source_fspec", "target_centers", "target_fspec"}
    with np.load(cache_path, allow_pickle=False) as cache:
        missing = required.difference(cache.files)
        if missing:
            raise RuntimeError(f"Foundation cache is missing arrays: {sorted(missing)}")
        if "target_gt_used_for_cache" not in cache.files:
            raise RuntimeError("Foundation cache lacks target-GT provenance metadata")
        if bool(np.asarray(cache["target_gt_used_for_cache"]).item()):
            raise RuntimeError("Refusing a foundation cache whose target sample set used target GT")

        cached_source_centers = np.asarray(cache["source_centers"], dtype=np.int64)
        source_features = np.asarray(cache["source_fspec"], dtype=np.float32)
        cached_target_centers = np.asarray(cache["target_centers"], dtype=np.int64)
        target_features = np.asarray(cache["target_fspec"], dtype=np.float32)
        teacher_checkpoint = str(np.asarray(cache.get("teacher_checkpoint", "unknown")).item())

    if len(cached_source_centers) != len(source_features):
        raise RuntimeError("Source center/feature lengths differ in foundation cache")
    if len(cached_target_centers) != len(target_features):
        raise RuntimeError("Target center/feature lengths differ in foundation cache")
    if cached_source_centers.ndim != 2 or cached_source_centers.shape[1] != 2:
        raise RuntimeError("Invalid source center shape in foundation cache")
    if not np.array_equal(cached_target_centers, target_centers):
        raise RuntimeError(
            "Target cache must cover the complete unlabeled image in exact row-major order"
        )
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise RuntimeError("Foundation cache contains NaN/Inf")

    source_index = {tuple(center): i for i, center in enumerate(cached_source_centers)}
    try:
        train_features = np.stack(
            [source_features[source_index[tuple(center)]] for center in train_centers]
        )
    except KeyError as exc:
        raise RuntimeError(f"Source training center missing from foundation cache: {exc.args[0]}") from exc

    class_centers = []
    for cls in range(CLASSES):
        mask = train_labels == cls
        if not np.any(mask):
            raise RuntimeError(f"Source training split has no samples for class {cls}")
        class_centers.append(train_features[mask].mean(axis=0))
    class_centers = np.stack(class_centers).astype(np.float32)
    if np.any(np.linalg.norm(class_centers, axis=1) <= 1e-12):
        raise RuntimeError("Foundation source class center has zero norm")
    return target_features, class_centers, teacher_checkpoint


def foundation_weighted_consistency(weak_logits, strong_logits, teacher_features,
                                    teacher_class_centers, tau_h):
    """Return detached reliability-weighted KL and label-free diagnostics."""
    with torch.no_grad():
        weak_prob = F.softmax(weak_logits, dim=1)
        weak_class = weak_prob.argmax(dim=1)
        strong_class = strong_logits.argmax(dim=1)
        teacher_similarity = F.linear(
            F.normalize(teacher_features, dim=1),
            F.normalize(teacher_class_centers, dim=1),
        )
        foundation_prob = F.softmax(teacher_similarity / tau_h, dim=1)
        foundation_class = foundation_prob.argmax(dim=1)

        student_agreement = weak_class.eq(strong_class)
        foundation_agreement = weak_class.eq(foundation_class)
        student_reliability = weak_prob.max(dim=1).values * student_agreement
        foundation_reliability = foundation_prob.max(dim=1).values * foundation_agreement
        reliability = (student_reliability * foundation_reliability).detach()

    per_sample_kl = F.kl_div(
        F.log_softmax(strong_logits, dim=1),
        weak_prob.detach(),
        reduction="none",
    ).sum(dim=1)
    loss = (reliability * per_sample_kl).sum() / reliability.sum().clamp_min(1e-6)
    diagnostics = {
        "reliability": reliability.mean(),
        "reliable_fraction": (reliability > 0).float().mean(),
        "student_agreement": student_agreement.float().mean(),
        "foundation_agreement": foundation_agreement.float().mean(),
        "foundation_confidence": foundation_prob.max(dim=1).values.mean(),
    }
    return loss, diagnostics


def train_one(args):
    set_seed(args.optimization_seed)
    device = torch.device(args.device)
    source, source_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    # Only target imagery is opened during training. Target GT is post-hoc only.
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    adapted_source, adapted_target = ILDA(source, target, 2, 0.009)
    train_centers, train_x, _, train_y, _, val_x, _, val_y = paired_source_samples(
        adapted_source, adapted_source, source_gt, args.split_seed
    )

    source_flat = adapted_source.reshape(-1, adapted_source.shape[-1])
    target_flat = adapted_target.reshape(-1, adapted_target.shape[-1])
    source_mean, source_std = source_flat.mean(0), source_flat.std(0)
    target_mean, target_std = target_flat.mean(0), target_flat.std(0)
    target_centers = _full_image_centers(adapted_target.shape[:2])
    target_x = center_patches(adapted_target, target_centers, 7)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=args.batch_size,
    )

    teacher_class_centers = None
    teacher_checkpoint = None
    if args.foundation_reliability:
        target_teacher_features, class_centers, teacher_checkpoint = load_foundation_cache(
            args.foundation_cache, train_centers, train_y, target_centers
        )
        teacher_class_centers = torch.from_numpy(class_centers).to(device)
        target_dataset = TensorDataset(
            torch.from_numpy(target_x), torch.from_numpy(target_teacher_features)
        )
    else:
        target_dataset = torch.from_numpy(target_x)
    target_loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = SemanticSceneUDA().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cross_entropy = nn.CrossEntropyLoss()
    history = []
    best = {"val_acc": -1.0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        target_iterator = iter(target_loader)
        sums = {
            key: 0.0 for key in (
                "total", "classification", "target", "reliability",
                "reliable_fraction", "student_agreement", "foundation_agreement",
                "foundation_confidence",
            )
        }
        correct = total_samples = 0
        for source_x, source_y in train_loader:
            try:
                target_batch = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                target_batch = next(target_iterator)

            if args.foundation_reliability:
                target_x_batch, teacher_features = target_batch
                teacher_features = teacher_features.to(device)
            else:
                target_x_batch = target_batch
                teacher_features = None
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x_batch = target_x_batch.to(device)
            shifted_source = shift_view(
                source_x, source_mean, source_std, target_mean, target_std
            )

            source_logits = model(augment(source_x))[3]
            shifted_logits = model(augment(shifted_source))[3]
            # Preserve the established weak/raw and strong/two-pass flip views.
            strong_target_logits = model(augment(augment(target_x_batch)))[3]
            weak_target_logits = model(target_x_batch)[3]

            classification_loss = cross_entropy(source_logits, source_y)
            if args.scene_shift:
                classification_loss = classification_loss + 0.5 * cross_entropy(
                    shifted_logits, source_y
                )

            target_loss = torch.zeros((), device=device)
            reliability_stats = {
                key: torch.zeros((), device=device) for key in (
                    "reliability", "reliable_fraction", "student_agreement",
                    "foundation_agreement", "foundation_confidence",
                )
            }
            if args.foundation_reliability:
                target_loss, reliability_stats = foundation_weighted_consistency(
                    weak_target_logits,
                    strong_target_logits,
                    teacher_features,
                    teacher_class_centers,
                    args.tau_h,
                )

            loss = classification_loss + args.lambda_target * target_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(source_y)
            total_samples += batch_size
            correct += (source_logits.argmax(1) == source_y).sum().item()
            sums["total"] += loss.item() * batch_size
            sums["classification"] += classification_loss.item() * batch_size
            sums["target"] += target_loss.item() * batch_size
            for key, value in reliability_stats.items():
                sums[key] += value.item() * batch_size

        model.eval()
        val_correct = val_samples = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for val_batch, labels in val_loader:
                logits = model(val_batch.to(device))[3]
                labels = labels.to(device)
                val_loss_sum += cross_entropy(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_samples += len(labels)

        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / total_samples,
            "loss_cls": sums["classification"] / total_samples,
            "loss_target": sums["target"] / total_samples,
            "reliability_mean": sums["reliability"] / total_samples,
            "reliable_fraction": sums["reliable_fraction"] / total_samples,
            "student_agreement": sums["student_agreement"] / total_samples,
            "foundation_agreement": sums["foundation_agreement"] / total_samples,
            "foundation_confidence": sums["foundation_confidence"] / total_samples,
            "train_acc": correct / total_samples,
            "val_loss": val_loss_sum / val_samples,
            "val_acc": val_correct / val_samples,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_acc"] > best["val_acc"]:
            best = row.copy()
            torch.save(
                {
                    "model": model.state_dict(),
                    "split_seed": args.split_seed,
                    "optimization_seed": args.optimization_seed,
                    "config": vars(args),
                    "best": best,
                    "teacher_checkpoint": teacher_checkpoint,
                    "teacher_centers_updated": False,
                    "target_gt_used_for_training_or_selection": False,
                },
                args.output / "best.pth",
            )

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "summary.json").write_text(json.dumps({
        "config": config,
        "best": best,
        "teacher_checkpoint": teacher_checkpoint,
        "teacher_centers_updated": False,
        "target_gt_used_for_training_or_selection": False,
    }, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("baseline", "scene_shift", "scene_shift_foundation_reliability"),
        default="scene_shift",
    )
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--optimization-seed", type=int, default=1174)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lambda-target", type=float, default=0.1)
    parser.add_argument("--tau-h", type=float, default=0.1)
    parser.add_argument("--foundation-cache", type=Path, default=FOUNDATION_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.scene_shift = args.stage != "baseline"
    args.foundation_reliability = args.stage == "scene_shift_foundation_reliability"
    args.output.mkdir(parents=True, exist_ok=True)
    return args


if __name__ == "__main__":
    train_one(parse_args())
