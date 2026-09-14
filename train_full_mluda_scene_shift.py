"""Clean full-MLUDA versus full-MLUDA + Target-Guided Scene Shift.

This runner retains the Houston MLUDA model, ILDA preprocessing, LMMD,
source/target supervised contrastive losses (target labels are predictions),
Domain Occupancy loss, and the official SGD schedule.  It fixes the official
evaluation protocol for a clean UDA comparison:

* the target training loader samples the complete target cube without GT;
* checkpoints are selected only by held-out source validation accuracy;
* Houston18 GT is loaded only after training for post-hoc evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

MLUDA_ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(MLUDA_ROOT))

from config_Houston import (  # noqa: E402
    BATCH_SIZE, CLASS_NUM, HalfWidth, l2_decay, lr, momentum, nBand,
)
from contrastive_loss import SupConLoss  # noqa: E402
from net2 import DSANSS  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
import mmd  # noqa: E402
import utils  # noqa: E402


def set_seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def center_patches(cube: np.ndarray, centers: np.ndarray, width: int = 7) -> np.ndarray:
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    out = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for index, (row, col) in enumerate(centers):
        out[index] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return out


def source_split(gt: np.ndarray, split_seed: int):
    """Reproduce the official 180-per-class split with a local RNG."""
    rng = np.random.RandomState(split_seed)
    centers_train, centers_val = [], []
    for class_id in range(1, CLASS_NUM + 1):
        centers = np.argwhere(gt == class_id)
        rng.shuffle(centers)
        centers_train.append(centers[:180])
        centers_val.append(centers[180:])
    train = np.concatenate(centers_train).astype(np.int64)
    val = np.concatenate(centers_val).astype(np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    train_y = gt[train[:, 0], train[:, 1]].astype(np.int64) - 1
    val_y = gt[val[:, 0], val[:, 1]].astype(np.int64) - 1
    return train, train_y, val, val_y


class FullCubePatchDataset(Dataset):
    """Label-free patch access over every target pixel."""

    def __init__(self, cube: np.ndarray, width: int = 7):
        self.height, self.width, self.bands = cube.shape
        self.patch_width = width
        half = width // 2
        self.padded = np.pad(
            cube.astype(np.float32, copy=False),
            ((half, half), (half, half), (0, 0)),
            mode="constant",
        )

    def __len__(self):
        return self.height * self.width

    def __getitem__(self, index):
        row, col = divmod(int(index), self.width)
        width = self.patch_width
        patch = self.padded[row:row + width, col:col + width].transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(patch))


def scene_shift(x, source_mean, source_std, target_mean, target_std, strength=0.8):
    """Frozen validated global handcrafted Scene Shift implementation."""
    sm = torch.as_tensor(source_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ss = torch.as_tensor(source_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    tm = torch.as_tensor(target_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ts = torch.as_tensor(target_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    shifted = (x - sm) / (ss + 1e-5)
    shifted = shifted * (strength * ts + (1.0 - strength) * ss)
    shifted = shifted + strength * tm + (1.0 - strength) * sm
    scale = 1.0 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=x.device)
    smooth_noise = F.avg_pool2d(torch.randn_like(shifted), kernel_size=5, stride=1, padding=2)
    return (shifted * scale + 0.015 * smooth_noise).clamp(0.0, 1.0)


def official_augmentations(source, target, device):
    """Keep the original MLUDA radiation/flip augmentation behavior."""
    source_noise = utils.radiation_noise(source.cpu()).float().to(device)
    target_noise = utils.radiation_noise(target.cpu()).float().to(device)
    source_flip = utils.flip_augmentation(source.cpu()).float().to(device)
    target_flip = utils.flip_augmentation(target.cpu()).float().to(device)
    return source_noise, target_noise, source_flip, target_flip


def evaluate_source(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x, x)[3]
            loss_sum += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
    return loss_sum / count, correct / count


def evaluate_target(model, target_cube, target_gt, source_reference, device):
    centers = np.argwhere(target_gt > 0)
    labels = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(centers), 512):
            patches = torch.from_numpy(center_patches(target_cube, centers[start:start + 512])).to(device)
            for offset in range(0, len(patches), BATCH_SIZE):
                target_batch = patches[offset:offset + BATCH_SIZE]
                repeats = math.ceil(len(target_batch) / len(source_reference))
                reference = source_reference.repeat((repeats, 1, 1, 1))[:len(target_batch)].to(device)
                predictions.append(model(reference, target_batch)[8].argmax(1).cpu().numpy())
    prediction = np.concatenate(predictions)
    confusion = metrics.confusion_matrix(labels, prediction, labels=np.arange(CLASS_NUM))
    per_class = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    return {
        "oa": float((prediction == labels).mean()),
        "aa": float(per_class.mean()),
        "kappa": float(metrics.cohen_kappa_score(labels, prediction, labels=np.arange(CLASS_NUM))),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": np.bincount(prediction, minlength=CLASS_NUM).tolist(),
        "num_target_evaluation_samples": int(len(labels)),
    }


def make_optimizer(model, learning_rate):
    # Match the official parameter groups and per-epoch optimizer construction.
    return torch.optim.SGD(
        [
            {"params": model.feature_layers.parameters()},
            {"params": model.fc1.parameters(), "lr": learning_rate},
            {"params": model.fc2.parameters(), "lr": learning_rate},
            {"params": model.head1.parameters(), "lr": learning_rate},
            {"params": model.head2.parameters(), "lr": learning_rate},
        ],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=l2_decay,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=(
            "full_mluda",
            "full_mluda_scene_shift",  # legacy extra supervised shift branch
            "mluda_source_only",
            "mluda_shift_counterpart",
            "mluda_dual_counterpart",
            "mluda_dual_counterpart_ce025",
            "mluda_dual_counterpart_ce05",
            "mluda_dual_target_control",
        ),
        required=True,
    )
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--optimization-seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scene-shift-alpha", type=float, default=0.8)
    parser.add_argument("--scene-shift-weight", type=float, default=0.5)
    parser.add_argument("--dual-gamma", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.optimization_seed is None:
        args.optimization_seed = args.split_seed
    args.output.mkdir(parents=True, exist_ok=True)
    set_seed(args.optimization_seed)
    device = torch.device(args.device)

    source_raw, source_gt = utils.load_data_houston(
        str(MLUDA_ROOT / "datasets/Houston/Houston13.mat"),
        str(MLUDA_ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    # Imagery only: Houston18 ground truth is deliberately not opened here.
    target_raw = hdf5storage.loadmat(
        str(MLUDA_ROOT / "datasets/Houston/Houston18.mat")
    )["ori_data"]
    source, target = ILDA(source_raw, target_raw, 2, 0.009)
    source = source.astype(np.float32)
    target = target.astype(np.float32)

    train_centers, train_y, val_centers, val_y = source_split(source_gt, args.split_seed)
    train_x = center_patches(source, train_centers)
    val_x = center_patches(source, val_centers)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)),
        batch_size=BATCH_SIZE, shuffle=False,
    )
    target_dataset = FullCubePatchDataset(target)
    target_samples_per_epoch = len(train_loader) * BATCH_SIZE
    target_sampler = RandomSampler(
        target_dataset, replacement=False, num_samples=target_samples_per_epoch,
    )
    target_loader = DataLoader(
        target_dataset, batch_size=BATCH_SIZE, sampler=target_sampler, drop_last=True,
    )
    target_loader2 = None
    if args.method == "mluda_dual_target_control":
        target_sampler2 = RandomSampler(
            target_dataset, replacement=False, num_samples=target_samples_per_epoch,
        )
        target_loader2 = DataLoader(
            target_dataset, batch_size=BATCH_SIZE, sampler=target_sampler2, drop_last=True,
        )

    source_pixels = source.reshape(-1, nBand)
    target_pixels = target.reshape(-1, nBand)
    source_mean, source_std = source_pixels.mean(0), source_pixels.std(0)
    target_mean, target_std = target_pixels.mean(0), target_pixels.std(0)

    model = DSANSS(nBand, 7, CLASS_NUM).to(device)
    ce = nn.CrossEntropyLoss().to(device)
    source_contrast = SupConLoss(temperature=0.1).to(device)
    target_contrast = SupConLoss(temperature=0.1).to(device)
    domain_occupancy = utils.Domain_Occ_loss().to(device)
    history = []
    best = {"val_acc": -1.0}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        learning_rate = lr / math.pow(1 + 10 * (epoch - 1) / args.epochs, 0.75)
        optimizer = make_optimizer(model, learning_rate)
        target_iterator = iter(target_loader)
        target_iterator2 = iter(target_loader2) if target_loader2 is not None else None
        sums = {name: 0.0 for name in ("total", "cls", "shift", "lmmd", "scl_source", "scl_target", "domain", "dual_adapt", "dual_shift_ce")}
        correct = count = 0

        for source_data, source_label in train_loader:
            source_data = source_data.to(device)
            source_label = source_label.to(device)
            # A' is deliberately source-only: no target data enters training.
            if args.method == "mluda_source_only":
                counterpart_data = source_data
            elif args.method == "mluda_shift_counterpart":
                # Do not draw or forward a real target patch in C. The only
                # target information retained is the frozen global statistics
                # computed before training from the unlabeled target cube.
                counterpart_data = scene_shift(
                    source_data, source_mean, source_std,
                    target_mean, target_std, strength=args.scene_shift_alpha,
                )
            else:
                try:
                    target_data = next(target_iterator)
                except StopIteration:
                    target_iterator = iter(target_loader)
                    target_data = next(target_iterator)
                target_data = target_data.to(device)
                counterpart_data = target_data
                if args.method == "mluda_dual_target_control":
                    try:
                        target_data2 = next(target_iterator2)
                    except StopIteration:
                        target_iterator2 = iter(target_loader2)
                        target_data2 = next(target_iterator2)
                    target_data2 = target_data2.to(device)
            source_noise, counterpart_noise, source_flip, counterpart_flip = official_augmentations(
                source_data, counterpart_data, device
            )

            (source_features, _, _, source_logits, source_domain,
             target_features, _, _, target_logits, target_domain) = model(source_data, counterpart_data)
            (_, source_view1, _, _, _, _, _, target_view1, _, _) = model(source_noise, counterpart_noise)
            (_, source_view2, _, _, _, _, _, target_view2, _, _) = model(source_flip, counterpart_flip)

            cls_loss = ce(source_logits, source_label)
            if args.method == "mluda_source_only":
                # Keep the DSANSS wrapper and source-side CE, but disable every
                # target-driven objective for the fair same-framework baseline.
                lmmd_loss = source_logits.new_zeros(())
                source_scl = source_logits.new_zeros(())
                target_scl = source_logits.new_zeros(())
                domain_loss = source_logits.new_zeros(())
                total_loss = cls_loss
            else:
                pseudo_target = target_logits.softmax(1).detach().argmax(1)
                lmmd_loss = mmd.lmmd(
                    source_features, target_features, source_label,
                    target_logits.softmax(1), BATCH_SIZE=BATCH_SIZE, CLASS_NUM=CLASS_NUM,
                )
                source_scl = source_contrast(
                    torch.cat([source_view1.unsqueeze(1), source_view2.unsqueeze(1)], 1),
                    source_label,
                )
                target_scl = target_contrast(
                    torch.cat([target_view1.unsqueeze(1), target_view2.unsqueeze(1)], 1),
                    pseudo_target,
                )
                domain_loss = domain_occupancy(source_domain, target_domain)
                adaptation_weight = 2 / (1 + math.exp(-10 * epoch / args.epochs)) - 1
                total_loss = (
                    cls_loss + 0.01 * adaptation_weight * lmmd_loss
                    + source_scl + target_scl + domain_loss
                )

                if args.method in ("mluda_dual_counterpart", "mluda_dual_counterpart_ce025", "mluda_dual_counterpart_ce05", "mluda_dual_target_control"):
                    # Second adaptation counterpart: shifted source is treated
                    # as unlabeled. Its source labels are used only by the
                    # source side of LMMD/SCL, never as target supervision.
                    if args.method == "mluda_dual_target_control":
                        shifted_counterpart = target_data2
                    else:
                        shifted_counterpart = scene_shift(
                            source_data, source_mean, source_std,
                            target_mean, target_std, strength=args.scene_shift_alpha,
                        )
                    shift_src_noise, shift_tgt_noise, shift_src_flip, shift_tgt_flip = official_augmentations(
                        source_data, shifted_counterpart, device
                    )
                    (sx_feat, _, _, sx_logits, sx_domain,
                     st_feat, _, _, st_logits, st_domain) = model(source_data, shifted_counterpart)
                    (_, sx_v1, _, _, _, _, _, st_v1, _, _) = model(shift_src_noise, shift_tgt_noise)
                    (_, sx_v2, _, _, _, _, _, st_v2, _, _) = model(shift_src_flip, shift_tgt_flip)
                    dual_lmmd = mmd.lmmd(
                        sx_feat, st_feat, source_label, st_logits.softmax(1),
                        BATCH_SIZE=BATCH_SIZE, CLASS_NUM=CLASS_NUM,
                    )
                    dual_source_scl = source_contrast(
                        torch.cat([sx_v1.unsqueeze(1), sx_v2.unsqueeze(1)], 1), source_label
                    )
                    dual_target_scl = target_contrast(
                        torch.cat([st_v1.unsqueeze(1), st_v2.unsqueeze(1)], 1),
                        st_logits.softmax(1).detach().argmax(1),
                    )
                    dual_domain = domain_occupancy(sx_domain, st_domain)
                    dual_adapt = (
                        0.01 * adaptation_weight * dual_lmmd
                        + dual_source_scl + dual_target_scl + dual_domain
                    )
                    total_loss = total_loss + args.dual_gamma * dual_adapt
                    shift_ce_weight = {
                        "mluda_dual_counterpart": 0.0,
                        "mluda_dual_counterpart_ce025": 0.25,
                        "mluda_dual_counterpart_ce05": 0.5,
                        "mluda_dual_target_control": 0.0,
                    }[args.method]
                    dual_shift_ce = ce(st_logits, source_label)
                    total_loss = total_loss + shift_ce_weight * dual_shift_ce

            shift_loss = source_logits.new_zeros(())
            if args.method == "full_mluda_scene_shift":
                shifted_source = scene_shift(
                    source_data, source_mean, source_std, target_mean, target_std,
                    strength=args.scene_shift_alpha,
                )
                shifted_logits = model(shifted_source, target_data)[3]
                shift_loss = ce(shifted_logits, source_label)
                total_loss = total_loss + args.scene_shift_weight * shift_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_count = len(source_label)
            count += batch_count
            correct += (source_logits.argmax(1) == source_label).sum().item()
            for name, value in (
                ("total", total_loss), ("cls", cls_loss), ("shift", shift_loss),
                ("lmmd", lmmd_loss), ("scl_source", source_scl),
                ("scl_target", target_scl), ("domain", domain_loss),
                ("dual_adapt", source_logits.new_zeros(()) if args.method not in ("mluda_dual_counterpart", "mluda_dual_counterpart_ce025", "mluda_dual_counterpart_ce05", "mluda_dual_target_control") else dual_adapt),
                ("dual_shift_ce", source_logits.new_zeros(()) if args.method not in ("mluda_dual_counterpart", "mluda_dual_counterpart_ce025", "mluda_dual_counterpart_ce05", "mluda_dual_target_control") else dual_shift_ce),
            ):
                sums[name] += float(value.detach()) * batch_count

        val_loss, val_acc = evaluate_source(model, val_loader, device)
        row = {
            "epoch": epoch,
            "lr": learning_rate,
            "train_total_loss": sums["total"] / count,
            "train_cls_loss": sums["cls"] / count,
            "train_shift_loss": sums["shift"] / count,
            "train_lmmd_loss": sums["lmmd"] / count,
            "train_source_scl": sums["scl_source"] / count,
            "train_target_scl": sums["scl_target"] / count,
            "train_domain_loss": sums["domain"] / count,
            "train_dual_adapt": sums["dual_adapt"] / count,
            "train_dual_shift_ce": sums["dual_shift_ce"] / count,
            "train_acc": correct / count,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_acc > best["val_acc"]:
            best = row.copy()
            torch.save(
                {
                    "model": model.state_dict(),
            "method": args.method,
                    "split_seed": args.split_seed,
                    "optimization_seed": args.optimization_seed,
                    "best": best,
                    "target_training_sampling": "all target pixels; no target GT mask",
                    "target_gt_used_for_training_or_selection": False,
                },
                args.output / "best.pth",
            )

    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    checkpoint = torch.load(args.output / "best.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    # First and only target-GT load: final post-hoc evaluation.
    _, target_gt = utils.load_data_houston(
        str(MLUDA_ROOT / "datasets/Houston/Houston18.mat"),
        str(MLUDA_ROOT / "datasets/Houston/Houston18_7gt.mat"),
    )
    source_reference = torch.from_numpy(train_x[:BATCH_SIZE])
    result = evaluate_target(model, target, target_gt, source_reference, device)
    result.update({
        "method": args.method,
        "split_seed": args.split_seed,
        "optimization_seed": args.optimization_seed,
        "best_epoch": int(best["epoch"]),
        "source_val_accuracy": float(best["val_acc"]),
        "train_seconds": time.time() - start_time,
        "scene_shift_alpha": args.scene_shift_alpha if args.method in ("full_mluda_scene_shift", "mluda_shift_counterpart", "mluda_dual_counterpart", "mluda_dual_counterpart_ce025", "mluda_dual_counterpart_ce05") else None,
        "scene_shift_weight": args.scene_shift_weight if args.method == "full_mluda_scene_shift" else None,
        "adaptation_counterpart": {
            "full_mluda": "real target x_t",
            "full_mluda_scene_shift": "real target x_t + extra supervised shifted-source branch (legacy; not this counterpart experiment)",
            "mluda_source_only": "none (source x_s paired with itself)",
            "mluda_shift_counterpart": "unlabeled SceneShift(x_s; target stats, alpha=0.8)",
            "mluda_dual_counterpart": "real target x_t + unlabeled SceneShift(x_s; target stats, alpha=0.8)",
            "mluda_dual_counterpart_ce025": "real target x_t + unlabeled SceneShift counterpart + shifted-source CE (0.25)",
            "mluda_dual_counterpart_ce05": "real target x_t + unlabeled SceneShift counterpart + shifted-source CE (0.5)",
            "mluda_dual_target_control": "real target x_t1 + independent real target x_t2",
        }[args.method],
        "dual_gamma": args.dual_gamma,
        "shifted_source_ce_weight": {
            "mluda_dual_counterpart": 0.0,
            "mluda_dual_counterpart_ce025": 0.25,
            "mluda_dual_counterpart_ce05": 0.5,
        }.get(args.method, None),
        "target_training_sampling": "all target pixels; no target GT mask",
        "target_gt_used_for_training_or_selection": False,
        "full_mluda_components": [
            "ILDA", "DSANSS/MBCA", "LMMD", "source SCL", "target pseudo-label SCL",
            "Domain Occupancy loss", "radiation noise", "flip augmentation",
        ],
    })
    config = vars(args).copy()
    config["output"] = str(config["output"])
    config.update({
        "batch_size": BATCH_SIZE,
        "optimizer": "official per-epoch SGD",
        "initial_lr": lr,
        "momentum": momentum,
        "weight_decay": l2_decay,
        "patch_size": 7,
        "source_train_per_class": 180,
        "checkpoint_selection": "source validation accuracy only",
        "target_training_sampling": "all target pixels; no target GT mask",
        "dual_gamma": args.dual_gamma,
    })
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    with (args.output / "per_class.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "accuracy"])
        writer.writerows((index + 1, value) for index, value in enumerate(result["per_class_accuracy"]))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
