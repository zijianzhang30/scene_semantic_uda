"""Cross-dataset clean DAMamba-style SceneShift benchmark.

This is intentionally a dataset adapter around the corrected Houston Mamba
recipe, not a new backbone or UDA method.  ``MambaFeature`` hard-codes a
12x12 grid, so Pavia and Shanghai use symmetric 12x12 extraction too.  Their
native clean preprocessing remains per-scene z-score and SceneShift is not
clamped on those standardized tensors.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import hdf5storage
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

from mamba_model import MambaBackboneClassifier

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024/datasets")
SPLITS = (1174, 1703, 2141)
PATCH = 12  # Architectural requirement of DAMamba_basenet.MambaFeature.
DATASETS = {
    "pavia": {"classes": 7, "bands": 102, "normalization": "zscore"},
    "shanghai_hangzhou": {"classes": 3, "bands": 198, "normalization": "zscore"},
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def zscore(cube: np.ndarray) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1]).astype(np.float32)
    return ((flat - flat.mean(0)) / (flat.std(0) + 1e-8)).reshape(cube.shape).astype(np.float32)


def load_cubes(dataset: str):
    """Return images and source GT only; target GT stays unopened until eval."""
    if dataset == "pavia":
        source = sio.loadmat(ROOT / "Pavia/paviaU.mat")["paviaU"]
        target = sio.loadmat(ROOT / "Pavia/pavia.mat")["pavia"]
        source_gt = sio.loadmat(ROOT / "Pavia/paviaU_gt_7.mat")["paviaU_gt_7"]
    elif dataset == "shanghai_hangzhou":
        item = sio.loadmat(ROOT / "Shanghai-Hangzhou/DataCube.mat")
        source, target, source_gt = item["DataCube1"], item["DataCube2"], item["gt1"]
    else:
        raise ValueError(dataset)
    return zscore(source), zscore(target), np.asarray(source_gt, dtype=np.int64)


def load_target_gt(dataset: str) -> np.ndarray:
    if dataset == "pavia":
        return sio.loadmat(ROOT / "Pavia/pavia_gt_7.mat")["pavia_gt_7"].astype(np.int64)
    return sio.loadmat(ROOT / "Shanghai-Hangzhou/DataCube.mat")["gt2"].astype(np.int64)


def source_split(gt: np.ndarray, nclass: int, seed: int):
    rng = np.random.RandomState(seed)
    train, val = [], []
    for class_id in range(1, nclass + 1):
        centers = np.argwhere(gt == class_id)
        if len(centers) < 180:
            raise ValueError(f"class {class_id} has only {len(centers)} samples")
        rng.shuffle(centers)
        train.append(centers[:180])
        val.append(centers[180:])
    train, val = np.concatenate(train).astype(np.int64), np.concatenate(val).astype(np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    return train, gt[train[:, 0], train[:, 1]].astype(np.int64) - 1, val, gt[val[:, 0], val[:, 1]].astype(np.int64) - 1


def extract_patches(cube: np.ndarray, centers: np.ndarray) -> np.ndarray:
    half = PATCH // 2
    # Match the corrected DAMamba even-patch convention.
    padded = np.pad(cube, ((half + 1, half + 1), (half + 1, half + 1), (0, 0)), mode="symmetric")
    out = np.empty((len(centers), cube.shape[-1], PATCH, PATCH), dtype=np.float32)
    for i, (row, col) in enumerate(centers):
        out[i] = padded[row + 1:row + 1 + PATCH, col + 1:col + 1 + PATCH].transpose(2, 0, 1)
    return out


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(()) < 0.5:
        x = x.flip(-1)
    if torch.rand(()) < 0.5:
        x = x.flip(-2)
    return x


def scene_shift(x, sm, ss, tm, ts, alpha: float, clamp: bool):
    to = lambda a: torch.as_tensor(a, dtype=x.dtype, device=x.device)[None, :, None, None]
    shifted = (x - to(sm)) / (to(ss) + 1e-5)
    shifted = shifted * (alpha * to(ts) + (1.0 - alpha) * to(ss))
    shifted = shifted + alpha * to(tm) + (1.0 - alpha) * to(sm)
    shifted = shifted * (1.0 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=x.device))
    shifted = shifted + 0.015 * F.avg_pool2d(torch.randn_like(shifted), 5, 1, 2)
    return shifted.clamp(0, 1) if clamp else shifted


def make_optimizer(model: MambaBackboneClassifier, initial_lr: float):
    attention = list(model.channel_attention.parameters()) + list(model.spatial_attention.parameters())
    groups = [
        {"params": model.backbone.parameters(), "lr": 0.1},
        {"params": list(model.bottleneck.parameters()) + list(model.classifier.parameters()) + attention, "lr": 1.0},
    ]
    optimizer = torch.optim.SGD(groups, lr=initial_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: initial_lr * (1.0 + 0.0003 * step) ** (-0.75)
    )
    return optimizer, scheduler


def evaluate(model, cube, gt, nclass, device, batch_size=32):
    centers = np.argwhere(gt > 0)
    labels = gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    pred = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            x = torch.from_numpy(extract_patches(cube, centers[start:start + batch_size])).to(device)
            pred.append(model(x).argmax(1).cpu().numpy())
    pred = np.concatenate(pred)
    confusion = metrics.confusion_matrix(labels, pred, labels=np.arange(nclass))
    per_class = np.diag(confusion) / np.maximum(confusion.sum(1), 1)
    return {
        "oa": float((pred == labels).mean()),
        "aa": float(per_class.mean()),
        "kappa": float(metrics.cohen_kappa_score(labels, pred, labels=np.arange(nclass))),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": confusion.tolist(),
        "target_evaluation_samples": int(len(labels)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    p.add_argument("--method", choices=("ce", "scene_shift"), required=True)
    p.add_argument("--split-seed", choices=SPLITS, type=int, required=True)
    p.add_argument("--optimization-seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.optimization_seed is None:
        args.optimization_seed = args.split_seed
    set_seed(args.optimization_seed)
    device = torch.device(args.device)
    spec = DATASETS[args.dataset]
    source, target, source_gt = load_cubes(args.dataset)
    train_c, train_y, val_c, val_y = source_split(source_gt, spec["classes"], args.split_seed)
    train_x, val_x = extract_patches(source, train_c), extract_patches(source, val_c)
    sf, tf = source.reshape(-1, source.shape[-1]), target.reshape(-1, target.shape[-1])
    sm, ss, tm, ts = sf.mean(0), sf.std(0), tf.mean(0), tf.std(0)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size, shuffle=False)
    model = MambaBackboneClassifier(bands=source.shape[-1], classes=spec["classes"]).to(device)
    optimizer, scheduler = make_optimizer(model, args.lr)
    expected = {id(q) for q in model.parameters() if q.requires_grad}
    actual = {id(q) for group in optimizer.param_groups for q in group["params"]}
    assert expected == actual, "optimizer omits a trainable model parameter"
    ce = nn.CrossEntropyLoss()
    best = {"val_acc": -1.0}
    history = []
    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total = correct = seen = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            raw_logits = model(augment(x))
            loss = ce(raw_logits, y)
            if args.method == "scene_shift":
                shifted = scene_shift(x, sm, ss, tm, ts, args.alpha, clamp=False)
                loss = loss + 0.5 * ce(model(augment(shifted)), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.detach()) * len(y)
            correct += (raw_logits.argmax(1) == y).sum().item()
            seen += len(y)
        model.eval()
        val_loss = val_correct = val_seen = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x.to(device))
                y = y.to(device)
                val_loss += ce(logits, y).item() * len(y)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_seen += len(y)
        row = {
            "epoch": epoch, "train_loss": loss_total / seen, "train_acc": correct / seen,
            "val_loss": val_loss / val_seen, "val_acc": val_correct / val_seen,
            "backbone_lr": float(optimizer.param_groups[0]["lr"]),
            "head_refinement_lr": float(optimizer.param_groups[1]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_acc"] > best["val_acc"]:
            best = dict(row)
            torch.save({"model": model.state_dict(), "best": best, "dataset": args.dataset,
                        "method": args.method, "patch_size": PATCH, "bands": int(source.shape[-1]),
                        "classes": spec["classes"], "scene_shift_alpha": args.alpha if args.method == "scene_shift" else 0.0,
                        "scene_shift_clamp": False, "target_gt_used_for_training_or_selection": False,
                        "attention_refinement_optimized": True}, args.output / "best.pth")
    # First and only target-GT load, after source-val selection is complete.
    checkpoint = torch.load(args.output / "best.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, target, load_target_gt(args.dataset), spec["classes"], device)
    result.update({"dataset": args.dataset, "method": args.method, "split_seed": args.split_seed,
                   "optimization_seed": args.optimization_seed, "best_epoch": best["epoch"],
                   "source_val_accuracy": best["val_acc"], "train_seconds": time.time() - start_time,
                   "target_gt_used_for_training_or_selection": False})
    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update({"backbone": "corrected DAMamba clean wrapper", "patch_size": PATCH,
                   "normalization": "per-scene z-score", "scene_shift_clamp": False,
                   "optimizer": "SGD", "momentum": 0.9, "weight_decay": 5e-4,
                   "scheduler": "LambdaLR: lr*(1+0.0003*step)^(-0.75)",
                   "differential_lr": "backbone=0.1*scheduled multiplier; bottleneck/classifier/CA/SA=1.0*scheduled multiplier",
                   "source_train_per_class": 180, "target_gt_used_for_training_or_selection": False})
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "history.json").write_text(json.dumps(history, indent=2))
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    with (args.output / "per_class.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["class", "accuracy"])
        writer.writerows((i + 1, value) for i, value in enumerate(result["per_class_accuracy"]))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
