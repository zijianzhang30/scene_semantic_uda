"""Shared own-model training loop for SceneShiftNet-DCRN datasets."""
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

from models.sceneshift_net_dcrn import SceneShiftNetDCRN


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    target: str
    bands: int
    classes: int
    patch_size: int
    ilda_pca: int
    ilda_radius: float
    normalization: str


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected true/false, got {value!r}")


class PatchDataset(Dataset):
    def __init__(self, cube, centers, patch_size, labels=None):
        self.centers = np.asarray(centers, dtype=np.int64)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        half = patch_size // 2
        self.patch_size = patch_size
        self.padded = np.pad(
            np.asarray(cube, dtype=np.float32),
            ((half, half), (half, half), (0, 0)),
            mode="constant",
        )

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, index):
        row, column = self.centers[index]
        patch = self.padded[
            row : row + self.patch_size, column : column + self.patch_size
        ].transpose(2, 0, 1).copy()
        tensor = torch.from_numpy(patch)
        if self.labels is None:
            return tensor
        return tensor, torch.tensor(self.labels[index], dtype=torch.long)


def sample_source(source_gt, classes, samples_per_class, rng, official=False):
    if official:
        # Exact ordering used by official utils.get_sample_data: enumerate
        # nonzero padded-label positions, shuffle each class's index list with
        # global NumPy RNG, concatenate classes, then shuffle once globally.
        half = 0
        padded = np.pad(source_gt, half, mode="constant")
        row, col = np.nonzero(padded)
        train_indices = []
        for class_id in range(1, classes + 1):
            indices = [j for j in range(len(row)) if padded[row[j], col[j]] == class_id]
            np.random.shuffle(indices)
            train_indices.extend(indices[:samples_per_class])
        np.random.shuffle(train_indices)
        centers = np.stack([row[train_indices], col[train_indices]], axis=1)
        labels = padded[row[train_indices], col[train_indices]].astype(np.int64) - 1
        return centers, labels
    centers, labels = [], []
    for class_id in range(1, classes + 1):
        candidates = np.argwhere(source_gt == class_id)
        if len(candidates) < samples_per_class:
            raise RuntimeError(
                f"Class {class_id} has {len(candidates)} source pixels; "
                f"cannot sample {samples_per_class}"
            )
        rng.shuffle(candidates)
        centers.append(candidates[:samples_per_class])
        labels.append(np.full(samples_per_class, class_id - 1, dtype=np.int64))
    centers, labels = np.concatenate(centers), np.concatenate(labels)
    order = rng.permutation(len(centers))
    return centers[order], labels[order]


def evaluate(model, loader, device, classes):
    predictions, labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            logits = model(inputs.to(device))[1]
            predictions.append(logits.argmax(1).cpu().numpy())
            labels.append(target.numpy())
    predictions, labels = np.concatenate(predictions), np.concatenate(labels)
    confusion = metrics.confusion_matrix(labels, predictions, labels=np.arange(classes))
    denominators = confusion.sum(1)
    per_class = np.divide(
        np.diag(confusion), denominators, out=np.zeros(classes, dtype=float), where=denominators > 0
    )
    return {
        "OA": float((predictions == labels).mean() * 100),
        "AA": float(per_class.mean() * 100),
        "Kappa": float(metrics.cohen_kappa_score(labels, predictions) * 100),
        "per_class_accuracy": (per_class * 100).tolist(),
    }


def run(args, spec, load_cubes, output_root, runner_file):
    protocol_matched = getattr(args, "protocol_matched", False)
    if not protocol_matched:
        set_seed(args.seed)
    output = Path(output_root) / args.method / f"seed_{args.seed}"
    if args.epochs != 100:
        ilda_tag = "ilda" if args.use_ilda else "noilda"
        output = Path(str(output) + f"_smoke_{args.epochs}ep_{ilda_tag}_{args.normalization}")
    output.mkdir(parents=True, exist_ok=True)

    source, source_gt, target, target_gt, data_files = load_cubes(args.normalization)
    if args.use_ilda:
        # This is the same MLUDA-origin ILDA preprocessing currently used by
        # the Houston own-model runner; no MLUDA loss/head is imported.
        from UtilsCMS import ILDA

        source, target = ILDA(source, target, spec.ilda_pca, spec.ilda_radius)
    if protocol_matched:
        # Official Pavia calls ILDA before entering the per-seed loop.
        set_seed(args.seed)
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    assert source.shape[-1] == target.shape[-1] == spec.bands

    rng = np.random.RandomState(args.seed)
    source_centers, source_labels = sample_source(
        source_gt, spec.classes, args.source_per_class, rng,
        official=protocol_matched,
    )
    if protocol_matched:
        row, col = np.nonzero(target_gt)
        target_indices = []
        for class_id in range(1, spec.classes + 1):
            indices = [j for j in range(len(row)) if target_gt[row[j], col[j]] == class_id]
            np.random.shuffle(indices)
            target_indices.extend(indices)
        np.random.shuffle(target_indices)
        target_centers = np.stack([row[target_indices], col[target_indices]], axis=1)
        target_eval_centers = target_centers[: (len(target_centers) // args.batch_size) * args.batch_size]
    else:
        target_centers = np.argwhere(target_gt > 0)
        target_eval_centers = target_centers
    target_labels = target_gt[target_eval_centers[:, 0], target_eval_centers[:, 1]].astype(np.int64) - 1

    source_flat, target_flat = source.reshape(-1, spec.bands), target.reshape(-1, spec.bands)
    source_mean, source_std = source_flat.mean(0), source_flat.std(0)
    target_mean, target_std = target_flat.mean(0), target_flat.std(0)

    source_loader = DataLoader(
        PatchDataset(source, source_centers, spec.patch_size, source_labels),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    target_train_loader = DataLoader(
        PatchDataset(target, target_centers, spec.patch_size),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    target_test_loader = DataLoader(
        PatchDataset(target, target_eval_centers, spec.patch_size, target_labels),
        batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    config = {
        "dataset": spec.name,
        "method": args.method,
        "protocol_matched": protocol_matched,
        "source": spec.source,
        "target": spec.target,
        "seed": args.seed,
        "epochs": args.epochs,
        "source_samples_per_class": args.source_per_class,
        "bands": spec.bands,
        "classes": spec.classes,
        "patch_size": spec.patch_size,
        "normalization": args.normalization,
        "normalization_detail": spec.normalization if args.normalization == "loader" else "none; raw cube values",
        "normalization_per_domain": args.normalization == "loader",
        "ilda": args.use_ilda,
        "ilda_pca_guidance_components": spec.ilda_pca if args.use_ilda else None,
        "ilda_radius": spec.ilda_radius if args.use_ilda else None,
        "scene_shift_enabled": args.method == "sceneshift",
        "scene_shift_position": "after loader normalization and optional ILDA; applied to extracted source patches" if args.method == "sceneshift" else None,
        "scene_shift_stats": "all pixels of complete post-preprocessing source/target cubes, per band; no GT mask" if args.method == "sceneshift" else None,
        "scene_shift_type": "pure_affine" if args.method == "sceneshift" else None,
        "alpha": 0.8,
        "random_scale": False,
        "smooth_noise": False,
        "lambda_shift": 0.5,
        "lambda_target": 0.5,
        "pseudo_threshold": 0.9,
        "pseudo_warmup_epochs": 10,
        "optimizer": "SGD" if protocol_matched else "Adam",
        "lr": 0.001,
        "momentum": 0.9 if protocol_matched else None,
        "weight_decay": 5e-4 if protocol_matched else 0.0,
        "batch_size": args.batch_size,
        "official_target_eval_samples": int(len(target_eval_centers)),
        "checkpoint_selection": "final epoch only",
        "target_pool": "GT>0 support mask; target class labels excluded from training",
        "loss_formula": "L_src + 0.5*L_shift + 0.5*L_target" if args.method == "sceneshift" else "L_src + 0.5*L_target",
        "model": "SceneShiftNet-DCRN shared encoder and shared linear classifier",
        "mluda_losses_or_heads": False,
        "runner_sha256": file_sha256(runner_file),
        "common_trainer_sha256": file_sha256(__file__),
        "model_sha256": file_sha256(Path(__file__).parent / "models/sceneshift_net_dcrn.py"),
        "data_sha256": {str(path): file_sha256(path) for path in data_files},
    }
    (output / "config.json").write_text(json.dumps(config, indent=2))
    if args.prepare_only:
        print("PREPARED", output)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SceneShiftNetDCRN(spec.bands, spec.classes, spec.patch_size).to(device)
    if protocol_matched:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    stat_tensors = [
        torch.as_tensor(value, device=device, dtype=torch.float32)[None, :, None, None]
        for value in (source_mean, source_std, target_mean, target_std)
    ]
    sm, ss, tm, ts = stat_tensors
    history = []

    for epoch in range(1, args.epochs + 1):
        if protocol_matched:
            # Official Pavia entry reconstructs the SGD optimizer at every
            # epoch, resetting its momentum state.
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        model.train()
        target_iterator = iter(target_train_loader)
        sums = np.zeros(3, dtype=np.float64)
        correct = np.zeros(2, dtype=np.int64)
        selected = seen = count = 0
        histogram = np.zeros(spec.classes, dtype=np.int64)

        source_batches = enumerate(source_loader)
        for batch_index, (inputs, labels) in source_batches:
            if protocol_matched and batch_index == len(source_loader) - 1:
                # Official entry uses range(1, len_source_loader), omitting
                # the final complete batch each epoch.
                break
            try:
                target_inputs = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_train_loader)
                target_inputs = next(target_iterator)
            inputs, labels, target_inputs = inputs.to(device), labels.to(device), target_inputs.to(device)
            source_logits = model(inputs)[1]
            if args.method == "sceneshift":
                shifted = (inputs - sm) / (ss + 1e-5)
                shifted = shifted * (0.8 * ts + 0.2 * ss) + 0.8 * tm + 0.2 * sm
                shifted_logits = model(shifted)[1]
            else:
                shifted_logits = None
            target_logits = model(target_inputs)[1]
            confidence, pseudo = target_logits.softmax(1).max(1)
            mask = confidence > 0.9 if epoch > 10 else torch.zeros_like(confidence, dtype=torch.bool)
            target_loss = (
                criterion(target_logits[mask], pseudo[mask]) if mask.any() else target_logits.sum() * 0
            )
            source_loss = criterion(source_logits, labels)
            shifted_loss = criterion(shifted_logits, labels) if shifted_logits is not None else source_loss.new_zeros(())
            loss = source_loss + (0.5 * shifted_loss if args.method == "sceneshift" else 0)
            if epoch > 10:
                loss = loss + 0.5 * target_loss
            if not all(torch.isfinite(value) for value in (source_loss, shifted_loss, target_loss, loss)):
                raise RuntimeError(f"NaN/Inf at epoch {epoch}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch = len(labels)
            sums += np.array([source_loss.item(), shifted_loss.item(), target_loss.item()]) * batch
            correct[0] += (source_logits.argmax(1) == labels).sum().item()
            if shifted_logits is not None:
                correct[1] += (shifted_logits.argmax(1) == labels).sum().item()
            selected += int(mask.sum())
            seen += len(target_inputs)
            histogram += np.bincount(pseudo[mask].detach().cpu().numpy(), minlength=spec.classes)
            count += batch

        row = {
            "epoch": epoch,
            "L_src": float(sums[0] / count),
            "L_shift": float(sums[1] / count),
            "L_target": float(sums[2] / count),
            "pseudo_label_coverage": float(selected / seen),
            "pseudo_label_class_histogram": histogram.tolist(),
            "source_accuracy": float(correct[0] / count),
            "shifted_source_accuracy": float(correct[1] / count) if args.method == "sceneshift" else None,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save({"model": model.state_dict(), "seed": args.seed, "config": config}, output / "final.pth")
    (output / "history.json").write_text(json.dumps(history, indent=2))
    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    results = evaluate(model, target_test_loader, device, spec.classes)
    results.update({
        "source_accuracy": history[-1]["source_accuracy"],
        "shifted_source_accuracy": history[-1]["shifted_source_accuracy"],
        "pseudo_label_coverage": history[-1]["pseudo_label_coverage"],
        "pseudo_label_class_histogram": history[-1]["pseudo_label_class_histogram"],
        "seed": args.seed,
        "epoch": args.epochs,
    })
    (output / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
