"""Strict-MLUDA Houston experiment with agreement-aware weighted OT.

This entry is intentionally separate from ``train_houston_v04.py``.  It
executes the server's official ``MLUDA_hu.py`` data/model/evaluation path and
replaces only the adaptation objective with

    source CE + 0.1 * agreement-weighted class-conditional linear bridge CE.

Target labels are passed only to the detached diagnostics.  They never affect
membership, reliability, OT, pair sampling, or gradients.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
LEGACY = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(LEGACY))
import config_Houston as cfg
import utils
import UtilsCMS
sys.path.insert(0, str(ROOT))
from class_conditional_flow_uda import (
    bridge_classification_loss,
    compute_source_prototypes,
    sample_ot_pairs,
)

NUM_CLASSES = 7
EPS = 1e-8


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


@torch.no_grad()
def global_source_prototypes(model, train_x, train_y, target_context):
    """Compute epoch-global GT source prototypes on the official feature path."""
    was_training = model.training
    model.eval()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    features = []
    try:
        for start in range(0, len(train_x), 32):
            source = torch.as_tensor(train_x[start : start + 32]).cuda()
            features.append(model(source, target_context[: len(source)])[0])
        assert all(torch.equal(buffers[name], value) for name, value in model.named_buffers())
        features = torch.cat(features)
        labels = torch.as_tensor(train_y).cuda()
        return compute_source_prototypes(features, labels, NUM_CLASSES)
    finally:
        model.train(was_training)


def agreement_membership(
    target_features: torch.Tensor,
    classifier_probabilities: torch.Tensor,
    source_prototypes: torch.Tensor,
    valid_classes: torch.Tensor | None = None,
    prototype_temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return prototype probabilities s, product-of-experts r, and JS reliability a."""
    cosine = F.normalize(target_features, dim=1) @ F.normalize(source_prototypes, dim=1).t()
    if valid_classes is not None:
        cosine = cosine.masked_fill(~valid_classes[None, :], -torch.finfo(cosine.dtype).max)
    s = F.softmax(cosine / prototype_temperature, dim=1)
    q = classifier_probabilities
    r = q * s
    r = r / r.sum(dim=1, keepdim=True).clamp_min(EPS)
    midpoint = 0.5 * (q + s)
    js = 0.5 * (q * (q.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(1)
    js += 0.5 * (s * (s.clamp_min(EPS).log() - midpoint.clamp_min(EPS).log())).sum(1)
    agreement = (1.0 - js / np.log(2.0)).clamp(0.0, 1.0)
    return s.detach(), r.detach(), agreement.detach()


@torch.no_grad()
def classwise_sinkhorn_ot_weighted(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    soft_membership: torch.Tensor,
    agreement: torch.Tensor,
    num_classes: int,
    reg: float = 0.05,
    iterations: int = 100,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Class-wise cosine OT with uniform source and normalized a_j r_j^c target mass."""
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    for class_id in range(num_classes):
        source_mask = source_labels == class_id
        if not source_mask.any():
            continue
        raw_target_mass = agreement * soft_membership[:, class_id]
        if raw_target_mass.sum() < EPS:  # Numerical guard, not a reliability gate.
            continue
        source = source_features[source_mask]
        cost = 1.0 - F.normalize(source, dim=1) @ F.normalize(target_features, dim=1).t()
        source_mass = source.new_full((len(source),), 1.0 / len(source))
        target_mass = raw_target_mass / raw_target_mass.sum().clamp_min(EPS)
        kernel = torch.exp(-cost / reg).clamp_min(torch.finfo(cost.dtype).tiny)
        u = torch.ones_like(source_mass)
        v = torch.ones_like(target_mass)
        for _ in range(iterations):
            u = source_mass / (kernel @ v).clamp_min(1e-12)
            v = target_mass / (kernel.t() @ u).clamp_min(1e-12)
        coupling = u[:, None] * kernel * v[None, :]
        result[class_id] = {
            "coupling": coupling,
            "source_indices": source_mask.nonzero(as_tuple=False).flatten(),
            "target_indices": torch.arange(len(target_features), device=target_features.device),
            "cost": cost,
            "raw_target_mass": raw_target_mass,
        }
    return result


class EpochDiagnostics:
    def __init__(self):
        self.epoch = 0

    def begin(self, epoch: int):
        if epoch == self.epoch:
            return
        assert epoch == self.epoch + 1
        self.epoch = epoch
        self.n = self.q_correct = self.s_correct = self.r_correct = 0
        self.a_sum = self.a_sq_sum = 0.0
        self.correct_a_sum = self.incorrect_a_sum = 0.0
        self.correct_n = self.incorrect_n = 0
        self.mass = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.pairs = torch.zeros(NUM_CLASSES, dtype=torch.long)
        self.cost_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.cost_weight = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.source_loss_sum = self.bridge_loss_sum = 0.0
        self.batches = 0

    @torch.no_grad()
    def observe_target(self, q, s, r, agreement, labels):
        labels = labels.to(q.device)
        self.n += len(labels)
        self.q_correct += int((q.argmax(1) == labels).sum())
        self.s_correct += int((s.argmax(1) == labels).sum())
        r_correct = r.argmax(1) == labels
        self.r_correct += int(r_correct.sum())
        a = agreement.double()
        self.a_sum += float(a.sum())
        self.a_sq_sum += float((a * a).sum())
        self.correct_a_sum += float(a[r_correct].sum())
        self.incorrect_a_sum += float(a[~r_correct].sum())
        self.correct_n += int(r_correct.sum())
        self.incorrect_n += int((~r_correct).sum())
        self.mass += (agreement[:, None] * r).sum(0).double().cpu()

    @torch.no_grad()
    def observe_ot(self, ot, pair_classes):
        self.pairs += torch.bincount(pair_classes.cpu(), minlength=NUM_CLASSES)
        for class_id, item in ot.items():
            coupling = item["coupling"]
            weight = float(coupling.sum())
            self.cost_sum[class_id] += float((coupling * item["cost"]).sum())
            self.cost_weight[class_id] += weight

    def observe_losses(self, source_loss, bridge_loss):
        self.source_loss_sum += float(source_loss.detach())
        self.bridge_loss_sum += float(bridge_loss.detach())
        self.batches += 1

    def summary(self):
        mean = self.a_sum / max(self.n, 1)
        variance = max(0.0, self.a_sq_sum / max(self.n, 1) - mean * mean)
        costs = [
            float(self.cost_sum[c] / self.cost_weight[c]) if self.cost_weight[c] else None
            for c in range(NUM_CLASSES)
        ]
        return {
            "q_accuracy": self.q_correct / max(self.n, 1),
            "s_accuracy": self.s_correct / max(self.n, 1),
            "r_accuracy": self.r_correct / max(self.n, 1),
            "agreement_mean": mean,
            "agreement_std": variance**0.5,
            "agreement_correct_mean": self.correct_a_sum / max(self.correct_n, 1),
            "agreement_incorrect_mean": self.incorrect_a_sum / max(self.incorrect_n, 1),
            "per_class_target_mass_sum": self.mass.tolist(),
            "per_class_ot_pair_count": self.pairs.tolist(),
            "per_class_average_ot_cost": costs,
            "source_loss": self.source_loss_sum / max(self.batches, 1),
            "bridge_loss": self.bridge_loss_sum / max(self.batches, 1),
            "target_observations": self.n,
            "batches": self.batches,
        }


def adaptation(model, source_features, target_features, target_logits, source_labels,
               target_labels, prototypes, valid_classes, epoch, diagnostics):
    diagnostics.begin(epoch)
    with torch.no_grad():
        q = target_logits.detach().softmax(1)
        s, r, agreement = agreement_membership(
            target_features.detach(), q, prototypes, valid_classes, prototype_temperature=1.0
        )
        diagnostics.observe_target(q, s, r, agreement, target_labels)
        ot = classwise_sinkhorn_ot_weighted(
            source_features.detach(), source_labels, target_features.detach(), r, agreement,
            NUM_CLASSES, reg=0.05, iterations=100,
        )
        assert all(torch.isfinite(item["coupling"]).all() for item in ot.values())
    pair_source, pair_target, pair_classes = sample_ot_pairs(
        ot, source_features, target_features.detach(), max_pairs_per_class=32
    )
    diagnostics.observe_ot(ot, pair_classes)
    bridge_loss = source_features.sum() * 0.0
    if len(pair_classes):
        tau_max = 0.3 if epoch <= 20 else 0.5 if epoch <= 50 else 0.8
        bridge_loss = bridge_classification_loss(
            model.fc1, pair_source, pair_target, pair_classes, tau_max
        )
    assert torch.isfinite(bridge_loss)
    return bridge_loss


def metrics_from_predictions(predictions, labels):
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[truth, prediction] += 1
    per_class = np.divide(
        np.diag(confusion), confusion.sum(1), out=np.zeros(NUM_CLASSES), where=confusion.sum(1) > 0
    )
    oa = float(np.trace(confusion) / confusion.sum())
    expected = float((confusion.sum(0) * confusion.sum(1)).sum() / confusion.sum() ** 2)
    return {
        "oa": oa,
        "aa": float(per_class.mean()),
        "kappa": float((oa - expected) / (1.0 - expected + 1e-12)),
        "per_class_accuracy": per_class.tolist(),
        "evaluated_n": int(confusion.sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "train", "self-test"], default="train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--out", type=Path, default=ROOT / "runs_agreement_transport_1341")
    args = parser.parse_args()
    if args.mode == "self-test":
        torch.manual_seed(args.seed)
        ns, nt, dim, classes = 19, 13, 11, NUM_CLASSES
        source = torch.randn(ns, dim)
        labels = torch.arange(ns) % classes
        target = torch.randn(nt, dim)
        prototypes, valid = compute_source_prototypes(source, labels, classes)
        q = torch.randn(nt, classes).softmax(1)
        s, r, agreement = agreement_membership(target, q, prototypes, valid)
        ot = classwise_sinkhorn_ot_weighted(source, labels, target, r, agreement, classes)
        assert torch.allclose(r.sum(1), torch.ones(nt), atol=1e-6)
        assert agreement.min() >= 0 and agreement.max() <= 1
        for item in ot.values():
            assert torch.allclose(item["coupling"].sum(0), item["raw_target_mass"] / item["raw_target_mass"].sum(), atol=2e-3)
        print(json.dumps({"self_test": "ok", "classes_with_ot": sorted(ot), "r_row_sum_max_error": float((r.sum(1)-1).abs().max())}))
        return

    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "ilda.npz"
    if args.mode == "prepare":
        if cache.exists():
            raise FileExistsError("Refusing to overwrite paired ILDA preprocessing")
        source, _ = utils.load_data_houston(str(LEGACY / "datasets/Houston/Houston13.mat"), str(LEGACY / "datasets/Houston/Houston13_7gt.mat"))
        target, _ = utils.load_data_houston(str(LEGACY / "datasets/Houston/Houston18.mat"), str(LEGACY / "datasets/Houston/Houston18_7gt.mat"))
        source, target = UtilsCMS.ILDA(source, target, cfg.pca_n, cfg.radius)
        np.savez(cache, s=source, t=target)
        return
    if not cache.exists():
        shared_cache = ROOT / "runs_strict_mluda_1341" / "ilda.npz"
        if not shared_cache.exists():
            raise FileNotFoundError(f"Run --mode prepare first; missing {cache}")
        cache = shared_cache

    cached = np.load(cache)

    def paired_ilda(source, target, components, radius):
        assert components == 2 and radius == 0.009
        return cached["s"].copy(), cached["t"].copy()

    UtilsCMS.ILDA = paired_ilda
    cfg.seeds = [args.seed]
    cfg.nDataSet = 1
    cfg.epochs = args.epochs
    out = args.out / "agreement_transport"
    out.mkdir(exist_ok=True)
    official_path = LEGACY / "MLUDA_hu.py"
    original = official_path.read_text()
    code = original.replace(
        '    print("Training...")',
        '    audit_split(trainX, trainY, testX, testY, feature_encoder)\n    print("Training...")',
    )
    start = code.index("            # 0\n")
    end = code.index("            # Update parameters", start)
    replacement = '''            if i == 1:
                prototypes, valid_classes = global_source_prototypes(
                    feature_encoder, trainX, trainY, target_data.cuda())
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(
                    source_data.cuda(), target_data.cuda())
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            bridge_loss = adaptation(
                feature_encoder, source_features, target_features, target_outputs,
                source_label.cuda(), target_label, prototypes, valid_classes,
                epoch, diagnostics)
            loss = cls_loss + 0.1 * bridge_loss
            diagnostics.observe_losses(cls_loss, bridge_loss)
            # Values retained only for the unchanged official progress formatter.
            lmmd_loss = contrastive_loss_s = contrastive_loss_t = loss.detach() * 0

'''
    code = code[:start] + replacement + code[end:]
    code = code.replace(
        "        train_end = time.time()",
        "        audit_epoch(epoch, feature_encoder, source_data, test_loader)\n        train_end = time.time()",
    )

    history = []
    diagnostics = EpochDiagnostics()

    def audit_split(train_x, train_y, test_x, test_y, model):
        manifest = {
            "source_x": sha(train_x), "source_y": sha(train_y),
            "target_x": sha(test_x), "target_y": sha(test_y),
            "initial_model": hashlib.sha256(b"".join(
                value.detach().cpu().numpy().tobytes() for value in model.state_dict().values()
            )).hexdigest(),
            "seed": args.seed, "epochs": args.epochs,
            "source_counts": np.bincount(train_y).tolist(),
            "source_n": len(train_y), "target_n": len(test_y),
            "official_file_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "base_script": str(official_path),
        }
        (out / "split.json").write_text(json.dumps(manifest, indent=2))
        print("SPLIT", json.dumps(manifest), flush=True)

    @torch.no_grad()
    def audit_epoch(epoch, model, source_context, test_loader):
        was_training = model.training
        model.eval()
        predictions, labels = [], []
        for target_data, target_label in test_loader:
            outputs = model(source_context.cuda(), target_data.cuda())[8]
            predictions.extend(outputs.argmax(1).cpu().tolist())
            labels.extend(target_label.tolist())
        model.train(was_training)
        row = {"epoch": epoch, **diagnostics.summary(), **metrics_from_predictions(np.asarray(predictions), np.asarray(labels))}
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print("AGREEMENT_EPOCH", json.dumps(row), flush=True)

    namespace = {
        "__name__": "__main__",
        "audit_split": audit_split,
        "audit_epoch": audit_epoch,
        "global_source_prototypes": global_source_prototypes,
        "adaptation": adaptation,
        "diagnostics": diagnostics,
    }
    (out / "executed.py").write_text(code)
    os.chdir(LEGACY)
    exec(compile(code, str(official_path), "exec"), namespace)
    result = {"epoch": args.epochs, **history[-1], "selection": f"fixed_epoch{args.epochs}"}
    result["gpu_max_allocated"] = torch.cuda.max_memory_allocated()
    torch.save({"model": namespace["feature_encoder"].state_dict(), "metrics": result}, out / f"epoch{args.epochs}.pth")
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
