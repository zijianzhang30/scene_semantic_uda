"""Strict-MLUDA Houston experiment with JS-weighted OT and flow matching.

This entry is intentionally separate from ``train_houston_v04.py``.  It
executes the server's official ``MLUDA_hu.py`` data/model/evaluation path and
replaces only the adaptation objective with

    source CE + flow matching MSE + 0.1 * rollout semantic CE.

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
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
LEGACY = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(LEGACY))
import config_Houston as cfg
import utils
import UtilsCMS
sys.path.insert(0, str(ROOT))
from class_conditional_flow import (
    ConditionalFlowMLP,
    agreement_flow_matching_loss,
    agreement_flow_rollout,
    compute_source_prototypes,
    sample_ot_pairs,
)

NUM_CLASSES = 7
EPS = 1e-8


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


@torch.no_grad()
def global_source_prototypes(model, train_x, train_y, target_context, shift_stats=None):
    """Compute epoch-global GT source prototypes on the official feature path."""
    was_training = model.training
    model.eval()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    features = []
    try:
        for start in range(0, len(train_x), 32):
            source = torch.as_tensor(train_x[start : start + 32]).cuda()
            if shift_stats is not None:
                source = scene_shift_batch(source, *shift_stats, alpha=0.8)
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
        self.margin_sum = self.correct_margin_sum = self.incorrect_margin_sum = 0.0
        self.correct_n = self.incorrect_n = 0
        self.mass = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.pairs = torch.zeros(NUM_CLASSES, dtype=torch.long)
        self.cost_sum = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.cost_weight = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.source_loss_sum = self.fm_loss_sum = self.semantic_loss_sum = 0.0
        self.pair_distance_sum = self.pred_velocity_sum = self.target_velocity_sum = 0.0
        self.normalized_target_velocity_sum = self.restored_pred_velocity_sum = 0.0
        self.velocity_cosine_sum = self.rollout_distance_sum = self.transport_correct = 0.0
        self.transport_n = 0
        self.flow_grad_from_fm = None
        self.flow_grad_from_semantic = 0.0
        self.source_correct = self.shifted_source_correct = self.source_n = 0
        self.true6_hist = torch.zeros(NUM_CLASSES, dtype=torch.long)
        self.q6_correct = self.s6_correct = self.true6_n = 0
        self.source_target_distance_sum = self.shifted_target_distance_sum = 0.0
        self.source_target_mmd_sum = self.shifted_target_mmd_sum = 0.0
        self.batches = 0

    @torch.no_grad()
    def observe_target(self, q, s, r, agreement, margin, target_weight, labels):
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
        self.margin_sum += float(margin.double().sum())
        self.correct_margin_sum += float(margin[r_correct].double().sum())
        self.incorrect_margin_sum += float(margin[~r_correct].double().sum())
        self.correct_n += int(r_correct.sum())
        self.incorrect_n += int((~r_correct).sum())
        self.mass += (target_weight[:, None] * r).sum(0).double().cpu()
        true6 = labels == 6
        self.true6_n += int(true6.sum())
        self.q6_correct += int((q.argmax(1)[true6] == 6).sum())
        self.s6_correct += int((s.argmax(1)[true6] == 6).sum())
        self.true6_hist += torch.bincount(q.argmax(1)[true6].cpu(), minlength=NUM_CLASSES)

    @torch.no_grad()
    def observe_q_only(self, q, labels):
        """Record classifier-view target diagnostics without constructing OT."""
        labels = labels.to(q.device)
        self.n += len(labels)
        q_pred = q.argmax(1)
        self.q_correct += int((q_pred == labels).sum())
        true6 = labels == 6
        self.true6_n += int(true6.sum())
        self.q6_correct += int((q_pred[true6] == 6).sum())
        self.true6_hist += torch.bincount(q_pred[true6].cpu(), minlength=NUM_CLASSES)

    @torch.no_grad()
    def observe_scene_shift(self, source_logits, shifted_logits, labels, source, shifted, target):
        self.source_correct += int((source_logits.argmax(1) == labels).sum())
        self.shifted_source_correct += int((shifted_logits.argmax(1) == labels).sum())
        self.source_n += len(labels)
        ds = (source.mean(0) - target.mean(0)).norm()
        dss = (shifted.mean(0) - target.mean(0)).norm()
        self.source_target_distance_sum += float(ds)
        self.shifted_target_distance_sum += float(dss)
        self.source_target_mmd_sum += float(ds.square())
        self.shifted_target_mmd_sum += float(dss.square())

    @torch.no_grad()
    def observe_ot(self, ot, pair_classes):
        self.pairs += torch.bincount(pair_classes.cpu(), minlength=NUM_CLASSES)
        for class_id, item in ot.items():
            coupling = item["coupling"]
            weight = float(coupling.sum())
            self.cost_sum[class_id] += float((coupling * item["cost"]).sum())
            self.cost_weight[class_id] += weight

    def observe_losses(self, source_loss, fm_loss, semantic_loss, flow_stats, rollout, target, classes, classifier):
        self.source_loss_sum += float(source_loss.detach())
        self.fm_loss_sum += float(fm_loss.detach())
        self.semantic_loss_sum += float(semantic_loss.detach())
        if flow_stats:
            self.pair_distance_sum += float(flow_stats["pair_distance"])
            self.pred_velocity_sum += float(flow_stats["predicted_normalized_velocity_norm"])
            self.target_velocity_sum += float(flow_stats["original_target_velocity_norm"])
            self.normalized_target_velocity_sum += float(flow_stats["normalized_target_velocity_norm"])
            self.restored_pred_velocity_sum += float(flow_stats["restored_predicted_velocity_norm"])
            self.velocity_cosine_sum += float(flow_stats["velocity_cosine"])
            rollout_distance = float((rollout.detach() - target).norm(dim=1).mean())
            self.rollout_distance_sum += rollout_distance
            self.transport_correct += int((classifier(rollout.detach()).argmax(1) == classes).sum())
            self.transport_n += len(classes)
        self.batches += 1

    def observe_gradient_audit(self, fm_norm):
        if self.flow_grad_from_fm is None:
            self.flow_grad_from_fm = fm_norm

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
            "margin_mean": self.margin_sum / max(self.n, 1),
            "correct_margin_mean": self.correct_margin_sum / max(self.correct_n, 1),
            "wrong_margin_mean": self.incorrect_margin_sum / max(self.incorrect_n, 1),
            "per_class_target_mass_sum": self.mass.tolist(),
            "per_class_ot_pair_count": self.pairs.tolist(),
            "per_class_average_ot_cost": costs,
            "source_loss": self.source_loss_sum / max(self.batches, 1),
            "fm_loss": self.fm_loss_sum / max(self.batches, 1),
            "semantic_loss": self.semantic_loss_sum / max(self.batches, 1),
            "mean_pair_distance": self.pair_distance_sum / max(self.batches, 1),
            "mean_original_target_velocity_norm": self.target_velocity_sum / max(self.batches, 1),
            "mean_normalized_target_velocity_norm": self.normalized_target_velocity_sum / max(self.batches, 1),
            "mean_predicted_normalized_velocity_norm": self.pred_velocity_sum / max(self.batches, 1),
            "mean_restored_predicted_velocity_norm": self.restored_pred_velocity_sum / max(self.batches, 1),
            "mean_velocity_cosine": self.velocity_cosine_sum / max(self.batches, 1),
            "mean_rollout_target_distance": self.rollout_distance_sum / max(self.batches, 1),
            "rollout_distance_reduction_ratio": (
                (self.pair_distance_sum - self.rollout_distance_sum) /
                max(self.pair_distance_sum, 1e-12)
            ),
            "transported_accuracy": self.transport_correct / max(self.transport_n, 1),
            "flow_grad_from_fm": self.flow_grad_from_fm,
            "flow_grad_from_semantic": self.flow_grad_from_semantic,
            "target_observations": self.n,
            "batches": self.batches,
            "source_accuracy": self.source_correct / max(self.source_n, 1),
            "shifted_source_accuracy": self.shifted_source_correct / max(self.source_n, 1),
            "target_q_class6_accuracy": self.q6_correct / max(self.true6_n, 1),
            "target_q_true_class6_to_class5": (
                int(self.true6_hist[5]) / max(self.true6_n, 1)
            ),
            "target_s_ss_class6_accuracy": self.s6_correct / max(self.true6_n, 1),
            "true_class6_q_prediction_histogram": self.true6_hist.tolist(),
            "source_target_mean_feature_distance": self.source_target_distance_sum / max(self.batches, 1),
            "shifted_source_target_mean_feature_distance": self.shifted_target_distance_sum / max(self.batches, 1),
            "source_target_linear_mmd": self.source_target_mmd_sum / max(self.batches, 1),
            "shifted_source_target_linear_mmd": self.shifted_target_mmd_sum / max(self.batches, 1),
        }


def scene_shift_batch(x, source_mean, source_std, target_mean, target_std, alpha=0.8):
    mapped = (x - source_mean) / (source_std + 1e-5) * target_std + target_mean
    return (1.0 - alpha) * x + alpha * mapped


def forward_without_bn_update(model, source, target):
    """Second SceneShift view: retain gradients but do not update BN buffers again."""
    batch_norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    states = [m.training for m in batch_norms]
    try:
        for module in batch_norms:
            module.eval()
        return model(source, target)
    finally:
        for module, training in zip(batch_norms, states):
            module.train(training)


def adaptation(model, flow, source_features, target_features, target_logits, source_labels,
               target_labels, prototypes, valid_classes, epoch, diagnostics, flow_variant,
               weight_mode="js_product"):
    diagnostics.begin(epoch)
    with torch.no_grad():
        q = target_logits.detach().softmax(1)
        s, r, agreement = agreement_membership(
            target_features.detach(), q, prototypes, valid_classes, prototype_temperature=1.0
        )
        top2 = r.topk(2, dim=1).values
        margin = top2[:, 0] - top2[:, 1]
        if weight_mode == "q":
            ot_membership, ot_weight = q, torch.ones_like(agreement)
        elif weight_mode == "mean_qs":
            ot_membership, ot_weight = 0.5 * (q + s), torch.ones_like(agreement)
        elif weight_mode == "s":
            ot_membership, ot_weight = s, torch.ones_like(agreement)
        else:
            ot_membership, ot_weight = r, agreement
        diagnostics.observe_target(q, s, ot_membership, ot_weight, margin, ot_weight, target_labels)
        ot = classwise_sinkhorn_ot_weighted(
            source_features.detach(), source_labels, target_features.detach(), ot_membership, ot_weight,
            NUM_CLASSES, reg=0.05, iterations=100,
        )
        assert all(torch.isfinite(item["coupling"]).all() for item in ot.values())
    pair_source, pair_target, pair_classes = sample_ot_pairs(
        ot, source_features, target_features.detach(), max_pairs_per_class=32
    )
    diagnostics.observe_ot(ot, pair_classes)
    fm_loss = source_features.sum() * 0.0
    semantic_loss = source_features.sum() * 0.0
    rollout = pair_source
    flow_stats = None
    if len(pair_classes):
        fm_loss, flow_stats = agreement_flow_matching_loss(flow, pair_source, pair_target, pair_classes)
        if diagnostics.flow_grad_from_fm is None:
            gradients = torch.autograd.grad(fm_loss, tuple(flow.parameters()), retain_graph=True)
            gradient_norm = torch.stack([gradient.detach().norm() for gradient in gradients]).square().sum().sqrt()
            diagnostics.observe_gradient_audit(float(gradient_norm))
        # Diagnostic only: semantic loss has no graph to flow, classifier, or backbone.
        with torch.no_grad():
            rollout = agreement_flow_rollout(flow, pair_source.detach(), pair_classes, num_steps=4)
            semantic_loss = F.cross_entropy(model.fc1(rollout), pair_classes)
    assert torch.isfinite(fm_loss) and torch.isfinite(semantic_loss)
    return fm_loss, semantic_loss, flow_stats, rollout, pair_target, pair_classes


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
    parser.add_argument("--variant", choices=["flow_a", "flow_b"], default="flow_a")
    parser.add_argument("--weight-mode", choices=["js_product", "q", "mean_qs", "s"], default="js_product")
    parser.add_argument("--scene-shift", action="store_true")
    parser.add_argument("--scene-shift-only", action="store_true",
                        help="Fair SceneShift control: averaged source/shifted-source CE only")
    parser.add_argument("--out", type=Path, default=ROOT / "runs_agreement_transport_1341")
    args = parser.parse_args()
    if args.scene_shift_only and not args.scene_shift:
        parser.error("--scene-shift-only requires --scene-shift")
    if args.mode == "self-test":
        torch.manual_seed(args.seed)
        ns, nt, dim, classes = 19, 13, 11, NUM_CLASSES
        source = torch.randn(ns, dim)
        labels = torch.arange(ns) % classes
        target = torch.randn(nt, dim)
        prototypes, valid = compute_source_prototypes(source, labels, classes)
        q = torch.randn(nt, classes).softmax(1)
        s, r, agreement = agreement_membership(target, q, prototypes, valid)
        margin = r.topk(2, dim=1).values.diff(dim=1).neg().squeeze(1)
        weights = {"js": agreement, "soft": torch.ones_like(agreement), "margin": margin}
        for weight in weights.values():
            ot = classwise_sinkhorn_ot_weighted(source, labels, target, r, weight, classes)
            for item in ot.values():
                assert torch.allclose(item["coupling"].sum(0), item["raw_target_mass"] / item["raw_target_mass"].sum(), atol=2e-3)
        assert torch.allclose(r.sum(1), torch.ones(nt), atol=1e-6)
        assert agreement.min() >= 0 and agreement.max() <= 1
        assert margin.min() >= 0 and margin.max() <= 1
        flow = ConditionalFlowMLP(dim, classes, hidden_dim=dim)
        pair_source, pair_target = source[:8].clone().requires_grad_(), target[:8].clone()
        pair_classes = labels[:8]
        fm_loss, _ = agreement_flow_matching_loss(flow, pair_source, pair_target, pair_classes)
        rollout = agreement_flow_rollout(flow, pair_source, pair_classes, num_steps=4)
        semantic_loss = rollout.detach().square().mean()
        fm_grad = torch.autograd.grad(fm_loss, tuple(flow.parameters()), retain_graph=True)
        assert sum(float(g.norm()) for g in fm_grad) > 0
        fm_loss.backward(retain_graph=True)
        assert pair_source.grad is None, "FM must not update source/backbone features"
        flow.zero_grad(set_to_none=True)
        assert not semantic_loss.requires_grad
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
    source_flat = cached["s"].reshape(-1, cached["s"].shape[-1]).astype(np.float32)
    target_flat = cached["t"].reshape(-1, cached["t"].shape[-1]).astype(np.float32)
    scene_stats = tuple(torch.as_tensor(value)[None, :, None, None].cuda() for value in (
        source_flat.mean(0), source_flat.std(0), target_flat.mean(0), target_flat.std(0)
    ))

    def paired_ilda(source, target, components, radius):
        assert components == 2 and radius == 0.009
        return cached["s"].copy(), cached["t"].copy()

    UtilsCMS.ILDA = paired_ilda
    cfg.seeds = [args.seed]
    cfg.nDataSet = 1
    cfg.epochs = args.epochs
    out = args.out / args.variant
    out.mkdir(exist_ok=True)
    official_path = LEGACY / "MLUDA_hu.py"
    original = official_path.read_text()
    code = original.replace(
        '    print("Training...")',
        '    audit_split(trainX, trainY, testX, testY, feature_encoder)\n    print("Training...")',
    )
    code = code.replace("feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()",
                        "feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()\n    flow = ConditionalFlowMLP(288, CLASS_NUM, hidden_dim=288).cuda()")
    code = code.replace("{'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},",
                        "{'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},\n            {'params': flow.parameters(), 'lr': LEARNING_RATE},")
    code = code.replace("feature_encoder.train()", "feature_encoder.train()\n        flow.train()")
    start = code.index("            # 0\n")
    end = code.index("            # Update parameters", start)
    scene_forward = '''            shifted_data = scene_shift_batch(source_data.cuda(), *scene_stats, alpha=0.8)
            diagnostics.begin(epoch)
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(
                    source_data.cuda(), target_data.cuda())
            (shifted_features, _, _, shifted_outputs, _,
             target_features_ss, _, _, target_outputs_ss, _) = forward_without_bn_update(feature_encoder,
                    shifted_data, target_data.cuda())
            cls_loss = 0.5 * (crossEntropy(source_outputs, source_label.cuda()) + crossEntropy(shifted_outputs, source_label.cuda()))
            diagnostics.observe_scene_shift(source_outputs.detach(), shifted_outputs.detach(), source_label.cuda(),
                                            source_features.detach(), shifted_features.detach(), target_features_ss.detach())
            fm_loss, semantic_loss, flow_stats, rollout, pair_target, pair_classes = adaptation(
                feature_encoder, flow, shifted_features, target_features_ss, target_outputs_ss,
                source_label.cuda(), target_label, prototypes, valid_classes,
                epoch, diagnostics, flow_variant, "q")
'''
    base_forward = '''            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(
                    source_data.cuda(), target_data.cuda())
            cls_loss = crossEntropy(source_outputs, source_label.cuda())
            fm_loss, semantic_loss, flow_stats, rollout, pair_target, pair_classes = adaptation(
                feature_encoder, flow, source_features, target_features, target_outputs,
                source_label.cuda(), target_label, prototypes, valid_classes,
                epoch, diagnostics, flow_variant, weight_mode)
'''
    scene_only_forward = '''            shifted_data = scene_shift_batch(source_data.cuda(), *scene_stats, alpha=0.8)
            diagnostics.begin(epoch)
            (source_features, source1, _, source_outputs, source_out,
             target_features, _, target1, target_outputs, target_out) = feature_encoder(
                    source_data.cuda(), target_data.cuda())
            (shifted_features, _, _, shifted_outputs, _,
             target_features_ss, _, _, target_outputs_ss, _) = forward_without_bn_update(
                    feature_encoder, shifted_data, target_data.cuda())
            cls_loss = 0.5 * (crossEntropy(source_outputs, source_label.cuda()) +
                              crossEntropy(shifted_outputs, source_label.cuda()))
            with torch.no_grad():
                q = target_outputs_ss.detach().softmax(1)
                diagnostics.observe_q_only(q, target_label.cuda())
                diagnostics.observe_scene_shift(
                    source_outputs.detach(), shifted_outputs.detach(), source_label.cuda(),
                    source_features.detach(), shifted_features.detach(), target_features_ss.detach())
            fm_loss = cls_loss.detach() * 0.0
            semantic_loss = cls_loss.detach() * 0.0
            flow_stats = None
            rollout = shifted_features.detach()
            pair_target = target_features_ss.detach()
            pair_classes = source_label.cuda()
            loss = cls_loss
'''
    prototype_prefix = '' if args.scene_shift_only else '''            if i == 1:
                prototypes, valid_classes = global_source_prototypes(
                    feature_encoder, trainX, trainY, target_data.cuda(), scene_stats if scene_shift_enabled else None)
'''
    replacement = prototype_prefix + (scene_only_forward if args.scene_shift_only else (scene_forward if args.scene_shift else base_forward)) + '''
            loss = cls_loss + fm_loss
            diagnostics.observe_losses(cls_loss, fm_loss, semantic_loss, flow_stats,
                                       rollout, pair_target, pair_classes, feature_encoder.fc1)
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
            "seed": args.seed, "epochs": args.epochs, "variant": args.variant,
            "scene_shift": args.scene_shift, "scene_shift_alpha": 0.8 if args.scene_shift else 0.0,
            "scene_shift_only": args.scene_shift_only,
            "target_weight": None if args.scene_shift_only else ("q" if args.scene_shift else args.weight_mode),
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
        "flow_variant": args.variant,
        "weight_mode": args.weight_mode,
        "scene_stats": scene_stats,
        "scene_shift_batch": scene_shift_batch,
        "forward_without_bn_update": forward_without_bn_update,
        "scene_shift_enabled": args.scene_shift,
        "scene_shift_only": args.scene_shift_only,
        "ConditionalFlowMLP": ConditionalFlowMLP,
    }
    (out / "executed.py").write_text(code)
    os.chdir(LEGACY)
    exec(compile(code, str(official_path), "exec"), namespace)
    best_diagnostic = max(history, key=lambda row: row["oa"])
    result = {"epoch": args.epochs, **history[-1], "selection": f"fixed_epoch{args.epochs}",
              "method": "scene_shift_only" if args.scene_shift_only else "flow_transport",
              "scene_shift_only": args.scene_shift_only,
              "diagnostic_best_oa": best_diagnostic["oa"],
              "diagnostic_best_epoch": best_diagnostic["epoch"],
              "epoch100_minus_best_oa": history[-1]["oa"] - best_diagnostic["oa"]}
    result["gpu_max_allocated"] = torch.cuda.max_memory_allocated()
    torch.save({"model": namespace["feature_encoder"].state_dict(), "flow": namespace["flow"].state_dict(), "metrics": result}, out / f"epoch{args.epochs}.pth")
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
