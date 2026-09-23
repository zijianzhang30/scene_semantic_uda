"""Houston A-G ablations on the unchanged official MLUDA data/model protocol.

Only original-source CE, auxiliary SceneShift CE and class-wise cosine OT/FM
are optimized. Target labels are used solely for detached diagnostics/evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import sys
from typing import Dict, Tuple

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    compute_source_prototypes,
    sample_ot_pairs,
)

from source_val_protocol import (validation_data, evaluate_source, evaluate_target,
                                 isolated_evaluation, atomic_save, improves_source_validation)

NUM_CLASSES = 7
EPS = 1e-8


def sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


class _DeterministicGlobalPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.shape = value.shape
        return F.avg_pool3d(value, (1, value.shape[-2], value.shape[-1]))

    @staticmethod
    def backward(ctx, gradient):
        # Global, non-overlapping pooling: every input receives exactly one term.
        area = gradient.new_tensor(ctx.shape[-2] * ctx.shape[-1])
        return (gradient / area).expand(ctx.shape).contiguous()


class _DeterministicGlobalMaxPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        result, indices = F.adaptive_max_pool2d(value, 1, return_indices=True)
        ctx.shape = value.shape
        ctx.save_for_backward(indices)
        return result

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        positions = torch.arange(ctx.shape[-2] * ctx.shape[-1], device=gradient.device)
        mask = positions.reshape(1, 1, *ctx.shape[-2:]) == indices
        return gradient * mask


def install_deterministic_pool(model):
    pool = model.feature_layers.avgpool
    assert isinstance(pool, nn.AvgPool3d)
    assert pool.kernel_size == (1, model.feature_layers.sz, model.feature_layers.sz)
    # Preserve the official forward kernel and state_dict; replace only backward.
    pool.forward = _DeterministicGlobalPool.apply
    max_pool = model.feature_layers.ca.max_pool
    assert isinstance(max_pool, nn.AdaptiveMaxPool2d) and max_pool.output_size == 1
    max_pool.forward = _DeterministicGlobalMaxPool.apply
    return model


def tensor_sequence_hash(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_hash(model) -> str:
    return hashlib.sha256(b"".join(
        p.detach().cpu().numpy().tobytes() for p in model.parameters()
    )).hexdigest()


@torch.no_grad()
def classwise_cosine_ot(
    source_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_features: torch.Tensor,
    soft_membership: torch.Tensor,
    num_classes: int,
    reg: float = 0.05,
    iterations: int = 100,
    target_mask: torch.Tensor | None = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Class-wise cosine OT with uniform source and normalized filtered soft q target mass."""
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    if target_mask is None:
        target_mask = torch.ones(len(target_features), dtype=torch.bool, device=target_features.device)
    for class_id in range(num_classes):
        source_mask = source_labels == class_id
        if not source_mask.any():
            continue
        raw_target_mass = soft_membership[:, class_id].masked_fill(~target_mask, 0.0)
        if raw_target_mass.sum() < EPS:  # Numerical guard, not a reliability gate.
            continue
        source = source_features[source_mask]
        feature_cost = 1.0 - F.normalize(source, dim=1) @ F.normalize(target_features, dim=1).t()
        cost = feature_cost
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

def scene_shift_batch(x, source_mean, source_std, target_mean, target_std, alpha=0.8):
    mapped = (x - source_mean) / (source_std + 1e-5) * target_std + target_mean
    return (1.0 - alpha) * x + alpha * mapped


def forward_without_bn_update(model, source, target):
    """Historical running-statistics control, retained only for B_loss."""
    batch_norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    states = [m.training for m in batch_norms]
    try:
        for module in batch_norms:
            module.eval()
        return model(source, target)
    finally:
        for module, training in zip(batch_norms, states):
            module.train(training)


def forward_with_batch_stats_restore_buffers(model, source, target):
    """Use auxiliary batch statistics without retaining BN running updates."""
    saved = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            buffers = {name: None if getattr(module, name) is None else
                       getattr(module, name).detach().clone()
                       for name in ("running_mean", "running_var", "num_batches_tracked")}
            saved.append((module, module.training, buffers))
    try:
        for module, _, _ in saved:
            module.train()
        return model(source, target)
    finally:
        for module, mode, buffers in saved:
            # Rebind instead of copy_: backward may still reference forward buffers.
            for name, value in buffers.items():
                setattr(module, name, value)
            module.training = mode



# (auxiliary SceneShift CE, OT/FM, source-side FM backprop, reliability filtering)
ABLATIONS = {
    "A": (False, False, False, False),
    "B": (True, False, False, False),
    "C": (True, True, False, False),
    "D": (True, True, True, False),
    "E": (True, True, True, True),
    # Isolate OT/Flow from SceneShift.
    "F": (False, True, False, False),
    "G": (False, True, True, False),
    "B_BN": (True, False, False, False),  # Alias of corrected B; preserves the screen name.
    "B_loss": (True, False, False, False),
}


@torch.no_grad()
def target_membership(source, labels, target, logits, reliability):
    """Batch original-source prototypes only; return soft q and a sample mask."""
    q = logits.detach().softmax(1)
    keep = torch.ones(len(q), device=q.device, dtype=torch.bool)
    if reliability:
        prototypes, valid = compute_source_prototypes(source.detach(), labels, NUM_CLASSES)
        cosine = F.normalize(target.detach(), dim=1) @ F.normalize(prototypes, dim=1).t()
        cosine = cosine.masked_fill(~valid[None, :], -torch.finfo(cosine.dtype).max)
        keep = q.argmax(1).eq(cosine.argmax(1)) & valid[q.argmax(1)]
    return q, keep


def adaptation(flow, source, target, logits, labels, backprop, reliability, flow_rng):
    if not isinstance(flow_rng, torch.Generator):
        raise TypeError("OT/Flow requires an explicit torch.Generator")
    q, keep = target_membership(source, labels, target, logits, reliability)
    ot = classwise_cosine_ot(source, labels, target, q, NUM_CLASSES, target_mask=keep)
    pair_source, pair_target, pair_classes = sample_ot_pairs(
        ot, source, target.detach(), max_pairs_per_class=32,
        generator=flow_rng)
    fm = source.new_zeros(())
    stats = None
    if len(pair_classes):
        fm, stats = agreement_flow_matching_loss(
            flow, pair_source, pair_target, pair_classes, detach_source=not backprop,
            generator=flow_rng)
    assert torch.isfinite(fm)
    return fm, {"q": q, "keep": keep, "ot": ot, "pair_classes": pair_classes, "stats": stats}


def training_step(model, flow, source_data, target_data, labels, scene_stats, ablation, flow_rng):
    """One common original-source/target forward for every ablation."""
    shift, use_flow, backprop, reliability = ABLATIONS[ablation]
    outputs = model(source_data, target_data)
    source, target = outputs[0], outputs[5]
    ce_s = F.cross_entropy(outputs[3], labels)
    ce_ss = None
    if shift:
        shifted = scene_shift_batch(source_data, *scene_stats, alpha=0.8)
        # Auxiliary dropout must not advance the original-view RNG stream.
        with torch.random.fork_rng(devices=[source_data.device.index]):
            auxiliary_forward = (forward_with_batch_stats_restore_buffers
                                 if ablation != "B_loss" else forward_without_bn_update)
            shifted_outputs = auxiliary_forward(model, shifted, target_data)
        ce_ss = F.cross_entropy(shifted_outputs[3], labels)
    ce = ce_s if ce_ss is None else 0.5 * (ce_s + ce_ss)
    if ablation == "B_loss":
        ce = ce_s + 0.5 * ce_ss
    fm = ce.new_zeros(())
    details = None
    if use_flow:
        fm, details = adaptation(flow, source, target, outputs[8], labels, backprop, reliability, flow_rng)
    return {"loss": ce + fm, "ce": ce, "ce_s": ce_s, "ce_ss": ce_ss, "fm": fm,
            "source_logits": outputs[3], "target_logits": outputs[8],
            "source_features": source, "target_features": target, "details": details}


class EpochDiagnostics:
    def __init__(self):
        self.epoch = 0

    def begin(self, epoch):
        if epoch == self.epoch:
            return
        assert epoch == self.epoch + 1
        self.epoch = epoch
        self.batches = self.target_n = self.correct = self.kept = 0
        self.ce = self.ce_s = self.ce_ss = self.fm = 0.0
        self.mass = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.pairs = torch.zeros(NUM_CLASSES, dtype=torch.long)
        self.cost = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.cost_weight = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.source_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
        self.missing = torch.zeros(NUM_CLASSES, dtype=torch.long)
        self.flow_grad = self.source_grad = None

    def observe(self, epoch, step, source_labels, target_labels, flow):
        self.begin(epoch)
        self.batches += 1
        self.ce += float(step['ce'].detach())
        self.ce_s += float(step['ce_s'].detach())
        self.ce_ss += float(step['ce_ss'].detach()) if step['ce_ss'] is not None else 0.0
        self.fm += float(step['fm'].detach())
        with torch.no_grad():
            q = step['target_logits'].softmax(1)
            self.target_n += len(q)
            self.correct += int((q.argmax(1) == target_labels.to(q.device)).sum())
            d = step['details']
            if d is not None:
                self.kept += int(d['keep'].sum())
                self.mass += (d['q'] * d['keep'][:, None]).sum(0).double().cpu()
                self.pairs += torch.bincount(d['pair_classes'].cpu(), minlength=NUM_CLASSES)
                counts = torch.bincount(source_labels.cpu(), minlength=NUM_CLASSES)
                self.source_counts += counts
                self.missing += counts.eq(0).long()
                for c, item in d['ot'].items():
                    self.cost[c] += float((item['coupling'] * item['cost']).sum())
                    self.cost_weight[c] += float(item['coupling'].sum())
        if self.flow_grad is None and step['fm'].requires_grad:
            variables = (step['source_features'],) + tuple(flow.parameters())
            gradients = torch.autograd.grad(step['fm'], variables, retain_graph=True, allow_unused=True)
            self.source_grad = float(gradients[0].norm()) if gradients[0] is not None else 0.0
            self.flow_grad = sum(float(g.detach().square().sum()) for g in gradients[1:] if g is not None)**0.5

    def summary(self):
        return {"source_loss": self.ce / self.batches, "ce_s": self.ce_s / self.batches,
                "ce_ss": self.ce_ss / self.batches, "fm_loss": self.fm / self.batches,
                "q_accuracy": self.correct / max(self.target_n, 1),
                "reliable_ratio": self.kept / max(self.target_n, 1),
                "per_class_target_mass_sum": self.mass.tolist(),
                "per_class_ot_pair_count": self.pairs.tolist(),
                "per_class_average_ot_cost": [float(self.cost[c]/self.cost_weight[c]) if self.cost_weight[c] else None for c in range(NUM_CLASSES)],
                "prototype_source_counts": self.source_counts.tolist(),
                "prototype_missing_batch_fraction": (self.missing.double()/self.batches).tolist(),
                "flow_grad_from_fm": self.flow_grad, "source_feature_grad_from_fm": self.source_grad,
                "batches": self.batches}


def build_training_code(original):
    """Replace only model Flow attachment, objective, and detached audit hooks."""
    def replace_once(code, old, new):
        assert code.count(old) == 1, f"Official script anchor changed: {old}"
        return code.replace(old, new, 1)
    code = replace_once(original, '    print("Training...")',
                        '    audit_split(trainX, trainY, testX, testY, feature_encoder)\n    print("Training...")')
    code = replace_once(code, 'utils.set_seed(seeds[iDataSet])',
                        'utils.set_seed(seeds[iDataSet])\n'
                        '    flow_rng = torch.Generator(device="cpu")\n'
                        '    flow_rng.manual_seed(args_flow["sampling_seed"])')
    # Initialize the auxiliary Flow without advancing the official RNG stream.
    code = replace_once(code, 'feature_encoder = DSANSS(nBand, patch_size, CLASS_NUM).cuda()',
                        'feature_encoder = install_deterministic_pool(DSANSS(nBand, patch_size, CLASS_NUM).cuda())\n'
                        '    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):\n'
                        '        torch.random.default_generator.manual_seed(args_flow["init_seed"])\n'
                        '        torch.cuda.manual_seed(args_flow["init_seed"])\n'
                        '        flow = ConditionalFlowMLP(288, CLASS_NUM, hidden_dim=288).cuda()')
    code = replace_once(code, "{'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},",
                        "{'params': feature_encoder.head2.parameters(), 'lr': LEARNING_RATE},\n            {'params': flow.parameters(), 'lr': LEARNING_RATE},")
    code = replace_once(code, 'feature_encoder.train()', 'feature_encoder.train()\n        flow.train()')
    code = replace_once(code, '            optimizer.step()',
                        '            optimizer.step()\n'
                        '            audit_step(epoch, i, feature_encoder, step, source_data, target_data, source_label)')
    code = replace_once(code, '        iter_source = iter(train_loader_s)',
                        '        flow_rng = torch.Generator(device="cpu")\n'
                        '        flow_rng.manual_seed(args_flow["sampling_seed"] + epoch)\n'
                        '        iter_source = iter(train_loader_s)')
    start = code.index('            # 0\n')
    end = code.index('            # Update parameters', start)
    step = '''            step = training_step(feature_encoder, flow, source_data.cuda(),
                                 target_data.cuda(), source_label.cuda(), scene_stats, ablation, flow_rng)
            loss, cls_loss = step['loss'], step['ce']
            source_outputs = step['source_logits']
            diagnostics.observe(epoch, step, source_label, target_label, flow)
            # Retain the official progress formatter without extra loss terms.
            lmmd_loss = contrastive_loss_s = contrastive_loss_t = loss.detach() * 0

'''
    code = code[:start] + step + code[end:]
    code = replace_once(code, '        train_end = time.time()',
                        '        audit_epoch(epoch, feature_encoder, source_data, test_loader)\n        train_end = time.time()')
    compile(code, 'executed.py', 'exec')
    return code



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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr-horizon", type=int, default=None,
                        help="Keep the full-run LR schedule during short screening runs")
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--variant", choices=["flow_a"], default="flow_a")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="C",
                        help="A=CE_s, B=SceneShift, C=OT/Flow, D=source FM backprop, E=soft reliability")
    parser.add_argument("--out", type=Path, default=ROOT / "runs_agreement_transport_1341")
    parser.add_argument("--selection", choices=["source_val_best", "fixed_epoch"],
                        default="source_val_best")
    args = parser.parse_args()
    if args.lr_horizon is None:
        args.lr_horizon = args.epochs
    if not 1 <= args.epochs <= args.lr_horizon:
        parser.error("Require 1 <= epochs <= lr-horizon")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    scene_shift, use_flow, backprop, reliability = ABLATIONS[args.ablation]
    args.scene_shift = scene_shift
    args.scene_shift_only = not use_flow and scene_shift
    args.scene_shift_aux_original_flow = use_flow
    args.fm_backprop_source = backprop
    args.reliability_routing = reliability
    if args.mode == "self-test":
        torch.manual_seed(args.seed)
        model = nn.Linear(8, NUM_CLASSES)
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        flow = ConditionalFlowMLP(8, NUM_CLASSES, hidden_dim=16)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        flow_rng = torch.Generator(device="cpu").manual_seed(args.seed + 99173)
        source = torch.randn(12, 8, requires_grad=True)
        target = torch.randn(12, 8)
        labels = torch.arange(12) % NUM_CLASSES
        logits = model(source)
        fm, details = adaptation(flow, source, target, logits, labels, args.fm_backprop_source, args.reliability_routing, flow_rng)
        if not ABLATIONS[args.ablation][1]:
            assert args.ablation in ("A", "B", "B_BN", "B_loss")
            fm = fm.detach() * 0.0
        ce = F.cross_entropy(logits, labels)
        loss = ce + fm
        loss.backward()
        assert any(p.grad is not None for p in model.parameters())
        assert details['q'].shape == (12, NUM_CLASSES)
        if args.fm_backprop_source:
            assert source.grad is not None and source.grad.norm() > 0
        else:
            assert source.grad is not None  # CE path remains active
        print(json.dumps({"self_test": "ok", "ablation": args.ablation, "fm_requires_grad": fm.requires_grad, "kept": int(details["keep"].sum())}))
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
    if any(out.iterdir()):
        raise FileExistsError(f"Refusing to mix audit records in nonempty {out}")
    official_path = LEGACY / "MLUDA_hu.py"
    original = official_path.read_text()
    flow_init_seed = args.seed + 71039
    flow_sampling_seed = args.seed + 99173
    code = original
    # The generated official loop calls the common training_step for every A-G.
    replacement = build_training_code(code)
    code = replacement.replace("(epoch - 1) / epochs", "(epoch - 1) / lr_horizon")
    history = []
    diagnostics = EpochDiagnostics()
    validation = {}
    selected_row = None

    def audit_split(train_x, train_y, test_x, test_y, model):
        if args.selection == "source_val_best":
            _, source_gt = utils.load_data_houston(
                str(LEGACY / "datasets/Houston/Houston13.mat"),
                str(LEGACY / "datasets/Houston/Houston13_7gt.mat"))
            _, target_gt = utils.load_data_houston(
                str(LEGACY / "datasets/Houston/Houston18.mat"),
                str(LEGACY / "datasets/Houston/Houston18_7gt.mat"))
            loader, reference, split = validation_data(
                cached["s"], source_gt, args.seed, train_x, train_y)
            np.savez(out / "source_validation_split.npz", **split)
            validation.update(loader=loader, reference=reference, target_gt=target_gt)
            (out / "selection_protocol.json").write_text(json.dumps({
                "selection": args.selection, "metric": "source validation OA",
                "tie_rule": "earliest epoch", "source_forward": "model(x,x)[3]",
                "source_train_n": len(train_y),
                "source_validation_n": len(loader.dataset),
                "source_validation_labels_hash": sha(split["validation_labels"]),
                "source_validation_centers_hash": sha(split["validation_centers"]),
                "target_forward": "model(train_x[:32], target_batch)[8]",
                "target_order": "row-major labelled centers",
                "target_drop_last": False,
                "target_evaluated_n": int((target_gt > 0).sum()),
                "target_metrics_used_for_selection": False,
                "historical_reference": "4b467cba:train_full_mluda_scene_shift.py",
                "training_protocol": "current A-G; not historical full-MLUDA training",
                "lr_horizon": args.lr_horizon,
                "protocol_version": "sourceval_bn_batch_restore_v2",
                "auxiliary_bn": ("none" if not args.scene_shift else
                                 "running_stats" if args.ablation == "B_loss" else
                                 "batch_stats_restore_buffers"),
                "ce_weights": ([1.0, 0.0] if not args.scene_shift else
                               [1.0, 0.5] if args.ablation == "B_loss" else [0.5, 0.5]),
            }, indent=2))
        manifest = {
            "source_x": sha(train_x), "source_y": sha(train_y),
            "target_x": sha(test_x), "target_y": sha(test_y),
            "initial_model": hashlib.sha256(b"".join(
                value.detach().cpu().numpy().tobytes() for value in model.state_dict().values()
            )).hexdigest(),
            "seed": args.seed, "epochs": args.epochs, "variant": args.variant,
            "scene_shift": args.scene_shift, "scene_shift_alpha": 0.8 if args.scene_shift else 0.0,
            "scene_shift_only": args.scene_shift_only,
            "scene_shift_aux_original_flow": args.scene_shift_aux_original_flow,
            "fm_backprop_source": args.fm_backprop_source,
            "reliability_routing": args.reliability_routing,
            "prototype_mode": "batch_original_source" if args.reliability_routing else "unused",
            "target_weight": "q" if not args.reliability_routing else "filtered_soft_q",
            "source_counts": np.bincount(train_y).tolist(),
            "source_n": len(train_y), "target_n": len(test_y),
            "official_file_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "base_script": str(official_path),
        }
        (out / "split.json").write_text(json.dumps(manifest, indent=2))
        print("SPLIT", json.dumps(manifest), flush=True)

    @torch.no_grad()
    @torch.random.fork_rng(devices=[torch.cuda.current_device()])
    def audit_epoch(epoch, model, source_context, test_loader):
        nonlocal selected_row
        with isolated_evaluation(model):
            if args.selection == "source_val_best":
                val_loss, val_accuracy = evaluate_source(model, validation["loader"], "cuda")
                target_metrics = evaluate_target(
                    model, cached["t"], validation["target_gt"],
                    validation["reference"], "cuda")
                row = {"epoch": epoch, **diagnostics.summary(), **target_metrics,
                       "evaluated_n": target_metrics["num_target_evaluation_samples"],
                       "source_val_loss": val_loss, "source_val_accuracy": val_accuracy}
            else:
                predictions, labels = [], []
                for target_data, target_label in test_loader:
                    outputs = model(source_context.cuda(), target_data.cuda())[8]
                    predictions.extend(outputs.argmax(1).cpu().tolist())
                    labels.extend(target_label.tolist())
                row = {"epoch": epoch, **diagnostics.summary(),
                       **metrics_from_predictions(np.asarray(predictions), np.asarray(labels))}
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if args.selection == "source_val_best":
            checkpoint = {
                "model": model.state_dict(), "flow": namespace["flow"].state_dict(),
                "optimizer": namespace["optimizer"].state_dict(),
                "metrics": row, "epoch": epoch, "selection": args.selection,
                "seed": args.seed, "ablation": args.ablation,
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
                "flow_rng": namespace["flow_rng"].get_state(),
            }
            if improves_source_validation(row, selected_row):
                selected_row = dict(row)
                atomic_save(checkpoint, out / "best_source_val.pth")
                (out / "best_source_val.json").write_text(json.dumps(selected_row, indent=2))
            atomic_save(checkpoint, out / "last.pth")
        print("AGREEMENT_EPOCH", json.dumps(row), flush=True)

    def audit_step(epoch, index, model, step, source_data, target_data, labels):
        row = {
            "epoch": epoch, "step": index,
            "feature_encoder_hash": parameter_hash(model),
            "backbone_hash": parameter_hash(model.feature_layers),
            "classifier_hash": parameter_hash(model.fc1),
            "buffers_hash": tensor_sequence_hash(model.buffers()),
            "gradient_hash": tensor_sequence_hash(
                p.grad if p.grad is not None else torch.empty(0) for p in model.parameters()),
            "ce": float(step["ce"].detach()),
            "ce_s": float(step["ce_s"].detach()),
            "ce_ss": float(step["ce_ss"].detach()) if step["ce_ss"] is not None else None,
            "source_logits_hash": tensor_sequence_hash([step["source_logits"]]),
            "target_logits_hash": tensor_sequence_hash([step["target_logits"]]),
            "source_input_hash": tensor_sequence_hash([source_data, labels]),
            "target_input_hash": tensor_sequence_hash([target_data]),
            "cpu_rng_hash": tensor_sequence_hash([torch.get_rng_state()]),
            "cuda_rng_hash": tensor_sequence_hash([torch.cuda.get_rng_state()]),
        }
        with (out / "step_audit.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    namespace = {
        "__name__": "__main__",
        "lr_horizon": args.lr_horizon,
        "audit_split": audit_split,
        "audit_epoch": audit_epoch,
        "audit_step": audit_step,
        "args_flow": {"init_seed": args.seed + 71039, "sampling_seed": args.seed + 99173},
        "adaptation": adaptation,
        "training_step": training_step,
        "ablation": args.ablation,
        "flow_init_seed": flow_init_seed,
        "flow_sampling_seed": flow_sampling_seed,
        "diagnostics": diagnostics,
        "weight_mode": "q",
        "scene_stats": scene_stats,
        "scene_shift_batch": scene_shift_batch,
        "forward_without_bn_update": forward_without_bn_update,
        "scene_shift_enabled": args.scene_shift,
        "scene_shift_only": args.scene_shift_only,
        "scene_shift_aux_original_flow": args.scene_shift_aux_original_flow,
        "fm_backprop_source": args.fm_backprop_source,
        "reliability_routing": args.reliability_routing,
        "ConditionalFlowMLP": ConditionalFlowMLP,
        "install_deterministic_pool": install_deterministic_pool,
    }
    snapshot = out / "code_snapshot"
    snapshot.mkdir()
    provenance = {}
    for path in [Path(__file__), ROOT / "source_val_protocol.py", ROOT / "class_conditional_flow.py", official_path,
                 LEGACY / "net2.py", LEGACY / "utils.py", LEGACY / "UtilsCMS.py",
                 LEGACY / "config_Houston.py"]:
        data = path.read_bytes()
        (snapshot / path.name).write_bytes(data)
        provenance[str(path)] = hashlib.sha256(data).hexdigest()
    (out / "provenance.json").write_text(json.dumps({
        "files": provenance, "command": sys.argv, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(), "deterministic": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
    }, indent=2))
    (out / "executed.py").write_text(code)
    os.chdir(LEGACY)
    exec(compile(code, str(official_path), "exec"), namespace)
    best_diagnostic = max(history, key=lambda row: row["oa"])
    chosen = selected_row if args.selection == "source_val_best" else history[-1]
    result = {**chosen, "training_epochs": args.epochs,
              "lr_horizon": args.lr_horizon,
              "selection": "source_val_best" if selected_row else f"fixed_epoch{args.epochs}",
              "method": (
                  "scene_shift_only" if args.scene_shift_only else
                  ("scene_shift_aux_original_flow" if args.scene_shift_aux_original_flow else "flow_transport")
              ),
              "scene_shift_only": args.scene_shift_only,
              "scene_shift_aux_original_flow": args.scene_shift_aux_original_flow,
              "fm_backprop_source": args.fm_backprop_source,
              "reliability_routing": args.reliability_routing,
                "routing_assignment": "soft",
              "diagnostic_best_oa": best_diagnostic["oa"],
              "diagnostic_best_epoch": best_diagnostic["epoch"],
              "fixed_epoch_minus_best_oa": history[-1]["oa"] - best_diagnostic["oa"],
              "fixed_epoch_best_drop": best_diagnostic["oa"] - history[-1]["oa"],
              "ablation": args.ablation, "seed": args.seed}
    if args.epochs == 100:
        result["epoch100_minus_best_oa"] = result["fixed_epoch_minus_best_oa"]
        result["epoch100_best_drop"] = result["fixed_epoch_best_drop"]
    result["gpu_max_allocated"] = torch.cuda.max_memory_allocated()
    final_metrics = {**history[-1], "selection": f"fixed_epoch{args.epochs}"}
    result["fixed_epoch_metrics"] = final_metrics
    result["selected_checkpoint"] = "best_source_val.pth" if selected_row else f"epoch{args.epochs}.pth"
    result["best_epoch"] = chosen["epoch"]
    atomic_save({"model": namespace["feature_encoder"].state_dict(),
                 "flow": namespace["flow"].state_dict(), "metrics": final_metrics},
                out / f"epoch{args.epochs}.pth")
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("FINAL", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
