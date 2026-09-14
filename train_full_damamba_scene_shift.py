"""Leakage-free Full DAMamba UDA and Full DAMamba+SceneShift.

The DAMamba loss path is intentionally kept in ``DAMamba_model.TransferNet``.
For the dual version we call the same forward/loss machinery for a real target
view and for a target-guided shifted-source view; the latter never receives a
source label in a target-side loss.  Source CE is taken from the real-target
call exactly once.  Checkpoints are selected by a held-out source validation
set only; target labels are loaded only after training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn import metrics

DAMAMBA_ROOT = Path("/home/zhangzj26/DAMamba")
sys.path.insert(0, str(DAMAMBA_ROOT))
from hsi_dataset import get_dataset, HyperX, HyperX_w  # noqa: E402
from hsi_uti import sample_gt  # noqa: E402
from losses import Prototype_t  # noqa: E402
import DAMamba_model  # noqa: E402
import utils  # noqa: E402


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scene_shift(x, sm, ss, tm, ts, alpha=0.8):
    """Validated global per-band SceneShift used by the clean experiments."""
    def t(v): return torch.as_tensor(v, device=x.device, dtype=x.dtype)[None, :, None, None]
    sm, ss, tm, ts = t(sm), t(ss), t(tm), t(ts)
    y = (x - sm) / (ss + 1e-5)
    y = y * ((1.0 - alpha) * ss + alpha * ts)
    y = y + (1.0 - alpha) * sm + alpha * tm
    scale = 1.0 + 0.04 * torch.randn(x.shape[0], 1, 1, 1, device=x.device)
    noise = F.avg_pool2d(torch.randn_like(y), kernel_size=5, stride=1, padding=2)
    return (y * scale + 0.015 * noise).clamp(0.0, 1.0)


def make_hyperparams(n_class, bands, batch_size):
    return dict(n_classes=n_class, n_bands=bands, ignored_labels=[0],
                center_pixel=False, supervision='full', patch_size=12,
                batch_size=batch_size, flip_augmentation=True,
                radiation_augmentation=True, mixture_augmentation=True)


def build_data(seed, batch_size):
    # get_dataset performs exactly the official per-scene max/L2 normalization.
    data_dir = str(DAMAMBA_ROOT / 'Houston') + os.sep
    src_img, src_gt, _, _, _, _ = get_dataset('Houston13', data_dir)
    tgt_img, tgt_gt, _, _, _, _ = get_dataset('Houston18', data_dir)
    nclass, bands = int(src_gt.max()), int(src_img.shape[-1])
    # Official 5% stratified source sample.  A fixed, disjoint source-val subset
    # is carved out of the remaining source labels and never uses target GT.
    train_gt, rest_gt, train_set, rest_set = sample_gt(src_gt, 0.05, mode='random')
    rng = np.random.RandomState(seed + 9173)
    order = rng.permutation(len(rest_set))
    nval = min(len(rest_set), max(len(train_set), 2000))
    val_set = np.asarray(rest_set[order[:nval]], dtype=np.int64)
    val_gt = np.zeros_like(src_gt)
    val_gt[val_set[:, 0], val_set[:, 1]] = val_set[:, 2]
    # Symmetric padding and HyperX reproduce the official even 12x12 extraction.
    pad = 12 // 2 + 1
    src_pad = np.pad(src_img, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    tgt_pad = np.pad(tgt_img, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    hp = make_hyperparams(nclass, bands, batch_size)
    train_ds = HyperX(src_pad, train_gt, **hp)
    val_hp = dict(hp); val_hp.update(flip_augmentation=False, radiation_augmentation=False,
                                     mixture_augmentation=False)
    val_ds = HyperX(src_pad, val_gt, **val_hp)
    # Complete unlabeled target cube; labels are a mask of ones and never read.
    unlabeled = np.ones(tgt_pad.shape[:2], dtype=np.int64)
    target_s_ds = HyperX(tgt_pad, unlabeled, **hp)
    target_w_ds = HyperX_w(tgt_pad, unlabeled, **val_hp)
    # Strong/weak views must address the same target center pixel.  The
    # original construction independently shuffled both dataset index lists;
    # align the weak dataset to the strong dataset and keep both loaders in
    # the same deterministic order.  Augmentation randomness remains
    # independent inside each dataset __getitem__ call.
    target_w_ds.indices = target_s_ds.indices.copy()
    target_w_ds.labels = list(target_s_ds.labels)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=0, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0)
    # Target sampling is matched to the source number of batches per epoch.
    target_n = max(len(train_loader) * batch_size, batch_size)
    target_s_loader = DataLoader(target_s_ds, batch_size=batch_size, shuffle=False,
                                 drop_last=True, num_workers=0)
    target_w_loader = DataLoader(target_w_ds, batch_size=batch_size, shuffle=False,
                                 drop_last=True, num_workers=0)
    # Stats are computed from unlabeled image cubes only.
    sm = src_img.reshape(-1, bands).mean(0); ss = src_img.reshape(-1, bands).std(0)
    tm = tgt_img.reshape(-1, bands).mean(0); ts = tgt_img.reshape(-1, bands).std(0)
    return (src_img, src_gt, tgt_img, tgt_gt, train_loader, val_loader,
            target_s_loader, target_w_loader, nclass, bands, sm, ss, tm, ts)


def evaluate_source(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).long() - 1
            pred = model.predict(x).argmax(1)
            correct += int((pred == y).sum()); total += len(y)
    return correct / max(total, 1)


def eval_target(model, tgt_img, tgt_gt, device, batch_size=100):
    # Called only after training/checkpoint selection.
    centers = np.argwhere(tgt_gt > 0)
    labels = tgt_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    pad = 12 // 2
    padded = np.pad(tgt_img, ((pad, pad), (pad, pad), (0, 0)), mode='symmetric')
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            c = centers[start:start + batch_size]
            patches = np.stack([padded[r:r+12, col:col+12].transpose(2, 0, 1)
                                for r, col in c]).astype(np.float32)
            preds.append(model.predict(torch.from_numpy(patches).to(device)).argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    cm = metrics.confusion_matrix(labels, pred, labels=np.arange(7))
    pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return dict(oa=float((pred == labels).mean() * 100), aa=float(pc.mean() * 100),
                kappa=float(metrics.cohen_kappa_score(labels, pred, labels=np.arange(7)) * 100),
                per_class_accuracy=(pc * 100).tolist(), confusion_matrix=cm.tolist())


def adaptation_loss(out, args):
    _, l_batch, l_fix, l_inter, l_intra = out[:5]
    value = l_fix * args.fixmatch_weight + l_inter * args.inter_weight
    value = value + l_intra * args.intra_weight + l_batch * args.batch_weight
    return value


def add_ca_sa_params(model, initial_lr):
    # The current formal wrapper fixes the historical optimizer omission.
    params = model.get_parameters(initial_lr=initial_lr)
    params.append({'params': model.ca.parameters(), 'lr': initial_lr})
    params.append({'params': model.sa.parameters(), 'lr': initial_lr})
    return params


def run_one(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    (src_img, src_gt, tgt_img, tgt_gt, src_loader, val_loader, tgt_s_loader,
     tgt_w_loader, nclass, bands, sm, ss, tm, ts) = build_data(args.seed, args.batch_size)
    args.n_class = nclass
    model = DAMamba_model.TransferNet(bands, nclass, transfer_loss='mmd',
                                      use_bottleneck=True, bottleneck_width=256,
                                      max_iter=args.epochs * max(1, len(src_loader))).to(device)
    # Official DAMamba recipe: SGD and polynomial LambdaLR; CA/SA included.
    # get_parameters expects a multiplier when LambdaLR is enabled.  Passing
    # args.lr here would apply the base learning rate twice (0.2*0.2).
    opt = torch.optim.SGD(add_ca_sa_params(model, 1.0), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda x: args.lr * (1. + 0.0003 * float(x)) ** (-0.75))
    src_it = len(src_loader); n_batch = src_it
    best_val = -1.0; best_epoch = 0; best_state = None
    history = []; t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        args._epoch = epoch; model.train(); ep_loss = 0.0
        it_s, it_ts, it_tw = iter(src_loader), iter(tgt_s_loader), iter(tgt_w_loader)
        proto = Prototype_t(C=nclass, dim=256)
        for b in range(n_batch):
            xs, ys = next(it_s); xts, _ = next(it_ts); xtw, _ = next(it_tw)
            xs, ys = xs.to(device), (ys.to(device).long() - 1)
            xts, xtw = xts.to(device), xtw.to(device)
            # Real-target full DAMamba path.
            out_real = model(b, proto, xs, xts, xtw, ys, epoch, args.start_msc, args)
            base = out_real[0] * args.cls_weight
            real_adapt = adaptation_loss(out_real, args) if epoch > args.start_msc else torch.zeros_like(base)
            total = base + real_adapt
            # Intermediate-domain path: no source labels enter target-side loss;
            # the same source labels are only used internally by DAMamba's source
            # feature/prototype alignment term, exactly as in loss_unl.
            if args.method == 'scene_shift':
                xshift = scene_shift(xs, sm, ss, tm, ts, args.alpha)
                # independent weak/strong views; no GT CE is computed here
                xshift_s = xshift
                xshift_w = xshift.flip(-1) if (b + epoch) % 2 else xshift
                out_shift = model(b, proto, xs, xshift_s, xshift_w, ys, epoch, args.start_msc, args)
                shift_adapt = adaptation_loss(out_shift, args) if epoch > args.start_msc else torch.zeros_like(base)
                total = total + args.gamma * shift_adapt
            if not torch.isfinite(total):
                raise FloatingPointError(f'non-finite loss at epoch={epoch} batch={b}')
            opt.zero_grad(set_to_none=True); total.backward(); opt.step(); sched.step()
            # Share one prototype state across the two counterpart calls.
            proto.update(out_real[7], b, out_real[5], out_real[6], args, norm=True)
            if args.method == 'scene_shift':
                proto.update(out_shift[7], b, out_shift[5], out_shift[6], args, norm=True)
            ep_loss += float(total.detach())
        val_acc = evaluate_source(model, val_loader, device)
        row = {'epoch': epoch, 'loss': ep_loss / max(n_batch, 1),
               'source_val_accuracy': val_acc, 'lr': float(opt.param_groups[0]['lr'])}
        history.append(row)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f'{args.method} seed={args.seed} epoch={epoch}/{args.epochs} '
                  f'loss={row["loss"]:.4f} source_val={val_acc:.4f}', flush=True)
    model.load_state_dict(best_state); target_result = eval_target(model, tgt_img, tgt_gt, device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'model': best_state, 'best_epoch': best_epoch,
                'source_val_accuracy': best_val, 'seed': args.seed,
                'method': args.method, 'target_gt_used_for_training_or_selection': False},
               args.output / 'best.pth')
    result = dict(target_result, seed=args.seed, method=args.method,
                  best_epoch=best_epoch, source_val_accuracy=best_val,
                  elapsed_seconds=time.time() - t0,
                  protocol='target_gt_posthoc_only', alpha=args.alpha if args.method == 'scene_shift' else None,
                  gamma=args.gamma if args.method == 'scene_shift' else None,
                  damamba_active_losses=['source_CE','OT_prototype','FixMatch','inter','intra','batch'],
                  shifted_source_ce=False, params=sum(p.numel() for p in model.parameters()))
    (args.output / 'history.json').write_text(json.dumps(history, indent=2))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['baseline', 'scene_shift'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--epochs', type=int, default=500)
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.2)
    p.add_argument('--alpha', type=float, default=0.8)
    p.add_argument('--gamma', type=float, default=0.5)
    p.add_argument('--start-msc', type=int, default=10)
    p.add_argument('--cls-weight', type=float, default=0.8)
    p.add_argument('--fixmatch-weight', type=float, default=1.0)
    p.add_argument('--inter-weight', type=float, default=1.0)
    p.add_argument('--intra-weight', type=float, default=1.0)
    p.add_argument('--batch-weight', type=float, default=1.0)
    p.add_argument('--threshold1', type=float, default=0.95)
    p.add_argument('--threshold2', type=float, default=0.4)
    p.add_argument('--T', type=float, default=0.1)
    p.add_argument('--warm-steps', type=int, default=10)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args(); run_one(args)


if __name__ == '__main__': main()
