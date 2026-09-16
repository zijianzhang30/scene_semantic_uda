"""Houston raw Mean Teacher control; target labels are diagnostic-only."""
import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sceneshiftnet_dcrn_train import PatchDataset, sample_source, set_seed, evaluate, file_sha256
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from models.sceneshift_net_dcrn import SceneShiftNetDCRN

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'runs_sceneshiftnet_dcrn/houston_raw/sceneshift/seed_1341'


class IndexedTarget(PatchDataset):
    def __getitem__(self, index):
        return super().__getitem__(index), index


@torch.no_grad()
def update_teacher(teacher, student, momentum=.999):
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.mul_(momentum).add_(sp, alpha=1-momentum)
    for tb, sb in zip(teacher.buffers(), student.buffers()):
        if tb.is_floating_point():
            tb.mul_(momentum).add_(sb, alpha=1-momentum)
        else:
            tb.copy_(sb)


def views(x, noise_std, generator):
    h = torch.rand((len(x), 1, 1, 1), device=x.device, generator=generator) < .5
    v = torch.rand((len(x), 1, 1, 1), device=x.device, generator=generator) < .5
    weak = torch.where(h, x.flip(-1), x)
    weak = torch.where(v, weak.flip(-2), weak)
    noise = torch.randn(x.shape, device=x.device, generator=generator) * noise_std
    return weak, weak + noise, noise


def diagnostics(truth, pred, selected):
    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (truth[selected], pred[selected]), 1)
    precision = [float(cm[c, c]/cm[:, c].sum()) if cm[:, c].sum() else None for c in range(7)]
    recall = [float(cm[c, c]/(truth == c).sum()) if (truth == c).any() else None for c in range(7)]
    return dict(precision_per_class=precision, recall_per_class=recall,
                class1_to_class2_count=int(cm[0, 1]), confusion_matrix=cm.tolist())


@torch.no_grad()
def full_diagnostic(model, loader, device):
    model.eval()
    labels, predictions, confidence = [], [], []
    for x, y in loader:
        q, p = model(x.to(device))[1].softmax(1).max(1)
        labels.append(y.numpy()); predictions.append(p.cpu().numpy()); confidence.append(q.cpu().numpy())
    truth, pred, conf = map(np.concatenate, (labels, predictions, confidence))
    return dict(all=diagnostics(truth, pred, np.ones(len(truth), bool)),
                confident=diagnostics(truth, pred, conf > .9), coverage=float((conf > .9).mean()))


def run(args):
    out = ROOT / 'runs_sceneshiftnet_dcrn/houston_raw' / args.tag / f'seed_{args.seed}'
    out.mkdir(parents=True, exist_ok=False)
    set_seed(args.seed)
    s, sg, t, tg, paths = load_cubes('none')
    s, t = s.astype('float32'), t.astype('float32')
    centers, labels = sample_source(sg, 7, 180, np.random.RandomState(args.seed))
    tc = np.argwhere(tg > 0)
    sl = DataLoader(PatchDataset(s, centers, 7, labels), 32, shuffle=True, drop_last=True)
    tl = DataLoader(IndexedTarget(t, tc, 7), 32, shuffle=True, drop_last=True)
    el = DataLoader(PatchDataset(t, tc, 7, tg[tc[:, 0], tc[:, 1]].astype('int64')-1), 32)
    sm, ss = s.reshape(-1, 48).mean(0), s.reshape(-1, 48).std(0)
    tm, ts = t.reshape(-1, 48).mean(0), t.reshape(-1, 48).std(0)
    noise_std = .01 * ts
    config = dict(seed=args.seed, epochs=args.epochs, batch_size=32, normalization='none', use_ilda=False,
                  patch_size=7, bands=48, classes=7, source_per_class=180, optimizer='Adam', lr=.001,
                  weight_decay=0, alpha=.8, scene_shift_type='pure_affine', random_scale=False, smooth_noise=False,
                  scene_shift_stats='full float32 source/target cubes, all pixels, no GT', epsilon=1e-5,
                  warmup=10, threshold=.9, lambda_shift=.5, lambda_MT=.5, ema_momentum=.999,
                  teacher_bn='eval mode; EMA floating buffers; copy integer buffers after optimizer step',
                  noise_std_per_band=noise_std.tolist(), noise_scale=.01, shared_view_geometry=True,
                  augmentation_rng='independent CUDA generator seed+10000',
                  loss='L_src + .5 L_shift + .5 L_MT; MT enabled epoch > 10',
                  evaluation='final student, all target GT>0, background excluded; teacher separately reported',
                  target_gt_usage='support mask and offline diagnostics only',
                  code_hash={str(p):file_sha256(p) for p in [Path(__file__), ROOT/'models/sceneshift_net_dcrn.py', ROOT/'sceneshiftnet_dcrn_train.py']},
                  data_hash={str(p):file_sha256(p) for p in paths})
    (out/'config.json').write_text(json.dumps(config, indent=2))
    (out/'source_code.py').write_text(Path(__file__).read_text())
    np.savez(out/'indices.npz', source=centers, source_labels=labels, target=tc)
    print(json.dumps(config), flush=True)
    assert torch.cuda.is_available()
    device = 'cuda'
    student = SceneShiftNetDCRN(48, 7, 7).to(device)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=.001)
    sm, ss, tm, ts, noise_std = [torch.tensor(v, device=device)[None, :, None, None] for v in (sm, ss, tm, ts, noise_std)]
    generator = torch.Generator(device=device).manual_seed(args.seed+10000)
    history = []
    for epoch in range(1, args.epochs+1):
        student.train(); teacher.eval(); target_iter = iter(tl)
        sums = np.zeros(3); correct = np.zeros(2); n = selected_n = agreement_n = 0
        hist = np.zeros((2, 7), dtype=np.int64); records = []; noise_sq = noise_num = 0
        for x, y in sl:
            z, ids = next(target_iter)
            x, y, z = x.to(device), y.to(device), z.to(device)
            _, src = student(x)
            shifted = (x-sm)/(ss+1e-5)*(.2*ss+.8*ts)+.2*sm+.8*tm
            _, shift = student(shifted)
            weak, strong, noise = views(z, noise_std, generator)
            with torch.no_grad():
                tq, tp = teacher(weak)[1].softmax(1).max(1)
            _, target = student(strong)
            sq, sp = target.detach().softmax(1).max(1)
            selected = tq > .9
            mt = F.cross_entropy(target[selected], tp[selected]) if epoch > 10 and selected.any() else target.sum()*0
            src_loss, shift_loss = F.cross_entropy(src, y), F.cross_entropy(shift, y)
            loss = src_loss + .5*shift_loss + .5*mt
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at epoch {epoch}')
            optimizer.zero_grad(); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in student.parameters()):
                raise RuntimeError('Nonfinite gradient')
            optimizer.step(); update_teacher(teacher, student)
            bs = len(y); n += bs
            sums += np.array([src_loss.item(), shift_loss.item(), mt.item()])*bs
            correct += [int((src.argmax(1)==y).sum()), int((shift.argmax(1)==y).sum())]
            selected_n += int(selected.sum()); agreement_n += int((tp==sp).sum())
            hist[0] += np.bincount(tp[selected].cpu().numpy(), minlength=7)
            hist[1] += np.bincount(sp[sq > .9].cpu().numpy(), minlength=7)
            noise_sq += noise.square().sum().item(); noise_num += noise.numel()
            records.append((ids.numpy(), tp.cpu().numpy(), sp.cpu().numpy(), selected.cpu().numpy(), (sq > .9).cpu().numpy()))
        # GT is first consulted after all optimizer steps for this epoch.
        ids, tp, sp, chosen, student_chosen = [np.concatenate(v) for v in zip(*records)]
        truth = tg[tc[ids, 0], tc[ids, 1]].astype('int64')-1
        row = dict(epoch=epoch, L_src=sums[0]/n, L_shift=sums[1]/n, L_MT=sums[2]/n,
                   source_accuracy=correct[0]/n, shifted_source_accuracy=correct[1]/n,
                   teacher_high_confidence_coverage=selected_n/n, MT_selected_count=selected_n if epoch > 10 else 0,
                   teacher_pseudo_label_histogram=hist[0].tolist(), student_pseudo_label_histogram=hist[1].tolist(),
                   teacher_student_agreement_ratio=agreement_n/n, actual_noise_rms=float(np.sqrt(noise_sq/noise_num)),
                   teacher_diagnostic=diagnostics(truth, tp, chosen), student_diagnostic=diagnostics(truth, sp, student_chosen),
                   teacher_all_diagnostic=diagnostics(truth, tp, np.ones(n, bool)),
                   student_all_diagnostic=diagnostics(truth, sp, np.ones(n, bool)), finite=True)
        history.append(row)
        (out/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
    torch.save(dict(model=student.state_dict(), teacher=teacher.state_dict(), config=config, seed=args.seed), out/'final.pth')
    result = evaluate(student, el, device, 7)
    result['student_diagnostic'] = full_diagnostic(student, el, device)
    result['teacher_metrics'] = evaluate(teacher, el, device, 7)
    result['teacher_diagnostic'] = full_diagnostic(teacher, el, device)
    result['last_epoch'] = history[-1]
    baseline = SceneShiftNetDCRN(48, 7, 7).to(device)
    baseline.load_state_dict(torch.load(BASE/'final.pth', map_location=device, weights_only=False)['model'])
    result['original_sceneshift_metrics'] = json.loads((BASE/'metrics.json').read_text())
    result['original_sceneshift_diagnostic'] = full_diagnostic(baseline, el, device)
    (out/'metrics.json').write_text(json.dumps(result, indent=2))
    with (out/'history.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=history[0]); writer.writeheader(); writer.writerows(history)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1341)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--tag', default='mean_teacher')
    run(parser.parse_args())
