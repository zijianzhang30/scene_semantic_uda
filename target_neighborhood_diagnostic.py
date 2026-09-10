"""Frozen-feature Target-Neighborhood Transferability diagnostic (split 1174).

No training is performed.  A single source-only (alpha=0) DCRN checkpoint is
used for every alpha, while the existing Scene Shift implementation is applied
to the same source patches.  Target labels are loaded only after all feature
scores have been computed, for post-hoc correlation with the stored sweep.
"""
from __future__ import annotations
import json, random
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

import train as clean
import utils
from model import DCRNClassifier

ROOT = Path('/home/zhangzj26/TGRS_MLUDA-2024')
HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs_target_neighborhood_diagnostic_1174'
ALPHAS = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
K = 10
TOP_ENTROPY = 50
TAU = 0.07
TARGET_BANK_SIZE = 10000
SEED = 1174


def patches(cube, centers, width=7):
    h = width // 2
    padded = np.pad(cube, ((h, h), (h, h), (0, 0)), mode='constant')
    out = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (r, c) in enumerate(centers):
        out[i] = padded[r:r + width, c:c + width].transpose(2, 0, 1)
    return out


def feat(model, arr, device, batch=512):
    vals = []
    with torch.no_grad():
        for i in range(0, len(arr), batch):
            x = torch.from_numpy(arr[i:i + batch]).to(device)
            vals.append(torch.nn.functional.normalize(model.forward_features(x), dim=1).cpu())
    return torch.cat(vals, 0).numpy().astype(np.float32)


def summarize(x):
    x = np.asarray(x, dtype=np.float64)
    return {'mean': float(x.mean()), 'std': float(x.std()),
            'min': float(x.min()), 'max': float(x.max())}


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {'pearson_r': None, 'pearson_p': None,
                'spearman_r': None, 'spearman_p': None, 'n': int(len(x))}
    pr, pp = pearsonr(x, y); sr, sp = spearmanr(x, y)
    return {'pearson_r': float(pr), 'pearson_p': float(pp),
            'spearman_r': float(sr), 'spearman_p': float(sp), 'n': int(len(x))}


def knn_scores(source_z, target_z, k=K, top_entropy=TOP_ENTROPY, tau=TAU, chunk=512):
    """Return KNN distance, top-K concentration, and top-50 soft entropy."""
    n = len(source_z)
    d = np.empty(n, np.float32); conc = np.empty(n, np.float32); ent = np.empty(n, np.float32)
    target_t = torch.from_numpy(target_z)
    for i in range(0, n, chunk):
        z = torch.from_numpy(source_z[i:i + chunk])
        sim = z @ target_t.T
        topv, _ = torch.topk(sim, k=max(k, top_entropy), dim=1)
        topk = topv[:, :k]
        d[i:i + len(z)] = (1.0 - topk).mean(1).numpy()
        conc[i:i + len(z)] = topk.mean(1).numpy()
        p = torch.softmax(topv[:, :top_entropy] / tau, dim=1)
        ent[i:i + len(z)] = (-(p * torch.log(p.clamp_min(1e-12))).sum(1)).numpy()
    return d, conc, ent


def main(device_name='cuda:0'):
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

    source, source_gt = utils.load_data_houston(
        str(ROOT / 'datasets/Houston/Houston13.mat'),
        str(ROOT / 'datasets/Houston/Houston13_7gt.mat'))
    target = hdf5storage.loadmat(str(ROOT / 'datasets/Houston/Houston18.mat'))['ori_data']
    source = source.astype(np.float32); target = target.astype(np.float32)

    # All labeled source centers (train+val), with source labels only.
    trc, try_, vac, vay = clean.source_split(source_gt, SEED)
    source_centers = np.concatenate([trc, vac], axis=0)
    source_y = np.concatenate([try_, vay], axis=0)
    source_x = patches(source, source_centers)

    # Fixed, unlabeled target bank shared by every alpha.
    all_centers = np.stack(np.meshgrid(np.arange(target.shape[0]),
                                       np.arange(target.shape[1]), indexing='ij'), -1).reshape(-1, 2)
    rng = np.random.RandomState(SEED)
    ids = rng.choice(len(all_centers), min(TARGET_BANK_SIZE, len(all_centers)), replace=False)
    target_centers = all_centers[ids]
    target_x = patches(target, target_centers)

    ck = torch.load(HERE / 'runs_scene_shift_strength_sweep_1174/alpha_0_0/best.pth', map_location='cpu')
    model = DCRNClassifier().to(device); model.load_state_dict(ck['model']); model.eval()
    target_z = feat(model, target_x, device)
    source_raw_z = feat(model, source_x, device)
    raw_d, raw_c, raw_h = knn_scores(source_raw_z, target_z)

    sf = source.reshape(-1, source.shape[-1]); tf = target.reshape(-1, target.shape[-1])
    sm, ss, tm, ts = sf.mean(0), sf.std(0), tf.mean(0), tf.std(0)
    rows = []
    per_class = {str(c + 1): {} for c in range(7)}
    global_arrays = {k: [] for k in ['knn_distance', 'delta_distance', 'entropy', 'topk_similarity']}

    # Use identical random Scene Shift perturbations across alpha by resetting seed.
    for ai, alpha in enumerate(ALPHAS):
        torch.manual_seed(SEED + ai)
        x_t = clean.scene_shift(torch.from_numpy(source_x).to(device), sm, ss, tm, ts, strength=float(alpha))
        z = feat(model, x_t.cpu().numpy(), device)
        d, conc, ent = knn_scores(z, target_z)
        delta = raw_d - d
        for key, val in [('knn_distance', d), ('delta_distance', delta), ('entropy', ent), ('topk_similarity', conc)]:
            global_arrays[key].append(float(val.mean()))
        rec = {'alpha': float(alpha), 'knn_distance': summarize(d),
               'delta_distance': summarize(delta), 'entropy': summarize(ent),
               'topk_similarity': summarize(conc), 'raw_knn_distance': summarize(raw_d)}
        rows.append(rec)
        for c in range(7):
            m = source_y == c
            per_class[str(c + 1)].setdefault('alpha', []).append(float(alpha))
            for key, val in [('knn_distance', d), ('delta_distance', delta),
                             ('entropy', ent), ('topk_similarity', conc)]:
                per_class[str(c + 1)].setdefault(key, []).append(summarize(val[m]))

    (OUT / 'neighborhood_global_summary.json').write_text(json.dumps({
        'protocol': {'split': SEED, 'frozen_checkpoint': str(HERE / 'runs_scene_shift_strength_sweep_1174/alpha_0_0/best.pth'),
                     'feature': 'L2-normalized DCRN forward_features (pre-classifier, 288D)',
                     'target_bank_size': int(len(target_x)), 'target_bank_seed': SEED,
                     'K': K, 'entropy_topK': TOP_ENTROPY, 'tau': TAU,
                     'target_gt_used_for_feature_scores': False},
        'alphas': rows, 'raw_source': {'knn_distance': summarize(raw_d),
                                       'topk_similarity': summarize(raw_c), 'entropy': summarize(raw_h)}
    }, indent=2))
    (OUT / 'neighborhood_per_class_summary.json').write_text(json.dumps({
        'protocol': {'source_labels_only': True, 'target_gt_used_for_feature_scores': False},
        'classes': per_class
    }, indent=2))

    # Target metrics are read only now, after feature diagnostics are complete.
    sweep = json.loads((HERE / 'runs_scene_shift_strength_sweep_1174/summary.json').read_text())['runs']
    metric = {round(float(r['alpha']), 4): r for r in sweep}
    target_rows = [metric[round(float(a), 4)] for a in ALPHAS]
    target_oa = [r['oa'] for r in target_rows]
    target_aa = [r['aa'] for r in target_rows]
    target_k = [r['kappa'] for r in target_rows]
    target_pc = np.asarray([r['per_class_accuracy'] for r in target_rows])
    feature_names = ['knn_distance', 'delta_distance', 'entropy', 'topk_similarity']
    global_corr = {score: {'OA': corr(global_arrays[score], target_oa),
                           'AA': corr(global_arrays[score], target_aa),
                           'Kappa': corr(global_arrays[score], target_k)} for score in feature_names}
    pc_corr = {}
    for c in range(7):
        pc_corr[str(c + 1)] = {}
        for score in feature_names:
            vals = [per_class[str(c + 1)][score][ai]['mean'] for ai in range(len(ALPHAS))]
            pc_corr[str(c + 1)][score] = corr(vals, target_pc[:, c])
    (OUT / 'correlation_summary.json').write_text(json.dumps({
        'alphas': ALPHAS.tolist(), 'target_metrics': {'OA': target_oa, 'AA': target_aa, 'Kappa': target_k,
                                                       'per_class_accuracy': target_pc.tolist()},
        'global': global_corr, 'per_class': pc_corr
    }, indent=2))

    # Global figure: each panel has neighborhood score and target metrics on a second axis.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = {'OA': target_oa, 'AA': target_aa, 'Kappa': target_k}
    for ax, score in zip(axes.flat, feature_names):
        y = global_arrays[score]
        ax.plot(ALPHAS, y, 'o-', label=score, color='tab:blue'); ax.set_xlabel('alpha'); ax.set_ylabel(score, color='tab:blue'); ax.grid(alpha=.25)
        ax2 = ax.twinx()
        for name, vals, col in [('OA', target_oa, 'tab:red'), ('AA', target_aa, 'tab:green'), ('Kappa', target_k, 'tab:purple')]:
            ax2.plot(ALPHAS, vals, '--', color=col, alpha=.75, label=name)
        ax2.set_ylabel('target metric', color='tab:red'); ax.set_title(score)
    fig.savefig(OUT / 'global_score_vs_alpha.png', dpi=160); plt.close(fig)

    def plot_classes(score, filename, ylabel):
        fig, ax = plt.subplots(figsize=(10, 6))
        for c in range(7):
            vals = [per_class[str(c + 1)][score][ai]['mean'] for ai in range(len(ALPHAS))]
            ax.plot(ALPHAS, vals, 'o-', label=f'C{c + 1}')
        ax.set_xlabel('alpha'); ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(ncol=4)
        fig.tight_layout(); fig.savefig(OUT / filename, dpi=160); plt.close(fig)
    plot_classes('knn_distance', 'per_class_distance_vs_alpha.png', 'mean KNN cosine distance')
    plot_classes('entropy', 'per_class_entropy_vs_alpha.png', 'top-50 soft neighborhood entropy')

    # Compact interpretation for the requested diagnostic questions.
    best_score = {s: float(ALPHAS[int(np.argmin(global_arrays[s]))] if s == 'knn_distance' else ALPHAS[int(np.argmax(global_arrays[s]))]) for s in feature_names}
    c6d = [per_class['6']['knn_distance'][i]['mean'] for i in range(6)]
    c7d = [per_class['7']['knn_distance'][i]['mean'] for i in range(6)]
    c3d = [per_class['3']['knn_distance'][i]['mean'] for i in range(6)]
    md = ['# Target-Neighborhood Transferability Diagnostic (split 1174)', '',
          'No training or loss modification was performed. A single frozen alpha=0 DCRN checkpoint and a fixed 10,000-pixel unlabeled target bank were used for every alpha. Target GT was read only after feature scores were computed, for correlation with the existing post-hoc sweep.', '',
          '## Global neighborhood scores', '',
          '| alpha | KNN distance | Δdistance | entropy | top-K cosine | OA | AA | Kappa |', '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for i, a in enumerate(ALPHAS):
        r = rows[i]; md.append(f"| {a:.1f} | {r['knn_distance']['mean']:.5f} | {r['delta_distance']['mean']:.5f} | {r['entropy']['mean']:.5f} | {r['topk_similarity']['mean']:.5f} | {target_oa[i]*100:.2f} | {target_aa[i]*100:.2f} | {target_k[i]*100:.2f} |")
    md += ['', f"Neighborhood-score optima (heuristic): `{best_score}`. Target OA best alpha is {ALPHAS[int(np.argmax(target_oa))]:.1f}; AA/Kappa best alpha is {ALPHAS[int(np.argmax(target_aa))]:.1f}/{ALPHAS[int(np.argmax(target_k))]:.1f}.", '', '## Correlation interpretation', '']
    for s in feature_names:
        md.append(f"- **{s} vs OA:** Pearson {global_corr[s]['OA']['pearson_r']:.3f}, Spearman {global_corr[s]['OA']['spearman_r']:.3f}; **vs AA:** Pearson {global_corr[s]['AA']['pearson_r']:.3f}, Spearman {global_corr[s]['AA']['spearman_r']:.3f}.")
    md += ['', '## Class diagnostic', '', '| class | distance at α=0 | distance at α=0.8 | distance at α=1.0 | target OA curve best α |', '|---|---:|---:|---:|---:|']
    for c in range(7):
        vals = [per_class[str(c + 1)]['knn_distance'][i]['mean'] for i in range(6)]
        md.append(f"| C{c + 1} | {vals[0]:.5f} | {vals[4]:.5f} | {vals[5]:.5f} | {ALPHAS[int(np.argmax(target_pc[:, c]))]:.1f} |")
    md += ['', f'C6 distance sequence: {[round(x, 5) for x in c6d]}', f'C7 distance sequence: {[round(x, 5) for x in c7d]}', f'C3 distance sequence: {[round(x, 5) for x in c3d]}', '', '## Conclusion', '', '1. The diagnostic tests whether manifold proximity tracks the existing target response; it does not select alpha during training.', '2. If distance/concentration improves monotonically while OA later falls, target-likeness alone is not sufficient and indicates over-shift/semantic distortion.', '3. Automatic alpha selection would only be justified if the correlations and class curves are consistent; otherwise this signal should remain diagnostic rather than become sample-wise weighting.']
    (OUT / 'summary.md').write_text('\n'.join(md) + '\n')
    print(json.dumps({'output': str(OUT), 'device': str(device), 'n_source': len(source_x), 'n_target_bank': len(target_x)}, indent=2))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cuda:0')
    main(ap.parse_args().device)
