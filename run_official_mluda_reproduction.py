"""Instrument the actual MLUDA entry points, without rewriting their training.

Runs three official seeds in a single process, with ILDA before the seed loop.
No SceneShift is enabled. Published-paper numbers are intentionally not guessed.
"""
import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path('/home/zhangzj26/TGRS_MLUDA-2024')
HERE = Path(__file__).resolve().parent
BASE = HERE / 'runs_mluda_official_reproduction_v1'
SETTINGS = {
    'pavia': ('MLUDA_up.py', 'config_UP2PC', [1622, 1322, 1256],
              ['Pavia/paviaU.mat', 'Pavia/paviaU_gt_7.mat', 'Pavia/pavia.mat', 'Pavia/pavia_gt_7.mat']),
    'shanghai_hangzhou': ('MLUDA_sh.py', 'config_SH2HZ', [1341, 1535, 1631],
                         ['Shanghai-Hangzhou/DataCube.mat']),
}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def instrument(source, config_module):
    """Only seed-count selection, export hooks and removal of plotting footer."""
    config_line = 'from ' + config_module + ' import *'
    assert source.count(config_line) == 1
    source = source.replace(config_line, config_line + '\nseeds = seeds[:3]\nnDataSet = len(seeds)')
    epoch_marker = '        train_end = time.time()'
    assert source.count(epoch_marker) == 1
    source = source.replace(epoch_marker, '        _record_epoch(globals())\n' + epoch_marker)
    seed_marker = '\nAA = np.mean(A, 1)'
    assert source.count(seed_marker) == 1
    source = source.replace(seed_marker, '\n    _save_seed(globals())\n' + seed_marker)
    plot_marker = '#################classification map'
    if source.count(plot_marker) == 1:
        source = source.split(plot_marker)[0]
    ast.parse(source)
    return source


def run(dataset, prepare_only=False, extension=None):
    entry, config_module, seeds, datasets = SETTINGS[dataset]
    root = (extension.BASE if extension else BASE) / dataset
    if (root / 'STARTED.json').exists():
        raise RuntimeError('Existing run found; refuse to overwrite or restart: ' + str(root))
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    # Snapshot Python dependencies, preserving local regular forward behavior.
    source_repo = (extension.SOURCE_REPO / dataset / 'source_snapshot'
                   if extension and hasattr(extension, 'SOURCE_REPO')
                   else (BASE / dataset / 'source_snapshot' if extension else REPO))
    for path in source_repo.glob('*.py'):
        shutil.copy2(path, snapshot / path.name)
    # The worktree contains only device-selection changes in these losses.
    # Use the tracked HEAD versions to match the entry points on logical cuda:0.
    for name in (() if extension else ('mmd.py', 'contrastive_loss.py')):
        code = subprocess.check_output(['git', '-C', str(REPO), 'show', 'HEAD:' + name])
        (snapshot / name).write_bytes(code)
    source = (snapshot / entry).read_text()
    executed = instrument(source, config_module)
    if extension:
        executed = extension.instrument(executed)
    (root / 'executed_entry.py').write_text(executed)
    shutil.copy2(__file__, root / 'instrumentation.py')
    head = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    manifest = {
        'dataset': dataset, 'entry': entry, 'local_repo_commit': head,
        'upstream_release_equivalence': 'not independently verified',
        'seeds': seeds, 'epochs': 100, 'batch_size': 32,
        'source_samples_per_class': 180, 'optimizer': 'SGD recreated every epoch',
        'lr': .001 if dataset == 'pavia' else .0003,
        'scheduler': None, 'momentum': .9, 'weight_decay': 5e-4,
        'patch_size': 11 if dataset == 'pavia' else 1,
        'bands': 102 if dataset == 'pavia' else 198,
        'normalization': 'official utils sklearn.preprocessing.scale on original dtype',
        'ilda': 'official ILDA once before seed loop; PCA=2 guidance; epsilon=.00009',
        'target_pool': 'official GT>0; get_all_data class-grouped shuffle',
        'target_class_label_supervision': False,
        'target_GT_access': 'mask AND class-grouped ordering; labels also used for final evaluation',
        'checkpoint_selection': 'epoch 100 final; no source-val or target-val selection',
        'evaluation': 'official drop_last=True, shuffled target order, last source training batch reference',
        'scene_shift': False,
        'instrumentation_changes': ['first three official seeds', 'epoch logging', 'per-seed final exports', 'omit post-evaluation visualization'],
        'source_sha256': {p.name: sha(p) for p in sorted(snapshot.glob('*.py'))},
        'executed_entry_sha256': sha(root / 'executed_entry.py'),
        'data_sha256': {name: sha(REPO / 'datasets' / name) for name in datasets},
    }
    if extension:
        extension.configure(manifest, dataset, root)
    (root / 'config.json').write_text(json.dumps(manifest, indent=2))
    (root / 'git_commit.txt').write_text(head + '\n')
    if prepare_only:
        print('PREPARED', dataset, flush=True)
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; execute outside sandbox')
    import numpy as np
    import scipy
    import sklearn
    manifest['versions'] = {'torch': torch.__version__, 'numpy': np.__version__,
                            'scipy': scipy.__version__, 'sklearn': sklearn.__version__}
    manifest['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    (root / 'config.json').write_text(json.dumps(manifest, indent=2))
    (root / 'STARTED.json').write_text(json.dumps({'pid': os.getpid(), 'time': time.time()}))
    def record_epoch(g):
        seed = int(g['seeds'][g['iDataSet']])
        row = {'seed': seed, 'epoch': int(g['epoch']), 'lr': float(g['LEARNING_RATE']),
               'last_batch_total_loss': float(g['loss'].detach().cpu().item()),
               'last_batch_ce': float(g['cls_loss'].detach().cpu().item()),
               'steps_per_epoch': int(g['num_iter']) - 1}
        if extension:
            row['shift_adaptation_loss'] = float(g['_shift_loss'].detach().cpu())
        with (root / 'epochs.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print('AUDIT_EPOCH ' + json.dumps(row), flush=True)
    def save_seed(g):
        idx = int(g['iDataSet']); seed = int(g['seeds'][idx])
        assert int(g['epoch']) == 100
        out = root / ('seed_' + str(seed)); out.mkdir(exist_ok=True)
        result = {
            'dataset': dataset, 'method': 'Original MLUDA official entry', 'seed': seed,
            'oa': float(g['acc'][idx, 0]), 'aa': float(g['A'][idx].mean() * 100),
            'kappa': float(g['k'][idx, 0] * 100), 'units': 'percent', 'epoch': 100,
            'per_class_accuracy': (g['A'][idx] * 100).tolist(),
            'target_eligible_samples': int(len(g['test_loader'].dataset)),
            'target_evaluated_samples': int(len(g['labels'])),
            'source_train_samples': int(len(g['train_dataset'])),
            'steps_per_epoch': int(g['len_source_loader']) - 1,
            'train_seconds': float(g['train_end'] - g['train_start']),
            'evaluation_seconds': float(g['test_end'] - g['train_end']),
            'checkpoint': 'final_epoch100.pth',
        }
        if extension:
            result['method'] = 'Official-aligned MLUDA + SceneShift'
        torch.save({'model': g['feature_encoder'].state_dict(), 'epoch': 100, 'seed': seed,
                    'optimizer': g['optimizer'].state_dict(), 'protocol': manifest}, out / 'final_epoch100.pth')
        np.savez_compressed(out / 'evaluation_provenance.npz', prediction=g['predict'],
                           labels=g['labels'], source_reference=g['source_data'].detach().cpu().numpy(),
                           target_order=g['RandPerm'], target_rows=g['Row'], target_cols=g['Column'])
        (out / 'config.json').write_text(json.dumps(dict(manifest, seed=seed), indent=2))
        with (out / 'per_class.csv').open('w', newline='') as f:
            w=csv.writer(f); w.writerow(['class','accuracy_percent'])
            w.writerows(enumerate(result['per_class_accuracy'], 1))
        (out / 'result.json').write_text(json.dumps(result, indent=2))
        print('AUDIT_RESULT ' + json.dumps(result), flush=True)
        summary_script = extension.SUMMARY if extension else HERE / 'summarize_official_mluda_reproduction.py'
        subprocess.run([sys.executable, str(summary_script)], check=True)
    sys.path.insert(0, str(snapshot))
    os.chdir(REPO)  # Preserve official relative ./datasets paths.
    namespace = {'__name__': '__main__', '__file__': str(root / 'executed_entry.py'),
                 '_record_epoch': record_epoch, '_save_seed': save_seed}
    if extension:
        namespace.update(extension.hooks(dataset, root))
    exec(compile(executed, str(root / 'executed_entry.py'), 'exec'), namespace)
    (root / 'COMPLETE.json').write_text(json.dumps({'time': time.time(), 'seeds': seeds}))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=SETTINGS, required=True)
    p.add_argument('--prepare-only', action='store_true'); a=p.parse_args()
    run(a.dataset, a.prepare_only)
