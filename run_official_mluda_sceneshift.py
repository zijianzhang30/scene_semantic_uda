"""Add only the intermediate adaptation path to saved official baseline code."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
import run_official_mluda_reproduction as baseline

HERE = Path(__file__).resolve().parent
BASE = HERE / 'runs_mluda_official_sceneshift_v1'
SUMMARY = HERE / 'summarize_official_mluda_sceneshift.py'


def instrument(code):
    marker = 'data_s,data_t = ILDA(data_s,data_t,pca_n,radius)'
    assert code.count(marker) == 1
    code = code.replace(marker, marker + '\n_shift_stats = _prepare_shift_stats(data_s, data_t)')
    marker = '            # Update parameters'
    assert code.count(marker) == 1
    return code.replace(marker, '            _shift_loss = _intermediate_adapt(globals())\n'
                        '            loss = loss + 0.5 * _shift_loss\n\n' + marker)


def configure(manifest, dataset, root):
    original = json.loads((baseline.BASE / dataset / 'config.json').read_text())
    assert manifest['source_sha256'] == original['source_sha256']
    assert manifest['data_sha256'] == original['data_sha256']
    manifest.update(scene_shift=True, alpha=0.8, gamma=0.5, shifted_gt_ce=False,
                    clamp=False, statistics='all pixels, per band, post-official ILDA; no GT mask',
                    baseline_config=str(baseline.BASE / dataset / 'config.json'),
                    added_loss='gamma*(official coefficient*q*LMMD + source SCL + pseudo-labeled shifted SCL)',
                    rng_policy='extra path saves/restores Python, NumPy, Torch CPU/CUDA RNG',
                    ilda_reproducibility='original pre-seed ILDA RNG state not archived; same call, bitwise identity unverified')
    manifest['instrumentation_changes'] += ['extra unlabeled intermediate adaptation path', 'RNG isolation for extra path']
    shutil.copy2(__file__, root / Path(__file__).name)
    manifest['extension_sha256'] = baseline.sha(__file__)
    # Reuse the exact existing cross-dataset SceneShift function, not its runner.
    src = (HERE / 'train_cross_dataset_dual_mluda.py').read_text()
    tree = ast.parse(src)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'shift')
    shift_code = ast.get_source_segment(src, function)
    (root / 'scene_shift_function.py').write_text(shift_code + '\n')
    manifest['shift_function_sha256'] = baseline.sha(root / 'scene_shift_function.py')


def hooks(dataset, root):
    import numpy as np
    import random
    import torch
    import torch.nn.functional as F
    ns = {'torch': torch, 'F': F}
    exec((root / 'scene_shift_function.py').read_text(), ns)
    shift = ns['shift']

    def prepare(source, target):
        s = source.reshape(-1, source.shape[-1])
        t = target.reshape(-1, target.shape[-1])
        stats = (s.mean(0), s.std(0), t.mean(0), t.std(0))
        assert all(np.isfinite(v).all() for v in stats)
        np.savez(root / 'band_statistics.npz', sm=stats[0], ss=stats[1], tm=stats[2], ts=stats[3])
        return stats

    def adapt(g):
        state = np.random.get_state()
        py_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                x = g['source_data']
                model = g['feature_encoder']
                shifted = shift(x.cuda(), *g['_shift_stats'], alpha=.8)
                assert shifted.shape == x.shape and torch.isfinite(shifted).all()
                # Match the original augmentation order and raw/augmented forwards.
                u = g['utils']
                x0 = u.radiation_noise(x).type(torch.FloatTensor)
                x1 = u.flip_augmentation(x)
                t0 = u.radiation_noise(shifted.cpu()).type(torch.FloatTensor)
                t1 = u.flip_augmentation(shifted.cpu())
                raw = model(x.cuda(), shifted)
                aug0 = model(x0.cuda(), t0.cuda())
                aug1 = model(x1.cuda(), t1.cuda())
                pseudo = raw[8].softmax(1).detach().argmax(1)
                lmmd = g['mmd'].lmmd(raw[0], raw[5], g['source_label'], raw[8].softmax(1),
                                     BATCH_SIZE=g['BATCH_SIZE'], CLASS_NUM=g['CLASS_NUM'])
                source_scl = g['ContrastiveLoss_s'](torch.stack([aug0[1], aug1[1]], 1), g['source_label'])
                shifted_scl = g['ContrastiveLoss_t'](torch.stack([aug0[7], aug1[7]], 1), pseudo)
                coef = .3 if dataset == 'pavia' else .01
                loss = coef * g['lambd'] * lmmd + source_scl + shifted_scl
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite intermediate adaptation loss')
                return loss
        finally:
            np.random.set_state(state)
            random.setstate(py_state)

    return {'_prepare_shift_stats': prepare, '_intermediate_adapt': adapt}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=baseline.SETTINGS, required=True)
    p.add_argument('--prepare-only', action='store_true')
    a = p.parse_args()
    baseline.run(a.dataset, a.prepare_only, extension=sys.modules[__name__])
