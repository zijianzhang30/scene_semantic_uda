"""Independent MLUDA + SceneShift + feature relational Gram consistency runner."""
import argparse, json, pathlib, sys, torch
import torch.nn.functional as F
import run_official_mluda_reproduction as baseline
import run_official_mluda_sceneshift as sceneshift

HERE = pathlib.Path(__file__).resolve().parent
BASE = HERE / 'runs_mluda_official_sceneshift_gram_v1'
OFFICIAL = HERE / 'organized_runs' / 'mluda_baseline' / 'runs_mluda_official_reproduction_v1'
SUMMARY = HERE / 'summarize_official_mluda_sceneshift.py'
LAMBDA_GRAM = 0.1

def instrument(code):
    code = sceneshift.instrument(code)
    code = code.replace('seeds = seeds[:3]\nnDataSet = len(seeds)', 'seeds = [1341]\nnDataSet = 1')
    marker = '            _shift_loss = _intermediate_adapt(globals())'
    assert code.count(marker) == 1
    return code.replace(marker, marker + "\n            _gram_loss, _gram_cosine, _gram_fro = _gram_consistency(globals())\n            loss = loss + _gram_loss")

def configure(manifest, dataset, root):
    sceneshift.configure(manifest, dataset, root)
    manifest.update(gram_consistency=True, lambda_gram=LAMBDA_GRAM,
                    gram_formula='mse(F.normalize(F_ss)@F.normalize(F_ss).T, (F.normalize(F_s)@F.normalize(F_s).T).detach())',
                    gram_feature='DSANSS output tuple index 0: source_features and index 5: shifted-source target_features',
                    inference_overhead='none; training-only loss')
    (root / 'config.json').write_text(json.dumps(manifest, indent=2))
    (root / 'gram_runner.py').write_text(pathlib.Path(__file__).read_text())
    manifest['gram_runner_sha256'] = baseline.sha(pathlib.Path(__file__))

def hooks(dataset, root):
    h = sceneshift.hooks(dataset, root)
    shift_ns = {'torch': torch, 'F': F}
    exec((root / 'scene_shift_function.py').read_text(), shift_ns)
    shift = shift_ns['shift']
    metrics_path = root / 'gram_metrics.jsonl'
    def gram(g):
        x = g['source_data']; stats = g['_shift_stats']; model = g['feature_encoder']
        sm,ss,tm,ts = [torch.as_tensor(v, device='cuda', dtype=x.dtype)[None,:,None,None] for v in stats]
        shifted = shift(x.cuda(), *stats, alpha=.8)
        raw = model(x.cuda(), shifted)
        fs = F.normalize(raw[0].flatten(1), p=2, dim=1)
        fss = F.normalize(raw[5].flatten(1), p=2, dim=1)
        gs = fs @ fs.transpose(0, 1)
        gss = fss @ fss.transpose(0, 1)
        loss = LAMBDA_GRAM * F.mse_loss(gss, gs.detach())
        cos = F.cosine_similarity(fs, fss, dim=1).mean()
        fro = torch.linalg.matrix_norm(gs - gss)
        with metrics_path.open('a') as f:
            f.write(json.dumps({'epoch': int(g['epoch']), 'gram_loss': float(loss.detach()),
                                'raw_gram_loss': float((loss/LAMBDA_GRAM).detach()),
                                'effective_gram_weight': LAMBDA_GRAM,
                                'mean_cosine': float(cos.detach()), 'gram_fro': float(fro.detach()),
                                'Gs_mean': float(gs.detach().mean()), 'Gs_std': float(gs.detach().std()),
                                'Gss_mean': float(gss.detach().mean()), 'Gss_std': float(gss.detach().std()),
                                'feature_shape_source': list(raw[0].shape),
                                'feature_shape_shifted_source': list(raw[5].shape)})+'\n')
        return loss, cos.detach(), fro.detach()
    h['_gram_consistency'] = gram
    return h

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--dataset', choices=baseline.SETTINGS, default='shanghai_hangzhou'); p.add_argument('--prepare-only', action='store_true'); a=p.parse_args()
    old_base = baseline.BASE; baseline.BASE = OFFICIAL
    class E: pass
    e=E(); e.BASE=BASE; e.SOURCE_REPO=OFFICIAL; e.instrument=instrument; e.configure=configure; e.hooks=hooks; e.SUMMARY=SUMMARY
    sceneshift.EXTRA_SOURCE_SCL=True; sceneshift.EXTRA_COMPONENT='both'
    baseline.run(a.dataset, a.prepare_only, extension=e)
