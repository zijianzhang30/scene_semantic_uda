"""Official training objective with the same diagnostic evaluator as Flow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from train_houston_flow_transport import LEGACY, ROOT, cfg, UtilsCMS, sha, metrics_from_predictions

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve() / 'official'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'history.json').exists():
        raise FileExistsError(f'Refusing to overwrite {out}')
    cache = np.load(ROOT / 'runs_strict_mluda_1341/ilda.npz')
    def paired_ilda(s, t, n, r):
        assert n == 2 and r == .009
        return cache['s'].copy(), cache['t'].copy()
    UtilsCMS.ILDA = paired_ilda
    cfg.seeds, cfg.nDataSet, cfg.epochs = [args.seed], 1, 100
    original = (LEGACY / 'MLUDA_hu.py').read_text()
    # Remove only post-training visualization, retain the official training/eval.
    code = original.split('#################classification map')[0]
    anchor = '    print("Training...")'
    assert code.count(anchor) == 1
    code = code.replace(anchor, '    audit_split(trainX, trainY, testX, testY, feature_encoder)\n' + anchor)
    anchor = '        train_end = time.time()'
    assert code.count(anchor) == 1
    code = code.replace(anchor, '        audit_epoch(epoch, feature_encoder, source_data, test_loader)\n' + anchor)
    history = []
    def audit_split(x, y, tx, ty, model):
        row = dict(source_x=sha(x), source_y=sha(y), target_x=sha(tx), target_y=sha(ty),
                   seed=args.seed, epochs=100, source_n=len(y), target_n=len(ty),
                   initial_model=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest(),
                   official_file_sha256=hashlib.sha256(original.encode()).hexdigest())
        (out / 'split.json').write_text(json.dumps(row, indent=2))
        print('SPLIT', json.dumps(row), flush=True)
    @torch.no_grad()
    def audit_epoch(epoch, model, source, loader):
        was = model.training
        model.eval()
        predictions, labels = [], []
        for data, truth in loader:
            predictions.extend(model(source.cuda(), data.cuda())[8].argmax(1).cpu().tolist())
            labels.extend(truth.tolist())
        model.train(was)
        row = dict(epoch=epoch, **metrics_from_predictions(np.asarray(predictions), np.asarray(labels)))
        assert row['evaluated_n'] == 53184
        history.append(row)
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        print('EPOCH', json.dumps(row), flush=True)
    namespace = dict(__name__='__main__', audit_split=audit_split, audit_epoch=audit_epoch)
    (out / 'executed.py').write_text(code)
    os.chdir(LEGACY)
    exec(compile(code, str(LEGACY / 'MLUDA_hu.py'), 'exec'), namespace)
    best = max(history, key=lambda x: x['oa'])
    result = dict(**history[-1], selection='fixed_epoch100', diagnostic_best_oa=best['oa'], diagnostic_best_epoch=best['epoch'])
    torch.save(dict(model=namespace['feature_encoder'].state_dict(), metrics=result), out / 'epoch100.pth')
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print('FINAL', json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
