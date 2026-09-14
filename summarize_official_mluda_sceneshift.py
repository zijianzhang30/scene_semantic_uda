"""Matched-only official-aligned comparison; never pool clean results."""
import csv
import json
import os
from pathlib import Path
import statistics as st
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / 'runs_mluda_official_reproduction_v1'
ROOT = HERE / 'runs_mluda_official_sceneshift_v1'
SEEDS = {'pavia': [1622,1322,1256], 'shanghai_hangzhou': [1341,1535,1631]}

def write(path, content):
    tmp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(content)
    tmp.replace(path)

def main():
    ROOT.mkdir(exist_ok=True)
    summary = {}; rows = []; classes = []
    lines = ['# Official-aligned MLUDA + SceneShift', '',
             'alpha=0.8, gamma=0.5; no shifted GT CE, no clamp. Epoch 100 final evaluation.',
             'Population std (ddof=0), all metrics in percent; deltas in percentage points.',
             'Original target mask/class-grouped ordering retained; no target class supervision.',
             'Pre-seed ILDA RNG was not archived for baseline; exact transformed-cube identity is unverified.', '',
             '| Dataset | Method | Completed pairs | OA | AA | Kappa |',
             '|---|---|---:|---:|---:|---:|']
    for dataset, seeds in SEEDS.items():
        pairs = []
        for seed in seeds:
            bp = BASE / dataset / f'seed_{seed}'
            sp = ROOT / dataset / f'seed_{seed}'
            if not (sp / 'result.json').exists():
                continue
            b = json.loads((bp / 'result.json').read_text())
            s = json.loads((sp / 'result.json').read_text())
            assert b['seed'] == s['seed'] == seed and b['epoch'] == s['epoch'] == 100
            with np.load(bp/'evaluation_provenance.npz') as bn, np.load(sp/'evaluation_provenance.npz') as sn:
                match = all(np.array_equal(bn[k], sn[k]) for k in ['labels','target_order','target_rows','target_cols'])
            if not match:
                raise RuntimeError(f'Target ordering mismatch: {dataset} {seed}')
            pairs.append((b,s))
            for name, r in [('Original MLUDA',b),('MLUDA + SceneShift',s)]:
                rows.append([dataset,seed,name,r['oa'],r['aa'],r['kappa']])
            for k,(bv,sv) in enumerate(zip(b['per_class_accuracy'],s['per_class_accuracy']),1):
                classes.append([dataset,seed,k,bv,sv,sv-bv])
        summary[dataset] = {'completed_pairs': len(pairs), 'seeds': seeds}
        if not pairs:
            lines.append(f'| {dataset} | Pending | 0/3 | TODO | TODO | TODO |')
            continue
        for idx,name in enumerate(['Original MLUDA','MLUDA + SceneShift']):
            metrics = {k: {'mean':st.mean(p[idx][k] for p in pairs), 'std':st.pstdev(p[idx][k] for p in pairs)} for k in ['oa','aa','kappa']}
            summary[dataset][name] = metrics
            vals = ' | '.join(f"{v['mean']:.2f}±{v['std']:.2f}" for v in metrics.values())
            lines.append(f'| {dataset} | {name} | {len(pairs)}/3 | {vals} |')
        summary[dataset]['paired'] = {k: {'mean_delta':st.mean(s[k]-b[k] for b,s in pairs),
                                             'wins':sum(s[k]>b[k] for b,s in pairs), 'total':len(pairs)} for k in ['oa','aa','kappa']}
    lines += ['', '## Paired improvements and wins', '', '```json', json.dumps({d:s.get('paired',{}) for d,s in summary.items()},indent=2), '```',
              '', '## Per-seed results', '', '| Dataset | Seed | Method | OA | AA | Kappa |','|---|---:|---|---:|---:|---:|']
    lines += ['| '+' | '.join(str(v) if not isinstance(v,float) else f'{v:.4f}' for v in r)+' |' for r in rows]
    lines += ['', '## Per-class paired comparison', '', '| Dataset | Class | Original mean | Shift mean | Mean delta |','|---|---:|---:|---:|---:|']
    for dataset in SEEDS:
        for k in sorted({r[2] for r in classes if r[0]==dataset}):
            rs = [r for r in classes if r[0]==dataset and r[2]==k]
            lines.append(f'| {dataset} | C{k} | '+ ' | '.join(f'{st.mean(r[j] for r in rs):.4f}' for j in [3,4,5])+' |')
    write(ROOT/'summary.json',json.dumps(summary,indent=2))
    for name,header,data in [('per_seed.csv',['dataset','seed','method','OA','AA','Kappa'],rows),
                             ('per_class.csv',['dataset','seed','class','original','shift','delta'],classes)]:
        import io
        f=io.StringIO(); w=csv.writer(f); w.writerow(header); w.writerows(data); write(ROOT/name,f.getvalue())
    write(HERE/'mluda_official_sceneshift_pavia_shanghai.md','\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
