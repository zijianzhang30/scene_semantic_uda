"""Collect completed SceneShift result.json files into the final tables.

This is intentionally discovery-based: it never retrains or overwrites an
experiment, and records the source path for every row so reused results remain
auditable.
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path('/home/zhangzj26/scene_semantic_uda')
OUT = ROOT / 'final_quantitative'

def metric(d, key):
    return d.get(key, d.get(key.lower()))

def add(rows, path, dataset, backbone, method, family):
    try:
        d = json.loads(path.read_text())
    except Exception:
        return
    if not all(k in d for k in ('oa','aa','kappa')):
        return
    p = d.get('per_class_accuracy', [])
    rows.append({
        'dataset': dataset, 'backbone': backbone, 'method': method,
        'family': family, 'seed': d.get('split_seed', d.get('seed')),
        'oa': d['oa'], 'aa': d['aa'], 'kappa': d['kappa'],
        'best_epoch': d.get('best_epoch'), 'source_path': str(path),
        'per_class_accuracy': p,
    })

def discover(rows):
    # Newly completed multi-seed tables.
    for p in (OUT/'backbone_cnn').glob('**/result.json'):
        parts = p.parts
        dataset = next((x for x in ('houston','pavia','shanghai_hangzhou') if x in parts), 'unknown')
        method = 'DCRN + SceneShift' if 'scene_shift' in parts else 'DCRN + CE'
        add(rows,p,dataset,'DCRN',method,'standalone')
    for p in (OUT/'backbone_ssftt').glob('**/result.json'):
        parts = p.parts
        dataset = next((x for x in ('houston','pavia','shanghai_hangzhou') if x in parts), 'unknown')
        method = 'SSFTT + SceneShift' if 'scene_shift' in parts else 'SSFTT + CE'
        add(rows,p,dataset,'SSFTT',method,'standalone')
    for p in (OUT/'mluda_multiseed').glob('**/result.json'):
        method = 'Original MLUDA' if 'full_mluda' in p.parts else 'Dual-SceneShift'
        add(rows,p,'Houston13→Houston18','DCRN/MLUDA',method,'full_uda')
    # Existing audited Houston MLUDA counterpart results.
    for p in (ROOT/'runs_mluda_counterpart').glob('split_*/**/result.json'):
        if 'mluda_dual_counterpart' in p.parts:
            add(rows,p,'Houston13→Houston18','DCRN/MLUDA','Dual-SceneShift','full_uda')
        elif 'full_mluda' in p.parts:
            add(rows,p,'Houston13→Houston18','DCRN/MLUDA','Original MLUDA','full_uda')
    # Existing standalone cross-scene DCRN and SSFTT single-seed results.
    for root, dataset in ((ROOT/'runs_pavia_scene_shift','PaviaU→PaviaC'),
                          (ROOT/'runs_shanghai_hangzhou_scene_shift','Shanghai→Hangzhou')):
        for p in root.glob('**/result.json'):
            method = 'DCRN + SceneShift' if 'scene_shift' in p.parts else 'DCRN + CE'
            add(rows,p,dataset,'DCRN',method,'standalone')
    ssroot = ROOT/'runs_ssftt_scene_shift'
    for p in ssroot.glob('**/result.json'):
        dataset = 'Houston13→Houston18' if 'houston' in p.parts else ('PaviaU→PaviaC' if 'pavia' in p.parts else 'Shanghai→Hangzhou')
        method = 'SSFTT + SceneShift' if 'scene_shift' in p.parts else 'SSFTT + CE'
        add(rows,p,dataset,'SSFTT',method,'standalone')
    # Stable full DAMamba report is already aggregated; preserve its per-seed rows.
    for p in (OUT/'full_damamba_scene_shift_stable_lr').glob('per_seed.csv'):
        try:
            with p.open() as f:
                for r in csv.DictReader(f):
                    rows.append({'dataset':'Houston13→Houston18','backbone':'Full DAMamba',
                                 'method':r.get('method','Full DAMamba'), 'family':'full_uda',
                                 'seed':r.get('seed'), 'oa':float(r['oa']), 'aa':float(r['aa']),
                                 'kappa':float(r['kappa']), 'best_epoch':r.get('best_epoch'),
                                 'source_path':str(p), 'per_class_accuracy':[]})
        except Exception:
            pass

def main():
    rows=[]; discover(rows)
    # Deduplicate exact paths and overlapping legacy/final records. Prefer the
    # final_quantitative record when the same (dataset, method, seed) exists.
    unique=[]; seen=set()
    for r in rows:
        key=(r['source_path'],r['method'])
        if key not in seen: unique.append(r); seen.add(key)
    rows=unique
    preferred={}
    for r in rows:
        key=(r['dataset'],r['backbone'],r['method'],str(r.get('seed')))
        old=preferred.get(key)
        if old is None or ('final_quantitative' in r['source_path'] and 'final_quantitative' not in old['source_path']):
            preferred[key]=r
    rows=list(preferred.values())
    OUT.mkdir(exist_ok=True)
    with (OUT/'all_results.csv').open('w',newline='') as f:
        fields=['dataset','backbone','method','family','seed','oa','aa','kappa','best_epoch','source_path']
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in rows)
    with (OUT/'all_per_seed.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['dataset','backbone','method','family','seed','class','accuracy','source_path'])
        for r in rows:
            for i,a in enumerate(r.get('per_class_accuracy') or [],1):
                w.writerow([r['dataset'],r['backbone'],r['method'],r['family'],r['seed'],i,a,r['source_path']])
    summary={'rows':len(rows),'generated_from_discovered_results':True,'rows_data':rows}
    (OUT/'overall_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    lines=['# SceneShift final quantitative summary','',
           'This table is generated only from completed, auditable `result.json`/`per_seed.csv` files; no experiment is retrained.','',
           '| Dataset | Backbone | Method | n | OA mean±std | AA mean±std | Kappa mean±std |','|---|---|---|---:|---:|---:|---:|']
    import numpy as np
    groups={}
    for r in rows: groups.setdefault((r['dataset'],r['backbone'],r['method']),[]).append(r)
    for (ds,bb,m),rs in sorted(groups.items()):
        vals=[]
        for k in ('oa','aa','kappa'):
            a=np.asarray([float(x[k]) for x in rs],dtype=float)
            vals.append(f'{a.mean()*100:.2f}±{a.std(ddof=0)*100:.2f}')
        lines.append(f'| {ds} | {bb} | {m} | {len(rs)} | {vals[0]} | {vals[1]} | {vals[2]} |')
    lines += ['', '## Protocol notes', '', '- Target labels are not used for training or source-validation checkpoint selection in the audited runners.', '- Rows with fewer than the requested number of seeds are retained but their `n` is shown explicitly; they are not presented as matched multi-seed evidence.', '- Full DAMamba rows refer to the stable corrected active-UDA implementation, while clean Mamba rows are separate supervised backbone experiments.']
    (OUT/'overall_summary.md').write_text('\n'.join(lines)+'\n')
    print(f'wrote {len(rows)} rows')

if __name__=='__main__': main()
