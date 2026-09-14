import csv, json, math, os
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(os.environ.get('DAMAMBA_SUMMARY_ROOT', '/home/zhangzj26/scene_semantic_uda/final_quantitative/full_damamba_scene_shift'))
SEEDS = [1174, 1703, 2141]
METHODS = ['baseline', 'scene_shift']
rows = []
for seed in SEEDS:
    for method in METHODS:
        p = ROOT / f'seed_{seed}' / method / 'result.json'
        d = json.loads(p.read_text())
        d['seed'] = seed; d['method'] = method
        rows.append(d)

with (ROOT / 'per_seed.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['seed','method','OA','AA','Kappa','best_epoch','source_val_accuracy'])
    for d in rows: w.writerow([d['seed'], d['method'], d['oa'], d['aa'], d['kappa'], d['best_epoch'], d['source_val_accuracy']])

pc_rows = []
for d in rows:
    for c, value in enumerate(d['per_class_accuracy'], 1): pc_rows.append([d['seed'],d['method'],c,value])
with (ROOT / 'per_class.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['seed','method','class','accuracy'])
    w.writerows(pc_rows)

def vals(method, key): return np.array([d[key] for d in rows if d['method'] == method], dtype=float)
summary = {'protocol': {'dataset':'Houston13->Houston18','seeds':SEEDS,'alpha':0.8,'gamma':0.5,
                        'checkpoint':'source-validation only','target_gt_used_for_training_or_selection':False,
                        'shifted_source_ce':False,'active_damamba_losses':['source CE','OT/prototype','FixMatch','inter','intra','batch']},
           'methods': {}, 'paired': {}}
for method in METHODS:
    summary['methods'][method] = {}
    for key in ['oa','aa','kappa']:
        v = vals(method,key); summary['methods'][method][key] = {'mean':float(v.mean()),'std':float(v.std(ddof=1)), 'per_seed':v.tolist()}
    pc = np.array([d['per_class_accuracy'] for d in rows if d['method']==method], dtype=float)
    summary['methods'][method]['per_class_accuracy'] = {'mean':pc.mean(0).tolist(),'std':pc.std(0,ddof=1).tolist()}
for key in ['oa','aa','kappa']:
    b, s = vals('baseline',key), vals('scene_shift',key); delta=s-b
    try: t_p=float(stats.ttest_rel(s,b).pvalue); w_p=float(stats.wilcoxon(s,b).pvalue)
    except Exception: t_p=w_p=None
    summary['paired'][key] = {'delta_per_seed':delta.tolist(),'mean_delta':float(delta.mean()),
                              'std_delta':float(delta.std(ddof=1)), 'win_rate':float((delta>0).mean()),
                              'paired_ttest_p':t_p,'wilcoxon_p':w_p}

(ROOT/'summary.json').write_text(json.dumps(summary, indent=2))
def fmt(method,key):
    q=summary['methods'][method][key]; return f"{q['mean']:.2f} +/- {q['std']:.2f}"
lines=['# Full DAMamba UDA + SceneShift (Houston13 -> Houston18)','',
       'This is the leakage-free matched 3-seed comparison. DAMamba active UDA machinery is unchanged; SceneShift adds one unlabeled intermediate-domain call with alpha=0.8 and gamma=0.5. Best checkpoints use source validation only; target GT is post-hoc only.','',
       '| Method | Adaptation domains | OA | AA | Kappa |','|---|---|---:|---:|---:|',
       f"| Full DAMamba | real target | {fmt('baseline','oa')} | {fmt('baseline','aa')} | {fmt('baseline','kappa')} |",
       f"| Full DAMamba + SceneShift | real target + shifted source (unlabeled) | {fmt('scene_shift','oa')} | {fmt('scene_shift','aa')} | {fmt('scene_shift','kappa')} |",'','## Per-seed metrics','', '| Seed | DAMamba OA | +SceneShift OA | Delta | DAMamba AA | +SceneShift AA | Delta | DAMamba Kappa | +SceneShift Kappa | Delta |','|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for seed in SEEDS:
    b=next(d for d in rows if d['seed']==seed and d['method']=='baseline'); s=next(d for d in rows if d['seed']==seed and d['method']=='scene_shift')
    lines.append(f"| {seed} | {b['oa']:.2f} | {s['oa']:.2f} | {s['oa']-b['oa']:+.2f} | {b['aa']:.2f} | {s['aa']:.2f} | {s['aa']-b['aa']:+.2f} | {b['kappa']:.2f} | {s['kappa']:.2f} | {s['kappa']-b['kappa']:+.2f} |")
lines += ['', '## Paired summary','', '| Metric | Mean delta (+SceneShift - Full DAMamba) | Win rate | Paired t-test p | Wilcoxon p |','|---|---:|---:|---:|---:|']
for key,label in [('oa','OA'),('aa','AA'),('kappa','Kappa')]:
    q=summary['paired'][key]; lines.append(f"| {label} | {q['mean_delta']:+.2f} | {q['win_rate']*100:.1f}% | {q['paired_ttest_p']:.4g} | {q['wilcoxon_p']:.4g} |")
lines += ['', '## Mean per-class accuracy (%)','', '| Class | Full DAMamba | Full DAMamba + SceneShift | Delta |','|---:|---:|---:|---:|']
bpc=np.array(summary['methods']['baseline']['per_class_accuracy']['mean']); spc=np.array(summary['methods']['scene_shift']['per_class_accuracy']['mean'])
for c,(x,y) in enumerate(zip(bpc,spc),1): lines.append(f'| C{c} | {x:.2f} | {y:.2f} | {y-x:+.2f} |')
lines += ['', '## Interpretation','', '- The second branch is the original DAMamba `loss_unl` machinery applied to SceneShifted source patches as unlabeled data; no shifted-source CE or new loss was added.', '- Use the paired deltas above as the primary mechanism test. A positive result supports SceneShift as an intermediate-domain plugin beyond clean DAMamba CE.']
(ROOT/'summary.md').write_text('\n'.join(lines)+'\n')
