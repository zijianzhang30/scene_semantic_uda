"""Summarize completed A/B/C Houston three-seed stability experiments."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'runs_sceneshiftnet_dcrn/houston_raw/three_seed_final'
METHODS={'A':'Original SceneShift','B':'Adaptive SceneShift','C':'Adaptive SceneShift + Original MCC'}
SEEDS=(1341,2024,3407)


def fmt(values):
    a=np.asarray(values,float);return f'{a.mean():.6f} ± {a.std(ddof=0):.6f}'


rows=[]
for method,name in METHODS.items():
    for seed in SEEDS:
        m=json.loads((ROOT/method/f'seed_{seed}'/'metrics.json').read_text())
        row={'Method':method,'Method Name':name,'Seed':seed,'OA':m['OA'],'AA':m['AA'],'Kappa':m['Kappa']}
        row.update({f'Class {i+1}':v for i,v in enumerate(m['per_class_accuracy'])});rows.append(row)
with (ROOT/'per_seed_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

summary=[];class_rows=[]
for method,name in METHODS.items():
    current=[r for r in rows if r['Method']==method]
    summary.append({'Method':method,'Method Name':name,**{f'{k} mean ± std':fmt([r[k] for r in current]) for k in ('OA','AA','Kappa')}})
    for c in range(1,8):
        values=np.array([r[f'Class {c}'] for r in current])
        class_rows.append({'Method':method,'Method Name':name,'Class':c,'Mean':values.mean(),'Std':values.std(ddof=0),'Mean ± Std':fmt(values)})
with (ROOT/'mean_std_summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
with (ROOT/'per_class_mean_std.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(class_rows[0]));w.writeheader();w.writerows(class_rows)

def metric(method,key):return np.array([r[key] for r in rows if r['Method']==method])
lines=['# Houston three-seed stability','',
 'All nine experiments were rerun through one deterministic runner. For each seed, A/B/C use identical source samples, target indices, DataLoader construction, model initialization stream, forward count/order, and Adam settings. Reported standard deviations are population SD (`ddof=0`) over the three fixed seeds.','',
 '| Method | OA mean ± std | AA mean ± std | Kappa mean ± std |','|---|---:|---:|---:|']
for s in summary:lines.append(f"| {s['Method']} {s['Method Name']} | {s['OA mean ± std']} | {s['AA mean ± std']} | {s['Kappa mean ± std']} |")
lines+=['','## Per-seed results','','| Method | Seed | OA | AA | Kappa |','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['Method']} | {r['Seed']} | {r['OA']:.2f} | {r['AA']:.2f} | {r['Kappa']:.2f} |")
lines+=['','## Per-class mean ± std','','| Method | C1 | C2 | C3 | C4 | C5 | C6 | C7 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for method,name in METHODS.items():lines.append('| '+method+' | '+' | '.join(fmt(metric(method,f'Class {c}')) for c in range(1,8))+' |')
oa_ab=metric('B','OA')-metric('A','OA');aa_ab=metric('B','AA')-metric('A','AA');oa_ca=metric('C','OA')-metric('A','OA');aa_ca=metric('C','AA')-metric('A','AA');k_ca=metric('C','Kappa')-metric('A','Kappa')
lines+=['','## Interpretation','',
 f'- Adaptive minus Original OA by seed: `{oa_ab.round(4).tolist()}`. Positive in `{int((oa_ab>0).sum())}/3` seeds; mean delta `{oa_ab.mean():.2f}` points.',
 f'- Adaptive minus Original AA by seed: `{aa_ab.round(4).tolist()}`. Negative in `{int((aa_ab<0).sum())}/3` seeds; mean delta `{aa_ab.mean():.2f}` points.',
 f'- Adaptive+MCC minus Original OA/AA/Kappa mean deltas: `{oa_ca.mean():.2f}/{aa_ca.mean():.2f}/{k_ca.mean():.2f}` points.',
 f'- MCC minus Adaptive AA by seed: `{(metric("C","AA")-metric("B","AA")).round(4).tolist()}`.',
 '- Class 1/6/7 stability must be judged from the per-class table and seed rows; a higher mean with a large SD is not considered stable.',
 '',
 'Historical seed-1341 references were not mixed into the aggregate because they came from different runner/DataLoader RNG construction. They remain useful external references: A 77.63/78.03/65.71, B 79.12/74.29/66.69, C 78.78/78.11/66.07. Unified C seed 1341 exactly follows the prior F3 initialization/order and reproduces C; unified A/B are matched controls for this stability study.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
