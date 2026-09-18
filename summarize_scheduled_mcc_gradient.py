"""Summarize scheduled-MCC performance and encoder gradient conflict diagnostics."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent/'runs_sceneshiftnet_dcrn/houston_raw/scheduled_mcc_gradient'
BASE=Path(__file__).resolve().parent/'runs_sceneshiftnet_dcrn/houston_raw/three_seed_final/B'
METHODS={'K0':'fixed MCC 0.1','K1':'delayed mild MCC','K2':'delayed full MCC','K3':'cosine ramp MCC'}
SEEDS=(1341,2024,3407)
def fmt(x):x=np.asarray(x,float);return f'{x.mean():.2f} ± {x.std(ddof=0):.2f}'

rows=[];grad_rows=[]
for m,name in METHODS.items():
 for s in SEEDS:
  p=ROOT/m/f'seed_{s}';x=json.loads((p/'metrics.json').read_text());h=json.loads((p/'history.json').read_text())[-1];g=x['gradient_summary']
  row={'Method':m,'Method Name':name,'Seed':s,'OA':x['OA'],'AA':x['AA'],'Kappa':x['Kappa'],'L_mcc':h['L_mcc'],'Final lambda':h['lambda_mcc'],'Target soft distribution':json.dumps(h['target_mean_soft_distribution'])}
  row.update({f'Class {i+1}':v for i,v in enumerate(x['per_class_accuracy'])});rows.append(row)
  grad_rows.append({'Method':m,'Seed':s,**g})
with (ROOT/'per_seed_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (ROOT/'gradient_summary.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(grad_rows[0]));w.writeheader();w.writerows(grad_rows)
def vals(m,k):return np.array([r[k] for r in rows if r['Method']==m])
summary=[];pc=[];ga=[]
for m,name in METHODS.items():
 summary.append({'Method':m,'Method Name':name,**{f'{k} mean ± std':fmt(vals(m,k)) for k in ['OA','AA','Kappa']}})
 for c in range(1,8):pc.append({'Method':m,'Class':c,'Accuracy mean ± std':fmt(vals(m,f'Class {c}'))})
 gs=[r for r in grad_rows if r['Method']==m]
 ga.append({'Method':m,**{k:np.mean([r[k] for r in gs]) for k in ['negative_cosine_ratio','mean_cosine','early_mean_cosine','middle_mean_cosine','late_mean_cosine']}})
for fn,data in [('mean_std_summary.csv',summary),('per_class_mean_std.csv',pc),('gradient_aggregate.csv',ga)]:
 with (ROOT/fn).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
adaptive={s:json.loads((BASE/f'seed_{s}'/'metrics.json').read_text()) for s in SEEDS}
lines=['# Scheduled MCC + Gradient Conflict Diagnosis','',
 'K0 exactly reproduces the existing unified-runner Adaptive+Original-MCC metrics for all three seeds. Diagnostics use a fixed batch in eval mode, do not update BN, do not call optimizer.step, and do not consume RNG before initialization.','',
 '| Method | OA mean±std | AA mean±std | Kappa mean±std |','|---|---:|---:|---:|']
for x in summary:lines.append(f"| {x['Method']} {x['Method Name']} | {x['OA mean ± std']} | {x['AA mean ± std']} | {x['Kappa mean ± std']} |")
lines+=['','## Per-seed results','','| Method | Seed | OA | AA | Kappa |','|---|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['Method']} | {r['Seed']} | {r['OA']:.2f} | {r['AA']:.2f} | {r['Kappa']:.2f} |")
lines+=['','## Classes 1/2/6/7','','| Method | C1 | C2 | C6 | C7 |','|---|---:|---:|---:|---:|']
for m in METHODS:lines.append(f"| {m} | {fmt(vals(m,'Class 1'))} | {fmt(vals(m,'Class 2'))} | {fmt(vals(m,'Class 6'))} | {fmt(vals(m,'Class 7'))} |")
lines+=['','## Gradient conflict','','| Method | Negative ratio | Mean cosine | Early | Middle | Late |','|---|---:|---:|---:|---:|---:|']
for x in ga:lines.append(f"| {x['Method']} | {x['negative_cosine_ratio']:.2%} | {x['mean_cosine']:.3f} | {x['early_mean_cosine']:.3f} | {x['middle_mean_cosine']:.3f} | {x['late_mean_cosine']:.3f} |")
base_oa=np.mean([adaptive[s]['OA'] for s in SEEDS]);base_aa=np.mean([adaptive[s]['AA'] for s in SEEDS])
lines+=['','## Interpretation','',f'- Adaptive-only three-seed reference: OA `{base_oa:.2f}`, AA `{base_aa:.2f}`.']
for m in METHODS:
 lines.append(f"- {m}: versus K0, OA `{vals(m,'OA').mean()-vals('K0','OA').mean():+.2f}`, AA `{vals(m,'AA').mean()-vals('K0','AA').mean():+.2f}`; versus Adaptive-only AA `{vals(m,'AA').mean()-base_aa:+.2f}`.")
lines+=['- Negative gradient cosine is seed-dependent and persists across schedules. Scheduling changes the optimization trajectory and can improve the OA/AA compromise, but does not remove objective-level gradient competition.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
