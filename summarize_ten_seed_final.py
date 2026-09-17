"""Summarize matched Houston DCRN A/B/C ten-seed experiments."""
import csv, json
from pathlib import Path
import numpy as np
from scipy import stats

ROOT=Path(__file__).resolve().parent/'runs_sceneshiftnet_dcrn/houston_raw/ten_seed_final'
METHODS={'A':'Original SceneShift','B':'Adaptive SceneShift','C':'Adaptive SceneShift + Original MCC'}
SEEDS=(1341,2024,3407,1174,1622,42,340,777,1024,2026)

def fmt(x):
    x=np.asarray(x,float);return f'{x.mean():.4f} ± {x.std(ddof=0):.4f}'

rows=[]
for method,name in METHODS.items():
    for seed in SEEDS:
        p=ROOT/method/f'seed_{seed}';m=json.loads((p/'metrics.json').read_text());h=json.loads((p/'history.json').read_text())[-1]
        row={'Method':method,'Method Name':name,'Seed':seed,'OA':m['OA'],'AA':m['AA'],'Kappa':m['Kappa'],
             'Source Accuracy':h['source_accuracy']*100,'Shifted-Source Accuracy':h['shifted_source_accuracy']*100}
        row.update({f'Class {i+1}':v for i,v in enumerate(m['per_class_accuracy'])});rows.append(row)
with (ROOT/'per_seed_results.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def values(method,key):return np.array([next(r[key] for r in rows if r['Method']==method and r['Seed']==s) for s in SEEDS])
summary=[];class_rows=[]
for method,name in METHODS.items():
    summary.append({'Method':method,'Method Name':name,**{f'{k} mean ± std':fmt(values(method,k)) for k in ('OA','AA','Kappa')}})
    for c in range(1,8):
        x=values(method,f'Class {c}');class_rows.append({'Method':method,'Method Name':name,'Class':c,'Mean':x.mean(),'Std':x.std(ddof=0),'Mean ± Std':fmt(x)})
for filename,data in [('mean_std_summary.csv',summary),('per_class_mean_std.csv',class_rows)]:
    with (ROOT/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

tests=[]
for left,right in [('A','B'),('B','C'),('A','C')]:
    for metric in ('OA','AA','Kappa'):
        x,y=values(left,metric),values(right,metric);d=y-x
        sh=stats.shapiro(d)
        if sh.pvalue<.05:
            test='Wilcoxon signed-rank';res=stats.wilcoxon(y,x,alternative='two-sided',zero_method='wilcox');stat,p=res.statistic,res.pvalue
        else:
            test='paired t-test';res=stats.ttest_rel(y,x);stat,p=res.statistic,res.pvalue
        tests.append({'Comparison':f'{left} vs {right}','Metric':metric,'Direction':f'{right} - {left}',
          'Mean Difference':d.mean(),'Std Difference':d.std(ddof=0),'Positive Seeds':int((d>0).sum()),'Negative Seeds':int((d<0).sum()),
          'Shapiro-Wilk W':sh.statistic,'Shapiro-Wilk p':sh.pvalue,'Selected Test':test,'Statistic':stat,'p-value':p,'Significant at 0.05':bool(p<.05)})
with (ROOT/'significance_tests.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(tests[0]));w.writeheader();w.writerows(tests)

def delta(l,r,k):return values(r,k)-values(l,k)
lines=['# Houston DCRN ten-seed stability','',
 'All 30 runs use the matched deterministic runner. Within each seed A/B/C use identical source and target indices, DataLoader construction, initialization stream, forward order, and Adam settings. Values are final-epoch results; standard deviations are population SD (`ddof=0`).','',
 '| Method | OA mean ± std | AA mean ± std | Kappa mean ± std |','|---|---:|---:|---:|']
for s in summary:lines.append(f"| {s['Method']} {s['Method Name']} | {s['OA mean ± std']} | {s['AA mean ± std']} | {s['Kappa mean ± std']} |")
lines+=['','## Per-seed results','','| Method | Seed | OA | AA | Kappa |','|---|---:|---:|---:|---:|']
for method in METHODS:
 for seed in SEEDS:
  r=next(r for r in rows if r['Method']==method and r['Seed']==seed);lines.append(f"| {method} | {seed} | {r['OA']:.2f} | {r['AA']:.2f} | {r['Kappa']:.2f} |")
lines+=['','## Per-class mean ± std','','| Method | C1 | C2 | C3 | C4 | C5 | C6 | C7 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for method in METHODS:lines.append('| '+method+' | '+' | '.join(fmt(values(method,f'Class {c}')) for c in range(1,8))+' |')
lines+=['','## Paired comparisons','',
 f"- B−A OA is positive in `{int((delta('A','B','OA')>0).sum())}/10` seeds; deltas `{delta('A','B','OA').round(3).tolist()}`; mean `{delta('A','B','OA').mean():+.2f}`.",
 f"- C−B AA is positive in `{int((delta('B','C','AA')>0).sum())}/10` seeds; deltas `{delta('B','C','AA').round(3).tolist()}`; mean `{delta('B','C','AA').mean():+.2f}`.",
 f"- C−A mean OA/AA/Kappa deltas are `{delta('A','C','OA').mean():+.2f}/{delta('A','C','AA').mean():+.2f}/{delta('A','C','Kappa').mean():+.2f}`.",
 '', '| Comparison | Metric | Mean difference | Positive | Test | p-value |', '|---|---|---:|---:|---|---:|']
for t in tests:lines.append(f"| {t['Direction']} | {t['Metric']} | {t['Mean Difference']:+.3f} | {t['Positive Seeds']}/10 | {t['Selected Test']} | {t['p-value']:.6f} |")
c1=delta('B','C','Class 1');c6=delta('B','C','Class 6');c7=delta('B','C','Class 7')
lines+=['','## Findings','',
 f"- MCC C1 change relative to Adaptive is positive in `{int((c1>0).sum())}/10` seeds (mean `{c1.mean():+.2f}`, SD `{c1.std(ddof=0):.2f}`).",
 f"- MCC C6/C7 changes are positive in `{int((c6>0).sum())}/10` and `{int((c7>0).sum())}/10` seeds; their SDs are `{c6.std(ddof=0):.2f}` and `{c7.std(ddof=0):.2f}`, respectively.",
 '- Statistical-test selection was made separately for each paired metric: Shapiro–Wilk was applied to paired differences; Wilcoxon was used when p<0.05, otherwise a paired t-test was used. No seed was removed.']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
