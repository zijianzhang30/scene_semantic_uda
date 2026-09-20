"""Population standard deviation (ddof=0); partial results explicitly counted."""
import json,csv,os
from pathlib import Path
import numpy as np
from train_v04_optimization_stability import SEEDS
base=Path(__file__).resolve().parent/'runs_v04_optimization_stability'
rows=[]; per=[]; summary={}
for s in SEEDS:
    p=base/f'seed_{s}/history.json'
    if not p.exists(): continue
    h=json.loads(p.read_text())
    for e in (30,60,80,100):
        for r in h:
            if r['epoch']==e: rows.append({'seed':s,'epoch':e,**{k:r[k] for k in ('oa','aa','kappa')}}); per.append((s,e,r['per_class']))
    if h[-1]['epoch']==100:
        r=max(h,key=lambda x:x['oa']); rows.append({'seed':s,'epoch':'best','oa':r['oa'],'aa':r['aa'],'kappa':r['kappa'],'best_epoch':r['epoch']})
for e in (30,60,80,100,'best'):
    rr=[r for r in rows if r['epoch']==e]
    if not rr: continue
    summary[str(e)]={'n':len(rr),'complete_10seeds':len(rr)==10,**{k:{'mean':float(np.mean([r[k] for r in rr])),'std':float(np.std([r[k] for r in rr]))} for k in ('oa','aa','kappa')}}
    if e=='best': summary[str(e)]['best_epochs']={r['seed']:r['best_epoch'] for r in rr}
    else:
        arr=np.asarray([v for _,ep,v in per if ep==e]); summary[str(e)]['per_class_mean']=arr.mean(0).tolist(); summary[str(e)]['per_class_std']=arr.std(0).tolist()
tmp=base/f'summary.{os.getpid()}.tmp'; tmp.write_text(json.dumps(summary,indent=2)); tmp.replace(base/'summary.json')
tmp=base/f'summary.{os.getpid()}.csv.tmp'
with tmp.open('w') as f:
    w=csv.DictWriter(f,fieldnames=['seed','epoch','oa','aa','kappa','best_epoch']); w.writeheader(); w.writerows(rows)
tmp.replace(base/'summary.csv')
tmp=base/f'per_class.{os.getpid()}.tmp'
with tmp.open('w') as f:
    w=csv.writer(f); w.writerow(['epoch','n','class','mean','std_ddof0'])
    for e in (30,60,80,100):
        if str(e) not in summary: continue
        st=summary[str(e)]
        for c,(m,s) in enumerate(zip(st['per_class_mean'],st['per_class_std'])): w.writerow([e,st['n'],c,m,s])
tmp.replace(base/'per_class_summary.csv')
print(json.dumps(summary,indent=2))
