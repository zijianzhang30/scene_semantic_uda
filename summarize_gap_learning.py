"""Read-only result aggregation; ratios are final-epoch observed target fractions."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'runs_sceneshiftnet_dcrn/houston_raw'
rows=[];classes=[]
for name,folder in [('A Original','sceneshift'),('B Adaptive','adaptive_shift'),('Previous reliability','adaptive_reliable'),('E1','gap_partition_diag'),('E2','smallgap_inter'),('E3','customized_gap_learning')]:
    p=ROOT/folder/'seed_1341';mp=p/'metrics.json'
    if not mp.exists(): continue
    m=json.loads(mp.read_text());h=json.loads((p/'history.json').read_text())[-1]
    small=h.get('small_gap_ratio',h.get('reliable_target_ratio'));large=h.get('large_gap_ratio')
    if large is None and small is not None:large=h['pseudo_label_coverage']-small
    rows.append({'Method':name,**{k:m[k] for k in ['OA','AA','Kappa','source_accuracy','shifted_source_accuracy']},'Small-gap Ratio':small,'Large-gap Ratio':large,'small_precision':h.get('small_precision'),'large_precision':h.get('large_precision'),'pseudo_histogram':m['pseudo_label_class_histogram']})
    classes.extend({'Method':name,'Class':c+1,'Accuracy':v} for c,v in enumerate(m['per_class_accuracy']))
for fname,data in [('gap_learning_summary.csv',rows),('gap_learning_per_class.csv',classes)]:
    with (ROOT/fname).open('w') as f:
        w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
print(json.dumps(rows,indent=2))
