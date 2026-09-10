import json,csv
from pathlib import Path
import numpy as np, torch
import sys
sys.path.insert(0,str(Path(__file__).parent))
from run_cross_scene_clean import load_dataset, split_source, patch_batch, scene_shift, summarize

ROOT=Path(__file__).parent
def fix_stats(name,outdir):
 source,target,sg,tg,c=load_dataset(name); ncls=c['classes']; tr,ty,va,vy=split_source(sg,ncls,1174); tx=patch_batch(source,tr,c['patch']); sf=source.reshape(-1,source.shape[-1]); tf=target.reshape(-1,target.shape[-1]); sm,ss,tm,ts=sf.mean(0),sf.std(0),tf.mean(0),tf.std(0); sample=tx[:min(512,len(tx))]; torch.manual_seed(1951); pre,sh=scene_shift(torch.from_numpy(sample),sm,ss,tm,ts,.8,return_pre=True); pre=pre.numpy(); sh=sh.numpy(); stats={'alpha':.8,'pre_clamp_lt0_ratio':float((pre<0).mean()),'pre_clamp_gt1_ratio':float((pre>1).mean()),'post_clamp_eq0_ratio':float((sh<=0).mean()),'post_clamp_eq1_ratio':float((sh>=1).mean()),'raw_source_range':summarize(sample),'pre_clamp_range':summarize(pre),'shifted_range':summarize(sh),'mean_gap_after_shift':float(np.mean(np.abs(sample.mean((0,2,3))-sh.mean((0,2,3))))),'std_gap_after_shift':float(np.mean(np.abs(sample.std((0,2,3))-sh.std((0,2,3)))))}
 for m in ['ce','scene_shift']:(ROOT/outdir/m/'scene_shift_stats.json').write_text(json.dumps(stats,indent=2))
def main():
 for n,d in [('pavia','runs_pavia_scene_shift'),('shanghai_hangzhou','runs_shanghai_hangzhou_scene_shift')]: fix_stats(n,d)
 rows=[]
 for ds,d in [('PaviaU→PaviaC','runs_pavia_scene_shift'),('Shanghai→Hangzhou','runs_shanghai_hangzhou_scene_shift')]:
  ce=json.load(open(ROOT/(d+'/ce/result.json'))); ss=json.load(open(ROOT/(d+'/scene_shift/result.json')))
  for m,r in [('DCRN + CE',ce),('DCRN + Scene Shift α=0.8',ss)]: rows.append((ds,m,r))
  with (ROOT/d/'per_class.csv').open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['method','class','accuracy']);[w.writerow([m,i,a]) for m,r in [('ce',ce),('scene_shift',ss)] for i,a in enumerate(r['per_class_accuracy'],1)]
  md=[f'# {ds} clean smoke test (seed 1174)','', '| Method | OA | AA | Kappa | Best epoch |','|---|---:|---:|---:|---:|']
  for m,r in [('DCRN + CE',ce),('DCRN + Scene Shift α=0.8',ss)]: md.append(f"| {m} | {r['oa']*100:.2f} | {r['aa']*100:.2f} | {r['kappa']*100:.2f} | {r['best_epoch']} |")
  md += ['',f"Delta: OA {(ss['oa']-ce['oa'])*100:+.2f} pp; AA {(ss['aa']-ce['aa'])*100:+.2f} pp; Kappa {(ss['kappa']-ce['kappa'])*100:+.2f} pp.",'','| Class | CE | Scene Shift | Delta |','|---:|---:|---:|---:|']
  for i,(a,b) in enumerate(zip(ce['per_class_accuracy'],ss['per_class_accuracy']),1):md.append(f'| C{i} | {a*100:.2f} | {b*100:.2f} | {(b-a)*100:+.2f} |')
  (ROOT/d/'summary.md').write_text('\n'.join(md)+'\n')
 md=['# Cross-scene clean Scene Shift summary (seed 1174)','', '| Dataset | Method | OA | AA | Kappa |','|---|---|---:|---:|---:|']
 for ds,m,r in rows: md.append(f'| {ds} | {m} | {r["oa"]*100:.2f} | {r["aa"]*100:.2f} | {r["kappa"]*100:.2f} |')
 md += ['', '## Paired deltas', '', '- PaviaU→PaviaC: Scene Shift − CE = **+8.55 OA**, **+5.37 AA**, **+8.98 Kappa**.', '- Shanghai→Hangzhou: Scene Shift − CE = **−3.15 OA**, **−2.34 AA**, **−5.05 Kappa**.', '', 'The α=0.8 zero-tuning transfer is positive on Pavia but negative on Shanghai→Hangzhou under the current clean protocol. Both datasets are band-aligned and use the official per-scene standardization. Because standardized values are approximately zero-mean/unit-variance, the existing clamp(0,1) truncates a large fraction of shifted values; this is a major implementation/protocol caveat, especially for Shanghai. Target GT is used only for final post-hoc metrics.']
 (ROOT/'cross_scene_summary.md').write_text('\n'.join(md)+'\n')
if __name__=='__main__':main()
