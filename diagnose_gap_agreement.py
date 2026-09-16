"""Offline paired checkpoint diagnosis; no training or checkpoint writes."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_sceneshiftnet_gap_learning import reliable
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from sceneshiftnet_dcrn_train import PatchDataset,file_sha256
from models.sceneshift_net_dcrn import SceneShiftNetDCRN
ROOT=Path(__file__).resolve().parent/'runs_sceneshiftnet_dcrn/houston_raw'
OUT=ROOT/'agreement_offline_seed1341'
def write(p,rows):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def stats(x):return dict(mean=float(np.mean(x)),median=float(np.median(x)),q10=float(np.quantile(x,.1)),q90=float(np.quantile(x,.9))) if len(x) else None
def main():
 OUT.mkdir(exist_ok=False)
 s,sg,t,tg,paths=load_cubes('none');s=s.astype('float32');t=t.astype('float32')
 sm,ss=s.reshape(-1,48).mean(0),s.reshape(-1,48).std(0);tm,ts=t.reshape(-1,48).mean(0),t.reshape(-1,48).std(0)
 configs={};comparisons=[];classes=[];groups=[];c1=[]
 ref=np.load(ROOT/'gap_partition_diag/seed_1341/indices.npz');tc=ref['target'];sc=ref['source'];sy=ref['source_labels']
 # Reproducible fixed source batches shared across checkpoints; not GT-derived target sampling.
 rng=np.random.RandomState(1341);order=rng.permutation(len(sc));sc=sc[order];sy=sy[order]
 np.savez(OUT/'pairing.npz',source=sc,source_labels=sy,target=tc)
 source_batches=list(DataLoader(PatchDataset(s,sc,7,sy),32,shuffle=False,drop_last=True))
 target_loader=DataLoader(PatchDataset(t,tc,7),32,shuffle=False,drop_last=False)
 st=[torch.tensor(v,device='cuda')[None,:,None,None] for v in (sm,ss,tm,ts)];sm,ss,tm,ts=st
 for exp,folder in [('E1','gap_partition_diag'),('E2','smallgap_inter'),('E3','customized_gap_learning')]:
  p=ROOT/folder/'seed_1341';cfg=json.loads((p/'config.json').read_text());assert cfg['normalization']=='none' and not cfg['use_ilda'] and cfg['alpha']==.8
  ix=np.load(p/'indices.npz');assert np.array_equal(ix['target'],tc) and np.array_equal(ix['source'],ref['source'])
  model=SceneShiftNetDCRN(48,7,7).cuda();model.load_state_dict(torch.load(p/'final.pth',map_location='cpu')['model']);model.eval();rows=[]
  with torch.no_grad():
   shifted_cache=[]
   for x,y in source_batches:
    x=x.cuda();shift=(x-sm)/(ss+1e-5)*(.8*ts+.2*ss)+.8*tm+.2*sm;shifted_cache.append((model(shift)[0],y.cuda()))
   offset=0
   for b,x in enumerate(target_loader):
    ft,logits=model(x.cuda());fss,y=shifted_cache[b%len(shifted_cache)]
    q,ag,small,pred,proto,cos,gap=reliable(ft,fss,y,logits,True);conf=logits.softmax(1).max(1).values
    arrays=[v.cpu().numpy() for v in (q,ag,small,pred,proto,cos,gap,conf)]
    for j in range(len(x)):
     k=offset+j;r,c=tc[k];qq,aa,ssmall,pr,pt,co,ga,cf=[v[j] for v in arrays]
     rows.append(dict(sample=k,row=int(r),column=int(c),true_label=int(tg[r,c]),classifier_label=int(pr)+1,classifier_confidence=float(cf),prototype_label=int(pt)+1,prototype_top1_cosine=float(co),prototype_margin=float(ga),agreement=bool(aa),small_gap=bool(ssmall),large_gap=bool(qq and not aa),source_batch=b%len(shifted_cache)))
    offset+=len(x)
  write(OUT/(exp+'_samples.csv'),rows)
  truth=np.array([r['true_label'] for r in rows]);pred=np.array([r['classifier_label'] for r in rows]);proto=np.array([r['prototype_label'] for r in rows]);agree=pred==proto;correct=pred==truth;small=np.array([r['small_gap'] for r in rows]);large=np.array([r['large_gap'] for r in rows]);false=agree & ~correct
  for c in range(1,8):
   actual=truth==c;selected=small&(pred==c);lg=large&(pred==c);tp=(selected&actual).sum()
   classes.append(dict(experiment=exp,cls=c,true_count=int(actual.sum()),small_gap_true_count=int((small&actual).sum()),small_gap_predicted_count=int(selected.sum()),small_precision=float(tp/selected.sum()) if selected.any() else None,small_recall=float(tp/actual.sum()),large_precision=float((lg&actual).sum()/lg.sum()) if lg.any() else None,agreement_ratio=float(agree[actual].mean()),false_agreement_count=int((false&actual).sum()),false_agreement_fraction_true_class=float(false[actual].mean()),classifier_confusion=np.bincount(pred[actual],minlength=8)[1:].tolist(),prototype_confusion=np.bincount(proto[actual],minlength=8)[1:].tolist()))
  for scope,sel in [('all',np.ones(len(rows),bool)),('true_class1',truth==1)]:
   for name,mask in [('correct_agreement',agree&correct),('false_agreement',false),('small_correct',small&correct),('small_false',small&~correct)]:
    take=sel&mask
    groups.append(dict(experiment=exp,scope=scope,group=name,count=int(take.sum()),**{key:stats(np.array([r[key] for r in rows])[take]) for key in ['classifier_confidence','prototype_top1_cosine','prototype_margin']}))
  one=truth==1
  c1.append(dict(experiment=exp,total=int(one.sum()),small_count=int((one&small).sum()),small_classifier_confusion=np.bincount(pred[one&small],minlength=8)[1:].tolist(),false_agreement_count=int((one&false).sum()),false_agreement_ratio=float(false[one].mean()),false_fraction_among_agreement=float((one&false).sum()/(one&agree).sum()) if (one&agree).any() else None))
  comparisons.append(dict(experiment=exp,OA=float(correct.mean()*100),small_ratio=float(small.mean()),large_ratio=float(large.mean()),agreement_ratio=float(agree.mean()),false_agreement_count=int(false.sum()),false_agreement_ratio=float(false.mean()),false_fraction_among_agreement=float(false.sum()/agree.sum()),small_precision=float(correct[small].mean()),**{'class1_acc':float(correct[one].mean()*100)}))
  configs[exp]=dict(config=cfg,checkpoint_sha256=file_sha256(p/'final.pth'))
  print(exp,comparisons[-1],flush=True)
 write(OUT/'classwise.csv',classes);write(OUT/'comparison.csv',comparisons)
 (OUT/'class1_confusion.json').write_text(json.dumps(c1,indent=2));(OUT/'agreement_statistics.json').write_text(json.dumps(groups,indent=2))
 (OUT/'provenance.json').write_text(json.dumps(dict(checkpoints=configs,source_pairing='seed1341 permutation of fixed training indices; 39 full batches cycled against sequential target batches; same for E1/E2/E3',space='eval-mode checkpoint features; 288d',target_gt='only post-inference diagnostic',precision='group selected predicted class correct / group selected predicted class',recall='group selected correct class / all true class',note='Offline full target eval; differs from train-mode minibatch statistics and training epoch ratios',script_sha256=file_sha256(__file__)),indent=2))
if __name__=='__main__':main()
