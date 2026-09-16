"""Houston source-guided discriminative-band preservation ablation."""
import argparse,csv,json,random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_dcrn import SceneShiftNetDCRN
from sceneshiftnet_dcrn_train import PatchDataset,evaluate,file_sha256,sample_source
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from train_sceneshiftnet_residual_mcc import Discriminator,mcc_loss

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs_sceneshiftnet_dcrn/houston_raw/band_preservation'
METHODS={
 'J0':dict(name='Adaptive SceneShift + Original MCC',semantic=False,mcc=True),
 'J1':dict(name='Semantic-Preserving Adaptive SceneShift',semantic=True,mcc=False),
 'J2':dict(name='Semantic-Preserving Adaptive SceneShift + Original MCC',semantic=True,mcc=True),
}


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def minmax(x):return (x-x.min())/(x.max()-x.min()+1e-5)


def fisher_scores(cube,gt,classes=7):
 values=cube[gt>0].astype(np.float64);labels=gt[gt>0].astype(np.int64);overall=values.mean(0);between=np.zeros(values.shape[1]);within=np.zeros(values.shape[1]);total=len(values)
 for c in range(1,classes+1):
  x=values[labels==c];mu=x.mean(0);between+=len(x)*(mu-overall)**2;within+=((x-mu)**2).sum(0)
 return between/total/(within/total+1e-5),between/total,within/total


@torch.no_grad()
def features(model,dataset,labels,device,batch=256):
 model.eval();out=[]
 for begin in range(0,len(dataset),batch):out.append(model.encoder(torch.stack([dataset[i] for i in range(begin,min(begin+batch,len(dataset)))]).to(device)).cpu().numpy())
 f=np.concatenate(out);centroids=np.stack([f[labels==c].mean(0) for c in range(7)]);radii=np.array([np.linalg.norm(f[labels==c]-centroids[c],axis=1).mean() for c in range(7)])
 return f,centroids,radii


def source_shift_diagnostic(model,source,shifted,centers,labels,device):
 fs,cs,rs=features(model,PatchDataset(source,centers,7),labels,device);fh,ch,rh=features(model,PatchDataset(shifted,centers,7),labels,device);rows=[]
 for c in range(7):
  own=float(cs[c]@ch[c]/(np.linalg.norm(cs[c])*np.linalg.norm(ch[c])+1e-12));before=F.normalize(torch.from_numpy(cs),dim=1).numpy()@F.normalize(torch.from_numpy(cs),dim=1).numpy().T;after=F.normalize(torch.from_numpy(ch),dim=1).numpy()@F.normalize(torch.from_numpy(ch),dim=1).numpy().T;b=before[c].copy();a=after[c].copy();b[c]=a[c]=-np.inf
  rows.append(dict(Class=c+1,Centroid_Euclidean_Movement=float(np.linalg.norm(ch[c]-cs[c])),Centroid_Cosine_Movement=1-own,Radius_Before=float(rs[c]),Radius_After=float(rh[c]),Spread_Change=float(rh[c]-rs[c]),Nearest_Class_Before=int(b.argmax()+1),Nearest_Class_After=int(a.argmax()+1),Nearest_Cosine_Before=float(b.max()),Nearest_Cosine_After=float(a.max())))
 geometry=[]
 for i in range(7):
  for j in range(i+1,7):
   before_d=float(np.linalg.norm(cs[i]-cs[j]));after_d=float(np.linalg.norm(ch[i]-ch[j]));geometry.append(dict(Class_i=i+1,Class_j=j+1,Distance_Before=before_d,Distance_After=after_d,Distance_Change=after_d-before_d,Inter_Intra_Before=before_d/((rs[i]+rs[j])/2+1e-12),Inter_Intra_After=after_d/((rh[i]+rh[j])/2+1e-12)))
 return rows,geometry


def run(args):
 spec=METHODS[args.method];seed_all(args.seed);suffix=f'seed_{args.seed}' if args.epochs==100 else f'seed_{args.seed}_smoke_{args.epochs}ep';out=OUT/args.method/suffix;out.mkdir(parents=True,exist_ok=False)
 source,sg,target,tg,paths=load_cubes('none');source=source.astype('float32');target=target.astype('float32');centers,labels=sample_source(sg,7,180,np.random.RandomState(args.seed));tc=np.argwhere(tg>0);target_labels=tg[tc[:,0],tc[:,1]].astype('int64')-1
 generator=torch.Generator().manual_seed(args.seed);sl=DataLoader(PatchDataset(source,centers,7,labels),32,shuffle=True,drop_last=True,generator=generator);tl=DataLoader(PatchDataset(target,tc,7),32,shuffle=True,drop_last=True,generator=generator);el=DataLoader(PatchDataset(target,tc,7,target_labels),32,shuffle=False)
 sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0);d_dom=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5);d_dom_norm=minmax(d_dom);alpha_base=(.4+.5*d_dom_norm).astype(np.float32);d_sem,between,within=fisher_scores(source,sg);d_sem_norm=minmax(d_sem);alpha_sem=np.clip(alpha_base*(1-.3*d_sem_norm),.2,.9).astype(np.float32);alpha=alpha_sem if spec['semantic'] else alpha_base
 config=dict(method=args.method,method_name=spec['name'],seed=args.seed,epochs=args.epochs,batch_size=32,patch_size=7,source_per_class=180,optimizer='Adam',lr=.001,weight_decay=0,normalization='none',use_ilda=False,deterministic=True,forward_order='source -> shifted-source -> target, once each',scene_shift='semantic_preserving_adaptive' if spec['semantic'] else 'adaptive',alpha_min=.2 if spec['semantic'] else .4,alpha_max=.9,lambda_sem=.3 if spec['semantic'] else 0,semantic_statistics='all source_gt>0 labeled source pixels only',original_mcc=spec['mcc'],temperature=2.5,lambda_mcc=.1 if spec['mcc'] else 0,target_hard_pseudo_labels=False,target_gt_usage='support mask and final evaluation only',loss='L_src + .5 L_shift'+(' + .1 L_mcc' if spec['mcc'] else ''),code_sha256=file_sha256(__file__),model_sha256=file_sha256(HERE/'models/sceneshift_net_dcrn.py'),data_hash={str(p):file_sha256(p) for p in paths})
 (out/'config.json').write_text(json.dumps(config,indent=2));np.savez(out/'indices.npz',source=centers,source_labels=labels,target=tc)
 band_rows=[dict(Band=b,Domain_Discrepancy=float(d_dom[b]),Domain_Normalized=float(d_dom_norm[b]),Semantic_Fisher=float(d_sem[b]),Semantic_Normalized=float(d_sem_norm[b]),Between_Class_Variance=float(between[b]),Within_Class_Variance=float(within[b]),Alpha_Base=float(alpha_base[b]),Alpha_Final=float(alpha[b]),Alpha_Reduction=float(alpha_base[b]-alpha[b])) for b in range(48)]
 with (out/'band_statistics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=band_rows[0]);w.writeheader();w.writerows(band_rows)
 (out/'band_statistics.json').write_text(json.dumps(band_rows,indent=2))
 assert np.isfinite(alpha).all() and np.isfinite(d_sem).all() and (alpha>=.2).all() and (alpha<=.9).all();assert torch.cuda.is_available();device='cuda';model=SceneShiftNetDCRN(48,7,7).to(device);unused_discriminator=Discriminator().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001);sm_t,ss_t,tm_t,ts_t,alpha_t=[torch.tensor(v,device=device)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)];history=[]
 for epoch in range(1,args.epochs+1):
  model.train();target_it=iter(tl);sums=np.zeros(3);correct=np.zeros(2,dtype=int);n=0;soft_sum=np.zeros(7)
  for x,y in sl:
   try:z=next(target_it)
   except StopIteration:target_it=iter(tl);z=next(target_it)
   x,y,z=x.to(device),y.to(device),z.to(device);fs,ls=model(x);shifted=(x-sm_t)/(ss_t+1e-5)*(alpha_t*ts_t+(1-alpha_t)*ss_t)+alpha_t*tm_t+(1-alpha_t)*sm_t;fss,lss=model(shifted);ft,lt=model(z);raw_mcc,prob,conf=mcc_loss(lt);src_loss=F.cross_entropy(ls,y);shift_loss=F.cross_entropy(lss,y);loss=src_loss+.5*shift_loss+(.1*raw_mcc if spec['mcc'] else 0)
   if not torch.isfinite(loss) or not torch.isfinite(fss).all() or not torch.isfinite(ft).all():raise RuntimeError(f'nonfinite epoch {epoch}')
   optimizer.zero_grad();loss.backward();optimizer.step();bs=len(y);n+=bs;sums+=np.array([src_loss.item(),shift_loss.item(),raw_mcc.item()])*bs;correct+=[int((ls.argmax(1)==y).sum()),int((lss.argmax(1)==y).sum())];soft_sum+=prob.detach().sum(0).cpu().numpy()
  row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_mcc=sums[2]/n if spec['mcc'] else 0,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,target_mean_soft_distribution=(soft_sum/n).tolist(),alpha_min=float(alpha.min()),alpha_mean=float(alpha.mean()),alpha_max=float(alpha.max()),finite=True);history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 torch.save({'model':model.state_dict(),'config':config,'seed':args.seed},out/'final.pth');metrics=evaluate(model,el,device,7);metrics.update(seed=args.seed,method=args.method,method_name=spec['name'],source_accuracy=history[-1]['source_accuracy'],shifted_source_accuracy=history[-1]['shifted_source_accuracy'],L_mcc=history[-1]['L_mcc']);shifted_cube=(source-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm;response,geometry=source_shift_diagnostic(model,source,shifted_cube,centers,labels,device);metrics['source_shift_response']=response;metrics['source_geometry']=geometry;(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
 with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 print(json.dumps(metrics),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=1341);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
