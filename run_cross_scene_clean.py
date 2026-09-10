"""Clean DCRN cross-scene smoke/benchmark runner for Pavia and SH-HZ.

Only CE and the validated global per-band Scene Shift are used. Target labels
are loaded only by the final evaluation path; source validation selects the
checkpoint. Scene Shift is intentionally unchanged, including clamp(0, 1).
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import scale
from sklearn import metrics
import torch.nn.functional as F
import train as clean
import utils
from model import DCRNClassifier

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent
CFG={
 'pavia': {'source':'Pavia/paviaU.mat','target':'Pavia/pavia.mat','source_gt':'Pavia/paviaU_gt_7.mat','target_gt':'Pavia/pavia_gt_7.mat','source_key':'paviaU','target_key':'pavia','source_gt_key':'paviaU_gt_7','target_gt_key':'pavia_gt_7','patch':11,'classes':7},
 'shanghai_hangzhou': {'file':'Shanghai-Hangzhou/DataCube.mat','patch':1,'classes':3}
}

def seed_all(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def standardize(x):
 y=scale(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype(np.float32)
 return y

def load_dataset(name):
 c=CFG[name]; droot=ROOT/'datasets'
 if name=='pavia':
  s=sio.loadmat(droot/c['source_key'].replace('paviaU','Pavia/paviaU') if False else droot/c['source'])
  t=sio.loadmat(droot/c['target']); sg=sio.loadmat(droot/c['source_gt']); tg=sio.loadmat(droot/c['target_gt'])
  source, target, source_gt, target_gt=s[c['source_key']],t[c['target_key']],sg[c['source_gt_key']],tg[c['target_gt_key']]
 else:
  z=sio.loadmat(droot/c['file']); source,target,source_gt,target_gt=z['DataCube1'],z['DataCube2'],z['gt1'],z['gt2']
 source=standardize(source.astype(np.float32)); target=standardize(target.astype(np.float32))
 return source,target,source_gt.astype(np.int64),target_gt.astype(np.int64),c

def split_source(gt, nclass, seed, per_class=180):
 rng=np.random.RandomState(seed); tr=[]; va=[]
 for cls in range(1,nclass+1):
  ij=np.argwhere(gt==cls); rng.shuffle(ij); tr.append(ij[:per_class]); va.append(ij[per_class:])
 tr=np.concatenate(tr); va=np.concatenate(va); rng.shuffle(tr); rng.shuffle(va)
 return tr,gt[tr[:,0],tr[:,1]]-1,va,gt[va[:,0],va[:,1]]-1

def patch_batch(cube, centers, width):
 h=width//2; pad=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant'); off=np.arange(width)-h
 rows=centers[:,0,None]+h+off[None,:]; cols=centers[:,1,None]+h+off[None,:]
 # B,H,W,C -> B,C,H,W
 return pad[rows[:,:,None],cols[:,None,:],:].transpose(0,3,1,2).astype(np.float32)

def scene_shift(x, sm, ss, tm, ts, alpha=.8, return_pre=False, clamp=True):
 sm=torch.as_tensor(sm,device=x.device,dtype=x.dtype)[None,:,None,None]; ss=torch.as_tensor(ss,device=x.device,dtype=x.dtype)[None,:,None,None]
 tm=torch.as_tensor(tm,device=x.device,dtype=x.dtype)[None,:,None,None]; ts=torch.as_tensor(ts,device=x.device,dtype=x.dtype)[None,:,None,None]
 y=(x-sm)/(ss+1e-5); y=y*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
 scale=1+.04*torch.randn(x.size(0),1,1,1,device=x.device); noise=F.avg_pool2d(torch.randn_like(y),5,1,2)
 pre = y*scale+.015*noise
 out = pre.clamp(0,1) if clamp else pre
 return (pre, out) if return_pre else out

def summarize(v):
 v=np.asarray(v,float); return {'mean':float(v.mean()),'std':float(v.std()),'min':float(v.min()),'max':float(v.max())}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=list(CFG),required=True); p.add_argument('--method',choices=['ce','scene_shift'],required=True); p.add_argument('--seed',type=int,default=1174); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=.002); p.add_argument('--device',default='cuda:0'); p.add_argument('--output',type=Path,required=True); p.add_argument('--no-clamp',action='store_true'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); seed_all(a.seed)
 dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); source,target,sg,tg,c=load_dataset(a.dataset); width=c['patch']; nclass=c['classes']; tr,ty,va,vy=split_source(sg,nclass,a.seed)
 tx=patch_batch(source,tr,width); vx=patch_batch(source,va,width); sf=source.reshape(-1,source.shape[-1]); tf=target.reshape(-1,target.shape[-1]); sm,ss,tm,ts=sf.mean(0),sf.std(0),tf.mean(0),tf.std(0)
 train_loader=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty)),batch_size=a.batch_size,shuffle=True,drop_last=True); val_loader=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy)),batch_size=a.batch_size,shuffle=False)
 model=DCRNClassifier(bands=source.shape[-1],patch_size=width,classes=nclass).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); best={'val_acc':-1}; hist=[]; t0=time.time()
 for ep in range(1,a.epochs+1):
  model.train(); loss_sum=correct=seen=0; pre=[]; post=[]
  for x,y in train_loader:
   x,y=x.to(dev),y.to(dev); z=model(clean.augment(x)); loss=ce(z,y)
   if a.method=='scene_shift':
    torch.manual_seed(a.seed*10000+ep); sh=scene_shift(x,sm,ss,tm,ts,.8,clamp=not a.no_clamp); pre.append(sh.detach().cpu().numpy()); post.append(sh.detach().cpu().numpy()); loss=loss+.5*ce(model(clean.augment(sh)),y)
   opt.zero_grad(); loss.backward(); opt.step(); loss_sum+=float(loss.detach())*len(y); correct+=(z.argmax(1)==y).sum().item(); seen+=len(y)
  model.eval(); vl=vc=vn=0
  with torch.no_grad():
   for x,y in val_loader:
    z=model(x.to(dev)); yy=y.to(dev); vl+=ce(z,yy).item()*len(y); vc+=(z.argmax(1)==yy).sum().item(); vn+=len(y)
  row={'epoch':ep,'train_loss':loss_sum/seen,'train_acc':correct/seen,'val_loss':vl/vn,'val_acc':vc/vn}; hist.append(row); print(json.dumps(row),flush=True)
  if row['val_acc']>best['val_acc']:
   best=row.copy(); torch.save({'model':model.state_dict(),'dataset':a.dataset,'method':a.method,'seed':a.seed,'patch_size':width,'bands':source.shape[-1],'classes':nclass,'scene_shift_alpha':.8 if a.method=='scene_shift' else 0.,'clamp':not a.no_clamp,'target_gt_used_for_training_or_selection':False,'best':best},a.output/'best.pth')
 # Post-hoc target evaluation in bounded batches.
 ck=torch.load(a.output/'best.pth',map_location='cpu'); model.load_state_dict(ck['model']); model.eval(); centers=np.argwhere(tg>0); y=tg[centers[:,0],centers[:,1]]-1; pred=[]
 with torch.no_grad():
  for i in range(0,len(centers),512): pred.append(model(torch.from_numpy(patch_batch(target,centers[i:i+512],width)).to(dev)).argmax(1).cpu().numpy())
 pred=np.concatenate(pred); cm=metrics.confusion_matrix(y,pred,labels=np.arange(nclass)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
 # Distribution/statistics diagnostics; shifted source sampled through all training centers.
 rng=np.random.RandomState(a.seed); sample=tx[rng.choice(len(tx),min(len(tx),512),replace=False)]; x=torch.from_numpy(sample).to(dev); torch.manual_seed(a.seed+777); pre_t,sh=scene_shift(x,sm,ss,tm,ts,.8,return_pre=True,clamp=not a.no_clamp); pre_n=pre_t.cpu().numpy(); shn=sh.cpu().numpy(); raw=sample
 ds={'source_shape':list(source.shape),'target_shape':list(target.shape),'source_bands':int(source.shape[-1]),'target_bands':int(target.shape[-1]),'source_classes':nclass,'target_labeled_classes':sorted(np.unique(tg[tg>0]).astype(int).tolist()),'normalization':'per-scene sklearn.preprocessing.scale (official Pavia/cubeData recipe)','patch_width':width,'source_train_per_class':180,'band_aligned':bool(source.shape[-1]==target.shape[-1]),'source_mean':summarize(sm),'source_std':summarize(ss),'target_mean':summarize(tm),'target_std':summarize(ts),'mean_abs_mean_gap':float(np.mean(np.abs(sm-tm))),'mean_abs_std_gap':float(np.mean(np.abs(ss-ts)))}
 preclip=((pre_n<0).mean(),(pre_n>1).mean()); clipped=((shn<=0).mean(),(shn>=1).mean()); ssout={'alpha':.8,'pre_clamp_lt0_ratio':float(preclip[0]),'pre_clamp_gt1_ratio':float(preclip[1]),'post_clamp_eq0_ratio':float(clipped[0]),'post_clamp_eq1_ratio':float(clipped[1]),'raw_source_range':summarize(raw),'pre_clamp_range':summarize(pre_n),'shifted_range':summarize(shn),'mean_gap_after_shift':float(np.mean(np.abs(raw.mean((0,2,3))-shn.mean((0,2,3))))),'std_gap_after_shift':float(np.mean(np.abs(raw.std((0,2,3))-shn.std((0,2,3)))))}
 result={'protocol':{'dataset':a.dataset,'method':a.method,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'optimizer':'AdamW','lr':a.lr,'checkpoint':'source-val only','clamp':not a.no_clamp,'target_gt_used_for_training_or_selection':False},'oa':float((pred==y).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(nclass))),'per_class_accuracy':pc.tolist(),'best_epoch':best['epoch'],'source_val_accuracy':best['val_acc'],'train_seconds':time.time()-t0}
 (a.output/'history.json').write_text(json.dumps(hist,indent=2)); (a.output/'result.json').write_text(json.dumps(result,indent=2)); (a.output/'dataset_stats.json').write_text(json.dumps(ds,indent=2)); (a.output/'scene_shift_stats.json').write_text(json.dumps(ssout,indent=2)); print(json.dumps(result,indent=2))

if __name__=='__main__': main()
