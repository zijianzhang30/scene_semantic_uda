"""Houston scene-shift / semantic-scene disentanglement UDA prototype.

This is intentionally independent of the legacy MLUDA/HyperSIGMA code. Target
GT is never loaded by training; it is only consumed by the separate evaluator.
"""
from __future__ import annotations
import argparse, json, math, random
from pathlib import Path
import hdf5storage, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(Path(__file__).resolve().parent))
from UtilsCMS import ILDA
import utils
from model import SemanticSceneUDA, orthogonality_loss
from config_Houston import HalfWidth

def center_patches(cube, centers, width):
    h=width//2; pad=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant'); out=np.empty((len(centers),cube.shape[-1],width,width),np.float32)
    for i,(r,c) in enumerate(centers): out[i]=pad[r:r+width,c:c+width].transpose(2,0,1)
    return out
def paired_source_samples(adapted, raw, gt, seed):
    rng=np.random.RandomState(seed); pgt=np.pad(gt,HalfWidth); rows,cols=np.nonzero(pgt); tr=[]; va=[]
    for cls in range(int(pgt.max())):
        inds=[j for j in range(len(rows)) if pgt[rows[j],cols[j]]==cls+1]; rng.shuffle(inds); tr+=inds[:180]; va+=inds[180:]
    rng.shuffle(tr); rng.shuffle(va); tc=np.asarray([(rows[j]-HalfWidth,cols[j]-HalfWidth) for j in tr],np.int64); vc=np.asarray([(rows[j]-HalfWidth,cols[j]-HalfWidth) for j in va],np.int64)
    return tc,center_patches(adapted,tc,7),center_patches(raw,tc,33),gt[tc[:,0],tc[:,1]].astype(np.int64)-1,vc,center_patches(adapted,vc,7),center_patches(raw,vc,33),gt[vc[:,0],vc[:,1]].astype(np.int64)-1

def set_seed(s):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class SpectralSpatialMambaLite(nn.Module):
 def __init__(self, bands=48, classes=7):
  super().__init__()
  self.spec=nn.Sequential(nn.Conv3d(1,32,(7,1,1),stride=(2,1,1)),nn.BatchNorm3d(32),nn.GELU(),nn.Conv3d(32,64,(5,1,1),stride=(2,1,1)),nn.BatchNorm3d(64),nn.GELU())
  self.spat=nn.Sequential(nn.Conv2d(bands,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU(),nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.GELU())
  self.fuse=nn.Sequential(nn.Conv2d(128,128,1),nn.BatchNorm2d(128),nn.GELU())
  self.sem=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.GELU())
  self.scene=nn.Sequential(nn.Linear(128,64),nn.LayerNorm(64),nn.GELU())
  self.cls=nn.Linear(128,classes)
 def forward(self,x):
  a=self.spec(x.unsqueeze(1)).mean(2); b=self.spat(x); h=self.fuse(torch.cat([a,b],1)).mean((2,3)); return self.sem(h),self.scene(h),self.cls(self.sem(h))

def shift_view(x, src_mean, src_std, tgt_mean, tgt_std, strength=.7):
 # label-preserving target-like spectral perturbation; smooth scaling and low
 # frequency noise are shared across the 7x7 patch.
 sm=torch.as_tensor(src_mean,device=x.device,dtype=x.dtype)[None,:,None,None]; ss=torch.as_tensor(src_std,device=x.device,dtype=x.dtype)[None,:,None,None]
 tm=torch.as_tensor(tgt_mean,device=x.device,dtype=x.dtype)[None,:,None,None]; ts=torch.as_tensor(tgt_std,device=x.device,dtype=x.dtype)[None,:,None,None]
 y=(x-sm)/(ss+1e-5)*(strength*ts+(1-strength)*ss)+strength*tm+(1-strength)*sm
 scale=1+0.04*torch.randn(x.size(0),1,1,1,device=x.device)
 lf=F.avg_pool2d(torch.randn_like(y),kernel_size=5,stride=1,padding=2)
 return (y*scale + 0.015*lf).clamp(0,1)

def aug(x):
 if torch.rand(())<.5: x=x.flip(-1)
 if torch.rand(())<.5: x=x.flip(-2)
 return x

def orth_loss(a,b): return orthogonality_loss(a,b)

def train_one(args):
 set_seed(args.optimization_seed); dev=torch.device(args.device)
 src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat'))
 # Training reads target imagery only.  The target GT file is intentionally
 # not opened here; it is consumed only by eval.py after training.
 tgt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']
 ds,dt=ILDA(src,tgt,2,0.009)
 trc, trx, _, try_, vac, vax, _, vay=paired_source_samples(ds,ds,sg,args.split_seed)
 # Source statistics and target statistics are unlabeled and computed from cubes.
 sm,ss=ds.reshape(-1,ds.shape[-1]).mean(0),ds.reshape(-1,ds.shape[-1]).std(0)
 tm,ts=dt.reshape(-1,dt.shape[-1]).mean(0),dt.reshape(-1,dt.shape[-1]).std(0)
 target_centers=np.argwhere(np.ones(dt.shape[:2],bool)).astype(np.int64)
 tx_all=center_patches(dt,target_centers,7)
 train_loader=DataLoader(TensorDataset(torch.from_numpy(trx),torch.from_numpy(try_)),batch_size=args.batch_size,shuffle=True,drop_last=True)
 val_loader=DataLoader(TensorDataset(torch.from_numpy(vax),torch.from_numpy(vay)),batch_size=args.batch_size)
 target_loader=DataLoader(torch.from_numpy(tx_all),batch_size=args.batch_size,shuffle=True,drop_last=True)
 model=SemanticSceneUDA().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss()
 hist=[]; best={'val_acc':-1}
 for ep in range(1,args.epochs+1):
  model.train(); it=iter(target_loader); sums={k:0. for k in ('total','cls','sem','orth','align')}; cor=n=0
  for xs,ys in train_loader:
   try: xt=next(it)
   except StopIteration: it=iter(target_loader); xt=next(it)
   xs,ys,xt=xs.to(dev),ys.to(dev),xt.to(dev); xs_shift=shift_view(xs,sm,ss,tm,ts)
   _,zs,ds_,logits=model(aug(xs)); _,zss,dss,logits_shift=model(aug(xs_shift)); _,zt,dt_,lt=model(aug(aug(xt))); _,ztw,_,ltw=model(xt)
   lcls=ce(logits,ys)
   if args.scene_shift: lcls=lcls+0.5*ce(logits_shift,ys)
   lsem=(1-F.cosine_similarity(zs,zss,dim=1)).mean() if args.semantic else torch.zeros((),device=dev)
   lorth=orth_loss(zs,ds_) if args.semantic else torch.zeros((),device=dev)
   lalign=torch.zeros((),device=dev)
   if args.alignment:
    probw=ltw.softmax(1); probs=lt.softmax(1); conf=probw.max(1).values.detach(); pseudo=probw.argmax(1).detach()
    consistency=(probw.argmax(1)==probs.argmax(1)).float().detach()
    prot=[]
    src_proto=torch.stack([zs[ys==k].mean(0) if (ys==k).any() else zs.mean(0) for k in range(7)])
    sim=F.cosine_similarity(ztw[:,None,:],src_proto[None,:,:],dim=2).max(1).values.detach().clamp_min(0)
    rel=(conf*consistency*sim).detach()
    for k in range(7):
     cs=zs[ys==k].mean(0) if (ys==k).any() else zs.mean(0)
     mask=(pseudo==k)&(conf>args.conf_threshold)&(consistency>0)&(sim>0)
     if mask.any():
      w=rel[mask]; ct=(zt[mask]*w[:,None]).sum(0)/(w.sum()+1e-6); prot.append((w.mean()*(cs-ct).pow(2).mean()))
     else: prot.append(torch.zeros((),device=dev))
    lalign=torch.stack(prot).mean()
   total=lcls + args.lambda_sem*lsem + args.lambda_orth*lorth + args.lambda_align*lalign
   opt.zero_grad(); total.backward(); opt.step(); bs=len(ys); n+=bs; cor+=(logits.argmax(1)==ys).sum().item()
   for k,v in (('total',total),('cls',lcls),('sem',lsem),('orth',lorth),('align',lalign)): sums[k]+=v.item()*bs
  model.eval(); vc=vl=vn=0
  with torch.no_grad():
   for x,y in val_loader:
    _,z,d_,o=model(x.to(dev)); y=y.to(dev); vl+=ce(o,y).item()*len(y); vc+=(o.argmax(1)==y).sum().item(); vn+=len(y)
  row={'epoch':ep,'train_loss':sums['total']/n,'loss_cls':sums['cls']/n,'loss_sem':sums['sem']/n,'loss_orth':sums['orth']/n,'loss_align':sums['align']/n,'train_acc':cor/n,'val_loss':vl/vn,'val_acc':vc/vn}; hist.append(row)
  print(json.dumps(row),flush=True)
  if row['val_acc']>best['val_acc']: best=row.copy(); torch.save({'model':model.state_dict(),'split_seed':args.split_seed,'optimization_seed':args.optimization_seed,'config':vars(args),'best':best,'target_gt_used_for_training_or_selection':False},args.output/'best.pth')
 cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}; (args.output/'history.json').write_text(json.dumps(hist,indent=2)); (args.output/'summary.json').write_text(json.dumps({'config':cfg,'best':best,'target_gt_used_for_training_or_selection':False},indent=2))

if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['baseline','scene_shift','scene_shift_sem','scene_shift_sem_orth','semantic','reliable_alignment','full'],default='baseline'); ap.add_argument('--split-seed',type=int,required=True); ap.add_argument('--optimization-seed',type=int,default=1174); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--lr',type=float,default=2e-3); ap.add_argument('--lambda-sem',type=float,default=.2); ap.add_argument('--lambda-orth',type=float,default=.01); ap.add_argument('--lambda-align',type=float,default=.1); ap.add_argument('--conf-threshold',type=float,default=.8); ap.add_argument('--device',default='cuda:0'); ap.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(); a.semantic=a.stage in ('scene_shift_sem','scene_shift_sem_orth','semantic','reliable_alignment','full'); a.scene_shift=a.stage in ('scene_shift','scene_shift_sem','scene_shift_sem_orth','semantic','reliable_alignment','full'); a.alignment=a.stage in ('reliable_alignment','full'); a.output.mkdir(parents=True,exist_ok=True); train_one(a)
