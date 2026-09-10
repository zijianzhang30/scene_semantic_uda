"""Clean DCRN dynamic Scene Shift experiment for split 1174.

Only the strength schedule differs from the existing clean global Scene Shift:
fixed alpha, per-sample U(0,1), or progressive 0->alpha_max.  No target labels
are loaded and checkpoint selection is source-validation only.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import hdf5storage, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import train as clean
import utils
from model import DCRNClassifier

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def shift_batch(x, sm, ss, tm, ts, alpha):
    sm=torch.as_tensor(sm,device=x.device,dtype=x.dtype)[None,:,None,None]
    ss=torch.as_tensor(ss,device=x.device,dtype=x.dtype)[None,:,None,None]
    tm=torch.as_tensor(tm,device=x.device,dtype=x.dtype)[None,:,None,None]
    ts=torch.as_tensor(ts,device=x.device,dtype=x.dtype)[None,:,None,None]
    if not torch.is_tensor(alpha): alpha=torch.tensor(alpha,device=x.device,dtype=x.dtype)
    alpha=alpha.to(device=x.device,dtype=x.dtype)
    if alpha.ndim==0: alpha=alpha.view(1,1,1,1)
    elif alpha.ndim==1: alpha=alpha.view(-1,1,1,1)
    shifted=(x-sm)/(ss+1e-5)
    shifted=shifted*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
    scale=1.0+0.04*torch.randn(x.size(0),1,1,1,device=x.device)
    noise=F.avg_pool2d(torch.randn_like(shifted),kernel_size=5,stride=1,padding=2)
    return (shifted*scale+0.015*noise).clamp(0,1)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['ce','fixed','random','progressive08','progressive10'],required=True)
    p.add_argument('--split-seed',type=int,default=1174); p.add_argument('--optimization-seed',type=int,default=1174)
    p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=.002)
    p.add_argument('--device',default='cuda:0'); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    set_seed(a.optimization_seed); dev=torch.device(a.device)
    source,gt=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat'))
    target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; source=source.astype(np.float32); target=target.astype(np.float32)
    tc,ty,vc,vy=clean.source_split(gt,a.split_seed); tx=clean.center_patches(source,tc); vx=clean.center_patches(source,vc)
    sf=source.reshape(-1,source.shape[-1]); tf=target.reshape(-1,target.shape[-1]); sm,ss,tm,ts=sf.mean(0),sf.std(0),tf.mean(0),tf.std(0)
    tl=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty)),batch_size=a.batch_size,shuffle=True,drop_last=True)
    vl=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy)),batch_size=a.batch_size,shuffle=False)
    model=DCRNClassifier().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); best={'val_acc':-1.0}; hist=[]
    for epoch in range(1,a.epochs+1):
        model.train(); ls=correct=seen=0; alphas=[]
        for x,y in tl:
            x,y=x.to(dev),y.to(dev); raw=model(clean.augment(x)); loss=ce(raw,y)
            if a.mode!='ce':
                if a.mode=='fixed': alpha=torch.tensor(.8,device=dev)
                elif a.mode=='random': alpha=torch.rand(len(y),device=dev); alphas.extend(alpha.detach().cpu().tolist())
                elif a.mode=='progressive08': alpha=.8*epoch/a.epochs
                else: alpha=1.0*epoch/a.epochs
                if a.mode!='random': alphas.append(float(alpha) if not torch.is_tensor(alpha) else float(alpha.item()))
                shifted=shift_batch(x,sm,ss,tm,ts,alpha); loss=loss+.5*ce(model(clean.augment(shifted)),y)
            opt.zero_grad(); loss.backward(); opt.step(); ls+=float(loss.detach())*len(y); correct+=(raw.argmax(1)==y).sum().item(); seen+=len(y)
        model.eval(); vlss=vcorr=vseen=0
        with torch.no_grad():
            for x,y in vl:
                z=model(x.to(dev)); yy=y.to(dev); vlss+=ce(z,yy).item()*len(y); vcorr+=(z.argmax(1)==yy).sum().item(); vseen+=len(y)
        row={'epoch':epoch,'train_loss':ls/seen,'train_acc':correct/seen,'val_loss':vlss/vseen,'val_acc':vcorr/vseen}
        if alphas: row['alpha_sampled']={'mean':float(np.mean(alphas)),'std':float(np.std(alphas)),'min':float(np.min(alphas)),'max':float(np.max(alphas))}
        hist.append(row); print(json.dumps(row),flush=True)
        if row['val_acc']>best['val_acc']:
            best=row.copy(); torch.save({'model':model.state_dict(),'mode':a.mode,'split_seed':a.split_seed,'optimization_seed':a.optimization_seed,'scene_shift_alpha_max':(.8 if a.mode=='progressive08' else 1.0 if a.mode=='progressive10' else .8 if a.mode=='fixed' else None),'target_gt_used_for_training_or_selection':False,'best':best},a.output/'best.pth')
    cfg = vars(a).copy(); cfg['output'] = str(a.output)
    (a.output/'history.json').write_text(json.dumps(hist,indent=2)); (a.output/'summary.json').write_text(json.dumps({'config':cfg,'best':best,'target_gt_used_for_training_or_selection':False},indent=2))

if __name__=='__main__': main()
