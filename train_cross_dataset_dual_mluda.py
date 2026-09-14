"""Official-style MLUDA and Dual-SceneShift on Pavia/Shanghai.

The target cube is used as an unlabeled patch stream during training.  Target
labels are opened only after the source-validation checkpoint has been saved.
The dataset-specific ILDA preprocessing, model, optimizer and losses follow
the corresponding official MLUDA entry points.
"""
from __future__ import annotations
import argparse, csv, json, math, random, sys, time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

ROOT = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import mmd, utils
from net2 import DSANSS
from contrastive_loss import SupConLoss
from UtilsCMS import ILDA

CFG = {
    'pavia': dict(file_s='Pavia/paviaU.mat', file_t='Pavia/pavia.mat',
                 key_s='paviaU', key_t='pavia', gt_s='Pavia/paviaU_gt_7.mat',
                 gt_key_s='paviaU_gt_7', gt_t='Pavia/pavia_gt_7.mat',
                 gt_key_t='pavia_gt_7', pca=2, radius=0.00009,
                 patch=11, classes=7, lr=0.001, lmmd_scale=0.3),
    'shanghai_hangzhou': dict(file_s='Shanghai-Hangzhou/DataCube.mat',
                 file_t='Shanghai-Hangzhou/DataCube.mat', key_s='DataCube1',
                 key_t='DataCube2', gt_s='DataCube1', gt_key_s='gt1',
                 gt_t='Shanghai-Hangzhou/DataCube.mat', gt_key_t='gt2', pca=2, radius=0.00009,
                 patch=1, classes=3, lr=0.0003, lmmd_scale=0.01),
}

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def load_data(name):
    c = CFG[name]; d = ROOT/'datasets'
    z = sio.loadmat(d/c['file_s'])
    source = z[c['key_s']].astype(np.float32)
    if name == 'pavia':
        t = sio.loadmat(d/c['file_t']); target = t[c['key_t']].astype(np.float32)
        gs = sio.loadmat(d/c['gt_s']); source_gt = gs[c['gt_key_s']].astype(np.int64)
    else:
        target = z[c['key_t']].astype(np.float32)
        source_gt = z[c['gt_key_s']].astype(np.int64)
    # Official loaders independently standardize each scene before ILDA.
    source = ((source.reshape(-1, source.shape[-1]) - source.reshape(-1, source.shape[-1]).mean(0)) /
              (source.reshape(-1, source.shape[-1]).std(0) + 1e-12)).reshape(source.shape)
    target = ((target.reshape(-1, target.shape[-1]) - target.reshape(-1, target.shape[-1]).mean(0)) /
              (target.reshape(-1, target.shape[-1]).std(0) + 1e-12)).reshape(target.shape)
    # Official MLUDA ILDA; no target labels are needed here.
    source, target = ILDA(source, target, c['pca'], c['radius'])
    return source.astype(np.float32), target.astype(np.float32), source_gt, c

def load_target_gt(name):
    c=CFG[name]; d=ROOT/'datasets'
    z=sio.loadmat(d/c['gt_t'])
    return z[c['gt_key_t']].astype(np.int64)

def split_source(gt, classes, seed, per_class=180):
    rng=np.random.RandomState(seed); tr=[]; va=[]
    for k in range(1, classes+1):
        ij=np.argwhere(gt==k); rng.shuffle(ij); tr.append(ij[:per_class]); va.append(ij[per_class:])
    tr=np.concatenate(tr).astype(np.int64); va=np.concatenate(va).astype(np.int64)
    rng.shuffle(tr); rng.shuffle(va)
    return tr, gt[tr[:,0],tr[:,1]]-1, va, gt[va[:,0],va[:,1]]-1

def patches(cube, centers, width):
    h=width//2; pad=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant')
    # ``pad`` already shifts the original coordinates by ``h``; adding h a
    # second time produces an out-of-bounds index near the lower/right edge.
    r=centers[:,0,None]+np.arange(width)[None,:]
    q=centers[:,1,None]+np.arange(width)[None,:]
    return pad[r[:,:,None],q[:,None,:],:].transpose(0,3,1,2).astype(np.float32)

class Cube(Dataset):
    def __init__(self,cube,width):
        self.cube=cube; self.width=width; self.h,self.w,_=cube.shape; h=width//2
        self.pad=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant')
    def __len__(self): return self.h*self.w
    def __getitem__(self,i):
        r,c=divmod(int(i),self.w); h=self.width//2
        a=self.pad[r:r+self.width,c:c+self.width].transpose(2,0,1)
        return torch.from_numpy(np.ascontiguousarray(a))

def shift(x, sm, ss, tm, ts, alpha=.8):
    to=lambda a: torch.as_tensor(a,device=x.device,dtype=x.dtype)[None,:,None,None]
    y=(x-to(sm))/(to(ss)+1e-5); y=y*(alpha*to(ts)+(1-alpha)*to(ss))+alpha*to(tm)+(1-alpha)*to(sm)
    scale=1+.04*torch.randn(x.size(0),1,1,1,device=x.device)
    y=y*scale+.015*F.avg_pool2d(torch.randn_like(y),5,1,2)
    # Official cross-scene preprocessing is z-score + ILDA, so bounded
    # [0,1] clipping is not valid here.  Keep the same SceneShift formula but
    # return the transported standardized values without an extra clamp.
    return y

def aug(x):
    return utils.radiation_noise(x.cpu()).float().to(x.device), utils.flip_augmentation(x.cpu()).float().to(x.device)

def eval_source(model, loader, device):
    model.eval(); ce=nn.CrossEntropyLoss(); loss=correct=n=0
    with torch.no_grad():
        for x,y in loader:
            out=model(x.to(device),x.to(device))[3]; yy=y.to(device)
            loss += ce(out,yy).item()*len(y); correct += (out.argmax(1)==yy).sum().item(); n += len(y)
    return loss/n, correct/n

def eval_target(model, cube, gt, ref, width, device, classes):
    centers=np.argwhere(gt>0); labels=gt[centers[:,0],centers[:,1]]-1; pred=[]; model.eval()
    with torch.no_grad():
        for st in range(0,len(centers),512):
            x=torch.from_numpy(patches(cube,centers[st:st+512],width)).to(device)
            for j in range(0,len(x),32):
                xx=x[j:j+32]; rr=ref[:len(xx)].to(device)
                if len(rr)<len(xx): rr=ref.repeat((math.ceil(len(xx)/len(ref)),1,1,1))[:len(xx)].to(device)
                pred.append(model(rr,xx)[8].argmax(1).cpu().numpy())
    pred=np.concatenate(pred); cm=metrics.confusion_matrix(labels,pred,labels=np.arange(classes)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
    return dict(oa=float((pred==labels).mean()),aa=float(pc.mean()),kappa=float(metrics.cohen_kappa_score(labels,pred,labels=np.arange(classes))),per_class_accuracy=pc.tolist(),confusion_matrix=cm.tolist())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=CFG,required=True); p.add_argument('--method',choices=['original_mluda','dual_scene_shift'],required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--device',default='cuda:0'); p.add_argument('--alpha',type=float,default=.8); p.add_argument('--gamma',type=float,default=.5); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); seed_all(a.seed)
    c=CFG[a.dataset]; dev=torch.device(a.device if torch.cuda.is_available() else 'cpu'); source,target,gt,c=load_data(a.dataset); tr,ty,va,vy=split_source(gt,c['classes'],a.seed)
    tx=patches(source,tr,c['patch']); vx=patches(source,va,c['patch']);
    train=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty)),batch_size=a.batch_size,shuffle=True,drop_last=True)
    val=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy)),batch_size=a.batch_size,shuffle=False)
    target_ds=Cube(target,c['patch']); ns=len(train)*a.batch_size
    tloader=DataLoader(target_ds,batch_size=a.batch_size,sampler=RandomSampler(target_ds,replacement=False,num_samples=ns),drop_last=True)
    sm,ss=source.reshape(-1,source.shape[-1]).mean(0),source.reshape(-1,source.shape[-1]).std(0); tm,ts=target.reshape(-1,target.shape[-1]).mean(0),target.reshape(-1,target.shape[-1]).std(0)
    model=DSANSS(source.shape[-1],c['patch'],c['classes']).to(dev); ce=nn.CrossEntropyLoss(); cs=SupConLoss(temperature=.1).to(dev); ct=SupConLoss(temperature=.1).to(dev); best={'val_acc':-1}; hist=[]; t0=time.time()
    for ep in range(1,a.epochs+1):
        model.train()
        # The official entry points recreate this SGD object at each epoch;
        # retain that behavior for a protocol-matched comparison.
        opt=torch.optim.SGD([{'params':model.feature_layers.parameters()},{'params':model.fc1.parameters(),'lr':c['lr']},{'params':model.fc2.parameters(),'lr':c['lr']},{'params':model.head1.parameters(),'lr':c['lr']},{'params':model.head2.parameters(),'lr':c['lr']}],lr=c['lr'],momentum=.9,weight_decay=5e-4)
        it=iter(tloader); sums=correct=n=0
        for x,y in train:
            x,y=x.to(dev),y.to(dev)
            try: t=next(it).to(dev)
            except StopIteration: it=iter(tloader); t=next(it).to(dev)
            def adapt(counter):
                x0,x1=aug(x); t0,t1=aug(counter)
                sf,sf1,_,so,sd,tf,tf1,_,to_,td=model(x,counter)
                _,sx,_,_,_,_,_,txv,_,_=model(x0,t0)
                _,sy,_,_,_,_,_,tyv,_,_=model(x1,t1)
                pseudo=to_.softmax(1).detach().argmax(1)
                return c['lmmd_scale']*(2/(1+math.exp(-10*ep/a.epochs))-1)*mmd.lmmd(sf,tf,y,to_.softmax(1),BATCH_SIZE=a.batch_size,CLASS_NUM=c['classes'])+cs(torch.cat([sx.unsqueeze(1),sy.unsqueeze(1)],1),y)+ct(torch.cat([txv.unsqueeze(1),tyv.unsqueeze(1)],1),pseudo)
            out=model(x,t)[3]; loss=ce(out,y)
            if a.method=='original_mluda': loss=loss+adapt(t)
            else:
                loss=loss+adapt(t)
                loss=loss+a.gamma*adapt(shift(x,sm,ss,tm,ts,a.alpha))
            opt.zero_grad(); loss.backward(); opt.step(); sums+=float(loss.detach())*len(y); correct+=(out.argmax(1)==y).sum().item(); n+=len(y)
        vl,va=eval_source(model,val,dev); row={'epoch':ep,'train_loss':sums/n,'train_acc':correct/n,'val_loss':vl,'val_acc':va}; hist.append(row); print(json.dumps(row),flush=True)
        if va>best['val_acc']: best=row.copy(); torch.save({'model':model.state_dict(),'best':best},a.output/'best.pth')
    model.load_state_dict(torch.load(a.output/'best.pth',map_location='cpu')['model']); tg=load_target_gt(a.dataset); result=eval_target(model,target,tg,torch.from_numpy(tx[:a.batch_size]),c['patch'],dev,c['classes']); result.update({'dataset':a.dataset,'method':a.method,'seed':a.seed,'best_epoch':best['epoch'],'source_val_accuracy':best['val_acc'],'train_seconds':time.time()-t0,'alpha':a.alpha if a.method=='dual_scene_shift' else None,'gamma':a.gamma if a.method=='dual_scene_shift' else None,'target_gt_used_for_training_or_selection':False})
    (a.output/'history.json').write_text(json.dumps(hist,indent=2)); (a.output/'result.json').write_text(json.dumps(result,indent=2));
    with (a.output/'per_class.csv').open('w',newline='') as f: w=csv.writer(f); w.writerow(['class','accuracy']); w.writerows((i+1,v) for i,v in enumerate(result['per_class_accuracy']))
    config = dict(vars(a))
    config.update({'official_lr': c['lr'], 'patch': c['patch'],
                   'classes': c['classes'], 'bands': int(source.shape[-1]),
                   'preprocessing': 'official ILDA',
                   'checkpoint': 'source validation only'})
    (a.output/'config.json').write_text(json.dumps(config, indent=2, default=str)); print(json.dumps(result, indent=2), flush=True)

if __name__=='__main__': main()
