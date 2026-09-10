from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np, scipy.io as sio, hdf5storage, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import scale
from sklearn import metrics
import train as clean
import utils
from ssftt_wrapper import SSFTTBandAligned

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent
CFG={'houston':{'s':'Houston/Houston13.mat','sg':'Houston/Houston13_7gt.mat','t':'Houston/Houston18.mat','tg':'Houston/Houston18_7gt.mat','sk':'ori_data','tk':'ori_data','sgk':'map','tgk':'map','patch':7,'classes':7,'norm':'native01'},'pavia':{'s':'Pavia/paviaU.mat','sg':'Pavia/paviaU_gt_7.mat','t':'Pavia/pavia.mat','tg':'Pavia/pavia_gt_7.mat','sk':'paviaU','tk':'pavia','sgk':'paviaU_gt_7','tgk':'pavia_gt_7','patch':11,'classes':7,'norm':'zscore'},'shanghai_hangzhou':{'file':'Shanghai-Hangzhou/DataCube.mat','patch':1,'classes':3,'norm':'zscore'}}
def seed(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def norm(x): return x if False else scale(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype(np.float32)
def load(name):
 c=CFG[name]; dr=ROOT/'datasets'
 if name=='shanghai_hangzhou':
  z=sio.loadmat(dr/c['file']); return norm(z['DataCube1'].astype(np.float32)),norm(z['DataCube2'].astype(np.float32)),z['gt1'].astype(np.int64),z['gt2'].astype(np.int64),c
 if name=='houston':
  s=hdf5storage.loadmat(str(dr/c['s']))[c['sk']].astype(np.float32); t=hdf5storage.loadmat(str(dr/c['t']))[c['tk']].astype(np.float32); sg=hdf5storage.loadmat(str(dr/c['sg']))[c['sgk']].astype(np.int64); tg=hdf5storage.loadmat(str(dr/c['tg']))[c['tgk']].astype(np.int64)
 else:
  s=sio.loadmat(dr/c['s'])[c['sk']].astype(np.float32);t=sio.loadmat(dr/c['t'])[c['tk']].astype(np.float32);sg=sio.loadmat(dr/c['sg'])[c['sgk']].astype(np.int64);tg=sio.loadmat(dr/c['tg'])[c['tgk']].astype(np.int64)
 if name!='houston':s,t=norm(s),norm(t)
 return s,t,sg,tg,c
def patches(c,cent,w):
 h=w//2; p=np.pad(c,((h,h),(h,h),(0,0)),mode='constant'); out=np.empty((len(cent),c.shape[-1],w,w),np.float32)
 for i,(r,col) in enumerate(cent):out[i]=p[r:r+w,col:col+w].transpose(2,0,1)
 return out
def split(gt,n,sd):
 rng=np.random.RandomState(sd);tr=[];va=[]
 for c in range(1,n+1):
  ij=np.argwhere(gt==c);rng.shuffle(ij);tr.append(ij[:180]);va.append(ij[180:])
 tr=np.concatenate(tr);va=np.concatenate(va);rng.shuffle(tr);rng.shuffle(va);return tr,gt[tr[:,0],tr[:,1]]-1,va,gt[va[:,0],va[:,1]]-1
def shift(x,sm,ss,tm,ts,a,clamp):
 sm=torch.as_tensor(sm,device=x.device,dtype=x.dtype)[None,:,None,None];ss=torch.as_tensor(ss,device=x.device,dtype=x.dtype)[None,:,None,None];tm=torch.as_tensor(tm,device=x.device,dtype=x.dtype)[None,:,None,None];ts=torch.as_tensor(ts,device=x.device,dtype=x.dtype)[None,:,None,None];y=(x-sm)/(ss+1e-5);y=y*(a*ts+(1-a)*ss)+a*tm+(1-a)*sm; y=y*(1+.04*torch.randn(x.size(0),1,1,1,device=x.device))+.015*F.avg_pool2d(torch.randn_like(y),5,1,2); return y.clamp(0,1) if clamp else y
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=CFG,required=True);p.add_argument('--method',choices=['ce','scene_shift'],required=True);p.add_argument('--seed',type=int,default=1174);p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--lr',type=float,default=.002);p.add_argument('--device',default='cuda:0');p.add_argument('--scene_shift',action='store_true');p.add_argument('--scene_shift_alpha',type=float,default=.8);p.add_argument('--scene_shift_weight',type=float,default=.5);p.add_argument('--scene_shift_clamp',choices=['auto','true','false'],default='auto');p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);seed(a.seed);dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');s,t,sg,tg,c=load(a.dataset);w=c['patch'];tr,ty,va,vy=split(sg,c['classes'],a.seed);tx=patches(s,tr,w);vx=patches(s,va,w);sf=s.reshape(-1,s.shape[-1]);tf=t.reshape(-1,t.shape[-1]);sm,ss,tm,ts=sf.mean(0),sf.std(0),tf.mean(0),tf.std(0); clamp=(c['norm']=='native01') if a.scene_shift_clamp=='auto' else a.scene_shift_clamp=='true';
 dl=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty)),batch_size=a.batch_size,shuffle=True,drop_last=True);vl=DataLoader(TensorDataset(torch.from_numpy(vx),torch.from_numpy(vy)),batch_size=a.batch_size);m=SSFTTBandAligned(s.shape[-1],c['classes'],w).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=1e-4);ce=nn.CrossEntropyLoss();best={'val_acc':-1};hist=[]
 for ep in range(1,a.epochs+1):
  m.train();ls=co=seen=0
  for x,y in dl:
   x,y=x.to(dev),y.to(dev);loss=ce(m(clean.augment(x)),y)
   if a.method=='scene_shift': sh=shift(x,sm,ss,tm,ts,a.scene_shift_alpha,clamp);loss=loss+a.scene_shift_weight*ce(m(clean.augment(sh)),y)
   opt.zero_grad();loss.backward();opt.step();ls+=loss.item()*len(y);co+=(m(clean.augment(x)).argmax(1)==y).sum().item();seen+=len(y)
  m.eval();vloss=vc=vn=0
  with torch.no_grad():
   for x,y in vl:z=m(x.to(dev));yy=y.to(dev);vloss+=ce(z,yy).item()*len(y);vc+=(z.argmax(1)==yy).sum().item();vn+=len(y)
  row={'epoch':ep,'train_loss':ls/seen,'train_acc':co/seen,'val_loss':vloss/vn,'val_acc':vc/vn};hist.append(row);print(json.dumps(row),flush=True)
  if row['val_acc']>best['val_acc']:best=row.copy();torch.save({'model':m.state_dict(),'best':best,'dataset':a.dataset,'method':a.method,'bands':s.shape[-1],'patch':w,'classes':c['classes'],'clamp':clamp,'target_gt_used_for_training_or_selection':False},a.output/'best.pth')
 ck=torch.load(a.output/'best.pth',map_location='cpu');m.load_state_dict(ck['model']);m.eval();cent=np.argwhere(tg>0);y=tg[cent[:,0],cent[:,1]]-1;pred=[]
 with torch.no_grad():
  for i in range(0,len(cent),512):pred.append(m(torch.from_numpy(patches(t,cent[i:i+512],w)).to(dev)).argmax(1).cpu().numpy())
 pred=np.concatenate(pred);cm=metrics.confusion_matrix(y,pred,labels=np.arange(c['classes']));pc=np.diag(cm)/np.maximum(cm.sum(1),1);res={'protocol':{'dataset':a.dataset,'method':a.method,'seed':a.seed,'epochs':a.epochs,'batch_size':a.batch_size,'optimizer':'AdamW','lr':a.lr,'scene_shift_alpha':a.scene_shift_alpha,'scene_shift_clamp':clamp,'target_gt_used_for_training_or_selection':False},'oa':float((pred==y).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(c['classes']))),'per_class_accuracy':pc.tolist(),'best_epoch':best['epoch'],'source_val_accuracy':best['val_acc']};
 cfg=vars(a).copy();cfg['output']=str(a.output);cfg.update({'clamp':clamp,'normalization':c['norm']}); (a.output/'config.json').write_text(json.dumps(cfg,indent=2));(a.output/'history.json').write_text(json.dumps(hist,indent=2));(a.output/'result.json').write_text(json.dumps(res,indent=2));(a.output/'per_class.csv').write_text('class,accuracy\n'+'\n'.join(f'{i+1},{v}' for i,v in enumerate(pc))); raw=tx[:min(512,len(tx))];torch.manual_seed(a.seed+777);pr=shift(torch.from_numpy(raw),sm,ss,tm,ts,.8,False).numpy();sh=shift(torch.from_numpy(raw),sm,ss,tm,ts,.8,clamp).numpy();(a.output/'scene_shift_stats.json').write_text(json.dumps({'alpha':.8,'clamp':clamp,'pre_clamp_lt0_ratio':float((pr<0).mean()),'pre_clamp_gt1_ratio':float((pr>1).mean()),'output_range':{'min':float(sh.min()),'max':float(sh.max()),'mean':float(sh.mean()),'std':float(sh.std())}},indent=2));print(json.dumps(res,indent=2))
if __name__=='__main__':main()
