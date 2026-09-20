"""Protocol-fixed Houston experiment (V0.3).

Auxiliary feature passes run in eval mode, prototypes are global per epoch, and
all modes resume from one shared source-only epoch-10 checkpoint.
"""
from __future__ import annotations
import hashlib, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
ROOT=Path(__file__).resolve().parent; LEGACY=Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path[:0]=[str(ROOT),str(LEGACY)]
import utils
from UtilsCMS import ILDA
from net2 import DSANSS
from class_conditional_flow_uda import compute_source_prototypes,compute_soft_membership,uncertainty_filter,classwise_sinkhorn_ot,sample_ot_pairs,ConditionalFlowMLP,flow_matching_loss,bridge_classification_loss
SEED, K, BATCH, CHUNK=1341,7,32,128; DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def seed(): random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED) if DEVICE.type=='cuda' else None
def enc(model,x): return model.feature_layers(x,x)[0]
def aux_features(model,x):
 was=model.training; model.eval(); out=[]
 with torch.no_grad():
  for i in range(0,len(x),CHUNK): out.append(enc(model,x[i:i+CHUNK]).detach())
 if was: model.train()
 return torch.cat(out)
def checksum(model): return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()[:16]
def met(pred,y):
 cm=np.zeros((K,K),int)
 for a,b in zip(y,pred): cm[a,b]+=1
 pc=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(K),where=cm.sum(1)>0); oa=float((pred==y).mean()); po=np.trace(cm)/len(y); pe=(cm.sum(0)*cm.sum(1)).sum()/len(y)**2
 return {'oa':oa,'aa':float(pc.mean()),'kappa':float((po-pe)/(1-pe+1e-12)),'per_class':pc.tolist()}
def load_data():
 s,sg=utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston13.mat'),str(LEGACY/'datasets/Houston/Houston13_7gt.mat')); t,tg=utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston18.mat'),str(LEGACY/'datasets/Houston/Houston18_7gt.mat')); s,t=ILDA(s,t,2,.009); sx,sy=utils.get_sample_data(s,sg,3,180); _,tx,ty,*_=utils.get_all_data(t,tg,3); return torch.tensor(sx).float(),torch.tensor(sy).long(),torch.tensor(tx).float(),torch.tensor(ty).long()
def lr_at(epoch): return .01/(1+10*(epoch-1)/100)**.75
def build(): return DSANSS(48,7,K).to(DEVICE),ConditionalFlowMLP(288,K).to(DEVICE)
def refine_membership(z,q,proto_s,r0,ema, tau_p=.95, m=.9, nmin=32, nsafe=4):
 pred=r0.argmax(1); tproto=ema.clone(); high=[]; rho=[]; valid=[]
 for c in range(K):
  mask=(pred==c)&(r0[:,c]>tau_p); n=int(mask.sum()); high.append(n); rho.append(min(1.,n/nmin)); valid.append(n>=nsafe)
  if n: tproto[c]=m*ema[c]+(1-m)*(r0[mask,c,None]*z[mask]).sum(0)/r0[mask,c].sum().clamp_min(1e-8)
 cos=torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(tproto,dim=1).t(); logits=torch.log(q.clamp_min(1e-8))+cos*0
 for c in range(K):
  if high[c]: logits[:,c]+=rho[c]*cos[:,c]
 r1=torch.softmax(logits,dim=1).detach(); return r1,tproto.detach(),high,rho,valid
def train(mode,epochs,out,warmup=None):
 seed(); sx,sy,tx,ty=load_data(); model,flow=build(); out.mkdir(parents=True,exist_ok=True)
 if warmup: model.load_state_dict(warmup['model']); flow.load_state_dict(warmup['flow']); print('warmup_checksum',checksum(model))
 opt=torch.optim.SGD(list(model.parameters())+([] if mode=='linear_bridge' else list(flow.parameters())),lr=.01,momentum=.9,weight_decay=5e-4)
 if warmup and 'optimizer' in warmup and mode=='flow_bridge': opt.load_state_dict(warmup['optimizer'])
 hist=[]; loader=DataLoader(TensorDataset(sx,sy),BATCH,shuffle=True); rng=torch.Generator().manual_seed(SEED+1000)
 start_epoch = 11 if warmup else 1
 for epoch in range(start_epoch,epochs+1):
  model.train(); flow.train(); sums=np.zeros(3); n=0; lr=lr_at(epoch); [g.update(lr=lr) for g in opt.param_groups]
  proto=valid=None; ema_t=torch.zeros(K,288,device=DEVICE); has_ema=torch.zeros(K,dtype=torch.bool,device=DEVICE); epoch_diag={'retained_ratio':0.0,'retained_membership_accuracy':0.0,'r0_target_accuracy':0.0,'r1_target_accuracy':0.0,'n_high':[0]*K,'rho':[0.0]*K,'target_proto_valid':[False]*K,'per_class_retained_target_count':[0]*K,'per_class_membership_mass':[0.0]*K,'per_class_ot_pair_count':[0]*K,'membership_max_mean':0.0,'membership_max_min':0.0,'membership_max_max':0.0,'membership_entropy':0.0,'tau_max':0.0}
  if mode!='source_only' and epoch>10:
   with torch.no_grad():
    sf=aux_features(model,sx.to(DEVICE)); proto,valid=compute_source_prototypes(sf,sy.to(DEVICE),K); print('global_proto_valid',valid.cpu().tolist(),'counts',torch.bincount(sy,minlength=K).tolist())
    ids=torch.randperm(len(tx),generator=rng)[:2048]; pool=tx[ids].to(DEVICE)
    with torch.no_grad(): ft=aux_features(model,pool); q=model.fc1(ft).softmax(-1); r0=compute_soft_membership(ft,q,proto,valid_classes=valid); r,tproto,high,rho,rel=refine_membership(ft,q,proto,r0,ema_t)
    rp=r.argmax(1); pool_y=ty[ids].to(DEVICE); epoch_diag.update({'retained_ratio':float(uncertainty_filter(r,.9).float().mean()),'retained_membership_accuracy':float((rp[uncertainty_filter(r,.9)]==pool_y[uncertainty_filter(r,.9)]).float().mean()),'r0_target_accuracy':float((r0.argmax(1)==pool_y).float().mean()),'r1_target_accuracy':float((rp==pool_y).float().mean()),'n_high':high,'rho':rho,'target_proto_valid':rel,'per_class_retained_target_count':torch.bincount(rp[uncertainty_filter(r,.9)],minlength=K).cpu().tolist(),'per_class_membership_mass':r.sum(0).cpu().tolist(),'membership_max_mean':float(r.max(1).values.mean()),'membership_max_min':float(r.max(1).values.min()),'membership_max_max':float(r.max(1).values.max()),'membership_entropy':float(-(r*r.clamp_min(1e-8).log()).sum(1).mean()),'tau_max':.3 if epoch<=20 else (.5 if epoch<=50 else .8)})
    print('target_ot_pool_checksum',hashlib.sha256(ids.numpy().tobytes()).hexdigest()[:12], 'retained',epoch_diag['retained_ratio'])
  for xb,yb in loader:
   xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); fs=enc(model,xb); lsrc=nn.functional.cross_entropy(model.fc1(fs),yb); lfm=fs.sum()*0; lbr=fs.sum()*0
   if mode!='source_only' and epoch>10:
    keep=uncertainty_filter(r,.9)
    # Reliability gate: classes with fewer than N_safe high-confidence targets
    # do not participate in OT/transport for this epoch.
    rho_t=torch.tensor(rho,device=DEVICE,dtype=r.dtype); r_ot=r*rho_t[None,:]
    r_ot[:,torch.tensor(high,device=DEVICE)<4]=0.0
    with torch.no_grad(): ot=classwise_sinkhorn_ot(fs.detach(),yb,ft.detach(),r_ot,K,filtered_mask=keep)
    ps,pt,pc=sample_ot_pairs(ot,fs,ft.detach(),32)
    epoch_diag['per_class_ot_pair_count']=[int(32 if c in ot else 0) for c in range(K)]
    if ps.numel():
     base_tau=.3 if epoch<=20 else (.5 if epoch<=50 else .8); nums=[]; fms=[]; brs=[]; ws=[]
     for c in torch.unique(pc).tolist():
      mask=pc==c; wc=float(rho[c]); tc=base_tau*wc
      if wc>0 and mask.any():
       brs.append(bridge_classification_loss(model.fc1,ps[mask],pt[mask],pc[mask],tc,flow_model=(flow if mode=='flow_bridge' else None))); ws.append(wc)
       if mode=='flow_bridge': fms.append(flow_matching_loss(flow,ps[mask].detach(),pt[mask].detach(),pc[mask],tc))
     if ws:
      w=torch.tensor(ws,device=ps.device); lbr=(torch.stack(brs)*w).sum()/w.sum(); lfm=(torch.stack(fms)*w).sum()/w.sum() if fms else lfm
   (lsrc+.1*lbr+.1*lfm).backward(); opt.step(); sums += [lsrc.item(),lfm.item(),lbr.item()]; n+=1
  model.eval(); pred=[]
  with torch.no_grad():
   for b in DataLoader(tx,BATCH*8): pred.extend(model.fc1(enc(model,b.to(DEVICE))).argmax(1).cpu().tolist())
  row={'epoch':epoch,'lr':lr,'source_loss':sums[0]/n,'fm_loss':sums[1]/n,'bridge_loss':sums[2]/n,**met(np.array(pred),ty.numpy()),**epoch_diag}; hist.append(row); print(json.dumps(row),flush=True)
  if epoch==10: torch.save({'model':model.state_dict(),'flow':flow.state_dict(),'optimizer':opt.state_dict(),'checksum':checksum(model)},out/'warmup_epoch10.pth')
 (out/'history.json').write_text(json.dumps(hist,indent=2)); return out/'warmup_epoch10.pth'
def main():
 import argparse; p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['warmup','source_only','linear_bridge','flow_bridge'],required=True); p.add_argument('--epochs',type=int,default=20); p.add_argument('--out',default=str(ROOT/'runs_v03')); a=p.parse_args(); base=Path(a.out)
 if a.mode=='warmup': train('source_only',10,base/'warmup'); return
 warm=torch.load(base/'warmup/warmup_epoch10.pth',map_location=DEVICE); train(a.mode,a.epochs,base/a.mode,warm)
if __name__=='__main__': main()
