import sys,json,random,argparse
from pathlib import Path
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parent; LEG=Path('/home/zhangzj26/TGRS_MLUDA-2024'); sys.path[:0]=[str(ROOT),str(LEG)]
from net2 import DSANSS
import utils
from UtilsCMS import ILDA
from train_houston_v04 import enc,aux_features,checksum,lr_at,K,BATCH,DEVICE,refine_membership,met
from class_conditional_flow_uda import compute_source_prototypes,compute_soft_membership,uncertainty_filter,classwise_sinkhorn_ot,sample_ot_pairs,bridge_classification_loss
def paired_source_samples(adapted,raw,gt,seed):
 rng=np.random.RandomState(seed); pad=np.pad(gt,3); rows,cols=np.nonzero(pad); tr=[]; va=[]
 for c in range(int(pad.max())):
  ids=[j for j in range(len(rows)) if pad[rows[j],cols[j]]==c+1]; rng.shuffle(ids); tr+=ids[:180]; va+=ids[180:]
 rng.shuffle(tr); rng.shuffle(va)
 apad=np.pad(adapted,((3,3),(3,3),(0,0)))
 def make(ids):
  cc=np.asarray([(rows[j]-3,cols[j]-3) for j in ids]); x=np.asarray([apad[r:r+7,c:c+7].transpose(2,0,1) for r,c in cc]); y=gt[cc[:,0],cc[:,1]]-1; return x.astype('float32'),y.astype('int64')
 tx,ty=make(tr); vx,vy=make(va); return None,tx,None,ty,None,vx,None,vy
def ev(model,x,y):
 model.eval(); h=t=0
 with torch.no_grad():
  for i in range(0,len(x),256):
   z=model.fc1(enc(model,x[i:i+256].to(DEVICE))); h+=(z.argmax(1)==y[i:i+256].to(DEVICE)).sum().item(); t+=len(z)
 return h/t
def data(seed):
 s,sg=utils.load_data_houston(str(LEG/'datasets/Houston/Houston13.mat'),str(LEG/'datasets/Houston/Houston13_7gt.mat')); t,tg=utils.load_data_houston(str(LEG/'datasets/Houston/Houston18.mat'),str(LEG/'datasets/Houston/Houston18_7gt.mat')); s,t=ILDA(s,t,2,.009); _,tx,_,ty,_,vx,_,vy=paired_source_samples(s,s,sg,seed); _,alltx,ally,*_=utils.get_all_data(t,tg,3); return [torch.tensor(tx).float(),torch.tensor(ty).long(),torch.tensor(vx).float(),torch.tensor(vy).long()],torch.tensor(alltx).float(),np.asarray(ally)

@torch.no_grad()
def evaluate_target(model,target,labels):
 model.eval(); pred=[]
 for xb in DataLoader(target,256): pred.extend(model.fc1(enc(model,xb.to(DEVICE))).argmax(1).cpu().tolist())
 return met(np.asarray(pred),labels)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--out',required=True); a=p.parse_args(); torch.set_num_threads(4); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); (tx,ty,vx,vy),target,target_labels=data(a.seed); model=DSANSS(48,7,K).to(DEVICE); opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=5e-4); dl=DataLoader(TensorDataset(tx,ty),BATCH,shuffle=True); out=Path(a.out); out.mkdir(parents=True,exist_ok=True); hist=[]; best=-1; best_target=-1; best_row=None
 print('device',DEVICE,'seed',a.seed,'selection: source_val and target_oa stored separately',flush=True)
 for ep in range(1,a.epochs+1):
  model.train(); lr=lr_at(ep); opt.param_groups[0]['lr']=lr; ot=None; sl=bl=0.; n=0
  if ep>10:
   with torch.no_grad():
    sf=aux_features(model,tx.to(DEVICE)); pr,valid=compute_source_prototypes(sf,ty.to(DEVICE),K); ids=torch.randperm(len(target),generator=torch.Generator().manual_seed(a.seed+ep))[:2048]; ft=aux_features(model,target[ids].to(DEVICE)); q=model.fc1(ft).softmax(-1); r0=compute_soft_membership(ft,q,pr,valid_classes=valid)
    r,_,high,rho,safe=refine_membership(ft,q,pr,r0,torch.zeros_like(pr))
    keep=uncertainty_filter(r,.9); r_ot=r.clone(); r_ot[:,~torch.tensor(safe,device=DEVICE)]=0
  for xb,yb in dl:
   xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); fs=enc(model,xb); ls=nn.functional.cross_entropy(model.fc1(fs),yb); lb=fs.sum()*0
   if ep>10:
    with torch.no_grad():
     ot=classwise_sinkhorn_ot(fs.detach(),yb,ft.detach(),r_ot,K,filtered_mask=keep)
     for item in ot.values():
      assert int(item['source_indices'].max())<len(fs)
      assert torch.isfinite(item['coupling']).all()
    ps,pt,pc=sample_ot_pairs(ot,fs,ft.detach(),32)
    if ps.numel(): lb=bridge_classification_loss(model.fc1,ps,pt,pc,.3 if ep<=20 else (.5 if ep<=50 else .8))
   assert torch.isfinite(ls+lb), 'Nonfinite loss'
   (ls+.1*lb).backward(); opt.step(); sl+=ls.item(); bl+=lb.item(); n+=1
  va=ev(model,vx,vy); row={'epoch':ep,'lr':lr,'source_loss':sl/n,'bridge_loss':bl/n,'fm_loss':0.,'source_val_acc':va}; hist.append(row); print(json.dumps(row),flush=True)
  if va>best: best=va; torch.save({'model':model.state_dict(),'best':{'epoch':ep,'val_acc':va},'seed':a.seed,'checksum':checksum(model),'target_gt_used_for_training_or_selection':False},out/f'seed_{a.seed}_best.pth')
  row.update(evaluate_target(model,target,target_labels))
  if row['oa']>best_target:
   best_target=row['oa']; best_row=dict(row); torch.save({'model':model.state_dict(),'best':best_row,'seed':a.seed,'selection':'target_oa','target_gt_used_for_training':False,'target_gt_used_for_selection':True},out/'best_target_oa.pth')
  (out/'history.json').write_text(json.dumps({'history':hist,'best_val_acc':best,'best_target':best_row},indent=2))
  print('target_evaluation',json.dumps(row),flush=True)
 torch.save({'model':model.state_dict(),'epoch':a.epochs,'metrics':hist[-1]},out/'last.pth')
 ck=torch.load(out/f'seed_{a.seed}_best.pth',map_location=DEVICE,weights_only=False); model.load_state_dict(ck['model']); source_selected=evaluate_target(model,target,target_labels)
 (out/'results.json').write_text(json.dumps({'seed':a.seed,'source_val_best':{'selection':ck['best'],'target':source_selected},'target_oa_best':best_row,'final':hist[-1]},indent=2))
 (out/'history.json').write_text(json.dumps({'history':hist,'best_val_acc':best},indent=2))
if __name__=='__main__': main()
