"""Post-hoc target membership diagnostics; target GT is never used for training."""
import sys, json
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parent; LEG=Path('/home/zhangzj26/TGRS_MLUDA-2024'); sys.path[:0]=[str(ROOT),str(LEG)]
from net2 import DSANSS
import utils
from UtilsCMS import ILDA
from class_conditional_flow_uda import compute_source_prototypes,compute_soft_membership
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); K=7
def cm(y,p):
 a=np.zeros((K,K),int)
 for x,z in zip(y,p): a[int(x),int(z)]+=1
 return a.tolist()
def main():
 src,sg=utils.load_data_houston(str(LEG/'datasets/Houston/Houston13.mat'),str(LEG/'datasets/Houston/Houston13_7gt.mat')); tgt,tg=utils.load_data_houston(str(LEG/'datasets/Houston/Houston18.mat'),str(LEG/'datasets/Houston/Houston18_7gt.mat')); src,tgt=ILDA(src,tgt,2,.009); sx,sy=utils.get_sample_data(src,sg,3,180); _,tx,ty,*_=utils.get_all_data(tgt,tg,3); sx=torch.tensor(sx).float().to(DEVICE); sy=torch.tensor(sy).long().to(DEVICE); tx=torch.tensor(tx).float().to(DEVICE); ty=torch.tensor(ty).long().to(DEVICE)
 for mode in ('source_only','linear_bridge','flow_bridge'):
  m=DSANSS(48,7,K).to(DEVICE); m.load_state_dict(torch.load(ROOT/'runs_v01'/mode/'best_target_oa.pth',map_location=DEVICE)['model']); m.eval(); fs=[]; ft=[]
  with torch.no_grad():
   for b in DataLoader(sx,256): fs.append(m.feature_layers(b,b)[0]);
   for b in DataLoader(tx,256): ft.append(m.feature_layers(b,b)[0])
   fs=torch.cat(fs); ft=torch.cat(ft); q=m.fc1(ft).softmax(-1); proto,valid=compute_source_prototypes(fs,sy,K); r=compute_soft_membership(ft,q,proto,valid_classes=valid); pq=q.argmax(1); pr=r.argmax(1); y=ty.cpu().numpy();
  conf=r.max(1).values.cpu().numpy(); print('\nMODE',mode,'q_acc',float((pq.cpu().numpy()==y).mean()),'r_acc',float((pr.cpu().numpy()==y).mean())); print('q_cm',cm(y,pq.cpu().numpy())); print('r_cm',cm(y,pr.cpu().numpy())); print('per_q',[(y==c).sum() and float(((pq.cpu().numpy()==c)&(y==c)).sum()/(y==c).sum()) for c in range(K)]); print('per_r',[(y==c).sum() and float(((pr.cpu().numpy()==c)&(y==c)).sum()/(y==c).sum()) for c in range(K)]); print('mass',r.sum(0).cpu().tolist(),'gt',np.bincount(y,minlength=K).tolist())
  for t in (.45,.6,.7,.8,.9,.95):
   keep=conf>t; print('threshold',t,'ratio',float(keep.mean()),'acc',float((pr.cpu().numpy()[keep]==y[keep]).mean()) if keep.any() else 0,'pred_count',np.bincount(pr.cpu().numpy()[keep],minlength=K).tolist())
  print('bins',[(lo,hi,float(((pr.cpu().numpy()[z]==y[z]).mean()) if z.any() else 0),int(z.sum())) for lo,hi in zip((.5,.6,.7,.8,.9),( .6,.7,.8,.9,1.01)) for z in [((conf>=lo)&(conf<hi))]])
if __name__=='__main__': main()
