import argparse, json, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE))
from model import SemanticSceneUDA
import utils
def center_patches(cube, centers, width):
 h=width//2; pad=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant'); out=np.empty((len(centers),cube.shape[-1],width,width),np.float32)
 for i,(r,c) in enumerate(centers): out[i]=pad[r:r+width,c:c+width].transpose(2,0,1)
 return out
def metric(y,p):
 cm=metrics.confusion_matrix(y,p,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
 return {'oa':float((y==p).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,p,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'prediction_distribution':np.bincount(p,minlength=7).tolist(),'confusion_matrix':cm.tolist()}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',type=Path,required=True); ap.add_argument('--device',default='cuda:0'); a=ap.parse_args(); d=torch.device(a.device)
 tgt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; c=np.argwhere(gt>0).astype(np.int64); x=center_patches(tgt,c,7); y=gt[c[:,0],c[:,1]].astype(np.int64)-1
 m=SemanticSceneUDA().to(d); m.load_state_dict(torch.load(a.checkpoint,map_location='cpu')['model'],strict=True); m.eval(); p=[]
 with torch.no_grad():
  for i in range(0,len(x),32): p.append(m(torch.from_numpy(x[i:i+32]).to(d))[3].argmax(1).cpu().numpy())
 print(json.dumps(metric(y,np.concatenate(p)),indent=2))
if __name__=='__main__': main()
