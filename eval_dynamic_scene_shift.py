from __future__ import annotations
import argparse, json, csv
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
import train as clean
import utils
from model import DCRNClassifier

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent

def patches(cube, centers, width=7):
 h=width//2; p=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant'); out=np.empty((len(centers),cube.shape[-1],width,width),np.float32)
 for i,(r,c) in enumerate(centers): out[i]=p[r:r+width,c:c+width].transpose(2,0,1)
 return out
def logits(m,x,d,b=512):
 out=[]
 with torch.no_grad():
  for i in range(0,len(x),b): out.append(m(torch.from_numpy(x[i:i+b]).to(d)).cpu())
 return torch.cat(out)
def eval_ck(path, source, target, tgt_gt, device):
 ck=torch.load(path,map_location='cpu'); m=DCRNClassifier().to(device); m.load_state_dict(ck['model']); m.eval()
 centers=np.argwhere(tgt_gt>0); x=patches(target,centers); y=tgt_gt[centers[:,0],centers[:,1]].astype(np.int64)-1
 pred=logits(m,x,device).argmax(1).numpy(); cm=metrics.confusion_matrix(y,pred,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
 return {'oa':float((pred==y).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'best_epoch':ck['best']['epoch'],'source_val_accuracy':ck['best']['val_acc']}
def main():
 p=argparse.ArgumentParser(); p.add_argument('--device',default='cuda:0'); p.add_argument('--output',type=Path,default=HERE/'runs_dynamic_scene_shift_1174'); a=p.parse_args(); d=torch.device(a.device)
 source,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; tg=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; source=source.astype(np.float32); target=target.astype(np.float32)
 paths={'CE':HERE/'runs_scene_shift_strength_sweep_1174/alpha_0_0/best.pth','Fixed alpha=0.8':HERE/'runs_scene_shift_strength_sweep_1174/alpha_0_8/best.pth','Random U(0,1)':a.output/'random/best.pth','Progressive 0->0.8':a.output/'progressive08/best.pth','Progressive 0->1.0':a.output/'progressive10/best.pth'}
 res={k:eval_ck(v,source,target,tg,d) for k,v in paths.items()}
 a.output.mkdir(exist_ok=True,parents=True); (a.output/'results.json').write_text(json.dumps(res,indent=2));
 with (a.output/'per_class.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['method','class','accuracy']);
  for n,r in res.items():
   for i,v in enumerate(r['per_class_accuracy'],1): w.writerow([n,i,v])
 print(json.dumps(res,indent=2))
if __name__=='__main__': main()
