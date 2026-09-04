import csv,json,sys
from pathlib import Path
import hdf5storage,numpy as np,torch
from sklearn import metrics
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(ROOT),str(HERE)]
from model import SemanticSceneUDA
def patches(c,cnt,w=7):
 h=w//2;p=np.pad(c,((h,h),(h,h),(0,0)));o=np.empty((len(cnt),c.shape[-1],w,w),np.float32)
 for i,(r,x) in enumerate(cnt):o[i]=p[r:r+w,x:x+w].transpose(2,0,1)
 return o
def met(y,p):
 cm=metrics.confusion_matrix(y,p,labels=np.arange(7));pc=np.diag(cm)/np.maximum(cm.sum(1),1);return {'oa':float((y==p).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,p,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'prediction_distribution':np.bincount(p,minlength=7).tolist(),'confusion_matrix':cm.tolist()}
def main():
 import argparse;ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=HERE/'runs');ap.add_argument('--device',default='cuda:0');a=ap.parse_args();d=torch.device(a.device)
 t=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data'];g=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map'];c=np.argwhere(g>0);x=patches(t,c);y=g[c[:,0],c[:,1]].astype(np.int64)-1;rows=[]
 for ck in sorted(a.root.glob('split_*/stage_*/best.pth')):
  z=torch.load(ck,map_location='cpu');m=SemanticSceneUDA().to(d);m.load_state_dict(z['model']);m.eval();p=[]
  with torch.no_grad():
   for i in range(0,len(x),32):p.append(m(torch.from_numpy(x[i:i+32]).to(d))[3].argmax(1).cpu().numpy())
  q=met(y,np.concatenate(p));rows.append({'split':ck.parent.parent.name,'stage':ck.parent.name,**q,'best_epoch':z['best']['epoch'],'source_val_accuracy':z['best']['val_acc'],'checkpoint':str(ck)})
 out={'protocol':{'target_gt':'post-hoc only','splits':[1174,1703,2141]},'runs':rows}
 (a.root/'summary.json').write_text(json.dumps(out,indent=2));
 with (a.root/'summary.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['split','stage','oa','aa','kappa','best_epoch','source_val_accuracy']);[w.writerow([r['split'],r['stage'],r['oa'],r['aa'],r['kappa'],r['best_epoch'],r['source_val_accuracy']]) for r in rows]
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
