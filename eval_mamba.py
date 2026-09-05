"""Post-hoc Houston18 evaluation for clean Mamba checkpoints."""
import argparse,json
from pathlib import Path
import hdf5storage,numpy as np,torch
from sklearn import metrics
import sys, importlib.util
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(ROOT),str(HERE)]
from mamba_model import MambaBackboneClassifier
_spec=importlib.util.spec_from_file_location("hsi_utils", ROOT/'utils.py'); utils=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(utils)
SPLITS=(1174,1703,2141)
def patches(cube,centers,w=12):
 # Match DAMamba HyperX: symmetric pad width w//2+1 and even-patch slicing.
 p=np.pad(cube,((w//2+1,w//2+1),(w//2+1,w//2+1),(0,0)),mode='symmetric'); o=np.empty((len(centers),cube.shape[-1],w,w),np.float32)
 for i,(r,c) in enumerate(centers): o[i]=p[r+1:r+1+w,c+1:c+1+w].transpose(2,0,1)
 return o
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
 target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; centers=np.argwhere(gt>0).astype(np.int64); y=gt[centers[:,0],centers[:,1]].astype(np.int64)-1; runs=[]
 for s in SPLITS:
  ck=torch.load(a.root/f'split_{s}'/'best.pth',map_location='cpu'); m=MambaBackboneClassifier().to(a.device); m.load_state_dict(ck['model']); m.eval(); x=patches(target,centers); pred=[]
  with torch.no_grad():
   for i in range(0,len(x),32): pred.append(m(torch.from_numpy(x[i:i+32]).to(a.device)).argmax(1).cpu().numpy())
  pred=np.concatenate(pred); cm=metrics.confusion_matrix(y,pred,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1); runs.append({'split':s,'oa':float((y==pred).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'best_epoch':ck['best']['epoch'],'source_val_accuracy':ck['best']['val_acc']})
 out={'protocol':{'splits':list(SPLITS),'target_gt':'post-hoc only','model':'DAMamba MambaFeature -> official ChannelAttention/SpatialAttention -> pool -> Linear(4608,256) -> Linear(256,7)','patch_size':12},'runs':runs,'aggregate':{m:{'mean':float(np.mean([r[m] for r in runs])),'std':float(np.std([r[m] for r in runs]))} for m in ('oa','aa','kappa')}}; out['aggregate']['per_class_accuracy']={'mean':np.mean([r['per_class_accuracy'] for r in runs],0).tolist(),'std':np.std([r['per_class_accuracy'] for r in runs],0).tolist()}; (a.root/'summary.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
