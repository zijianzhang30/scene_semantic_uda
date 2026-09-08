"""Post-hoc Houston18 evaluation for clean Mamba checkpoints."""
import argparse,json,shutil,time
from pathlib import Path
import hdf5storage,numpy as np,torch
from sklearn import metrics
import sys, importlib.util
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(ROOT),str(HERE)]
from mamba_model import MambaBackboneClassifier
from own_backbone import (SpectralSpatialGatedMambaClassifier,
                          SpectralPillarsClassifier,
                          DistributionalSpectralPillarsClassifier,
                          JointSpectralSpatialMambaClassifier,
                          SceneRobustJointSpectralSpatialMambaClassifier)
_spec=importlib.util.spec_from_file_location("hsi_utils", ROOT/'utils.py'); utils=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(utils)
SPLITS=(1174,1703,2141)
def patches(cube,centers,w=12):
 # Match DAMamba HyperX: symmetric pad width w//2+1 and even-patch slicing.
 p=np.pad(cube,((w//2+1,w//2+1),(w//2+1,w//2+1),(0,0)),mode='symmetric'); o=np.empty((len(centers),cube.shape[-1],w,w),np.float32)
 for i,(r,c) in enumerate(centers): o[i]=p[r+1:r+1+w,c+1:c+1+w].transpose(2,0,1)
 return o
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--device',default='cuda:0'); p.add_argument('--checkpoint-root',type=Path,default=None); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--splits',type=int,nargs='+',default=list(SPLITS),choices=SPLITS); a=p.parse_args()
 target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; centers=np.argwhere(gt>0).astype(np.int64); y=gt[centers[:,0],centers[:,1]].astype(np.int64)-1; runs=[]
 for s in a.splits:
  checkpoint_root = a.checkpoint_root or a.root
  ck=torch.load(checkpoint_root/f'split_{s}'/'best.pth',map_location='cpu'); m=({'own': SpectralSpatialGatedMambaClassifier, 'pillars': SpectralPillarsClassifier, 'distributional_pillars': DistributionalSpectralPillarsClassifier, 'joint_mamba': JointSpectralSpatialMambaClassifier, 'joint_mamba_medium': SceneRobustJointSpectralSpatialMambaClassifier}.get(ck.get('model_type'), MambaBackboneClassifier)()).to(a.device); m.load_state_dict(ck['model']); m.eval(); x=patches(target,centers); pred=[]
  with torch.no_grad():
   warm=torch.from_numpy(x[:min(len(x),a.batch_size)]).to(a.device)
   for _ in range(2): m(warm)
   if str(a.device).startswith('cuda'): torch.cuda.synchronize(a.device)
   t0=time.perf_counter()
  with torch.no_grad():
   for i in range(0,len(x),a.batch_size): pred.append(m(torch.from_numpy(x[i:i+a.batch_size]).to(a.device)).argmax(1).cpu().numpy())
  if str(a.device).startswith('cuda'): torch.cuda.synchronize(a.device)
  inference_seconds=time.perf_counter()-t0
  pred=np.concatenate(pred); cm=metrics.confusion_matrix(y,pred,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1); runs.append({'split':s,'model_type':ck.get('model_type','mamba'),'oa':float((y==pred).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'best_epoch':ck['best']['epoch'],'source_val_accuracy':ck['best']['val_acc'],'params':sum(p.numel() for p in m.parameters()),'inference_seconds':inference_seconds,'inference_ms_per_patch':1000*inference_seconds/len(x)})
 model_names={'own':'SpectralSpatialGatedMamba','pillars':'SpectralPillars','distributional_pillars':'DistributionalSpectralPillars(K=8)','joint_mamba':'JointSpectralSpatialMamba','joint_mamba_medium':'SceneRobustJointSpectralSpatialMamba'}; out={'protocol':{'splits':list(a.splits),'target_gt':'post-hoc only','model':model_names.get(runs[0].get('model_type') if runs else '', 'DAMamba MambaFeature -> official ChannelAttention/SpatialAttention -> pool -> Linear(4608,256) -> Linear(256,7)'),'patch_size':12,'flops':'not reported: Mamba selective-scan FLOPs are not covered by installed profilers'},'runs':runs,'aggregate':{m:{'mean':float(np.mean([r[m] for r in runs])),'std':float(np.std([r[m] for r in runs]))} for m in ('oa','aa','kappa')}}; out['aggregate']['per_class_accuracy']={'mean':np.mean([r['per_class_accuracy'] for r in runs],0).tolist(),'std':np.std([r['per_class_accuracy'] for r in runs],0).tolist()}; out['aggregate']['inference_ms_per_patch']={'mean':float(np.mean([r['inference_ms_per_patch'] for r in runs])),'std':float(np.std([r['inference_ms_per_patch'] for r in runs]))}; (a.root/'summary.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
