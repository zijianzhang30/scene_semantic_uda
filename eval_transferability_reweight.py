"""Post-hoc target evaluation and comparison for transferability reweighting."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent; sys.path[:0]=[str(ROOT),str(HERE)]
from model import DCRNClassifier

def patches(cube, centers, width=7):
    h=width//2; p=np.pad(cube,((h,h),(h,h),(0,0)),mode='constant'); out=np.empty((len(centers),cube.shape[-1],width,width),np.float32)
    for i,(r,c) in enumerate(centers): out[i]=p[r:r+width,c:c+width].transpose(2,0,1)
    return out

def evaluate(ck_path, x, y, device):
    ck=torch.load(ck_path,map_location='cpu'); m=DCRNClassifier().to(device); m.load_state_dict(ck['model'],strict=True); m.eval(); pred=[]
    with torch.no_grad():
        for i in range(0,len(x),32): pred.append(m(torch.from_numpy(x[i:i+32]).to(device)).argmax(1).cpu().numpy())
    pred=np.concatenate(pred); cm=metrics.confusion_matrix(y,pred,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
    return {'oa':float((y==pred).mean()),'aa':float(pc.mean()),'kappa':float(metrics.cohen_kappa_score(y,pred,labels=np.arange(7))),'per_class_accuracy':pc.tolist(),'best_epoch':ck['best']['epoch'],'source_val_accuracy':ck['best']['val_acc']}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=HERE/'runs_transferability_reweight_1174'); p.add_argument('--sweep-root',type=Path,default=HERE/'runs_scene_shift_strength_sweep_1174'); p.add_argument('--device',default='cuda:0'); args=p.parse_args()
    target=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']; gt=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18_7gt.mat'))['map']; centers=np.argwhere(gt>0).astype(np.int64); y=gt[centers[:,0],centers[:,1]].astype(np.int64)-1; x=patches(target,centers)
    models={'DCRN + CE (alpha=0)':args.sweep_root/'alpha_0_0'/'best.pth','Global Scene Shift alpha=0.8':args.sweep_root/'alpha_0_8'/'best.pth','Transferability reweight':args.root/'best.pth'}; results={k:evaluate(v,x,y,args.device) for k,v in models.items()}
    rows=[]
    for name,r in results.items():
        for c,a in enumerate(r['per_class_accuracy'],1): rows.append({'model':name,'class':c,'accuracy':a})
    with (args.root/'per_class_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['model','class','accuracy']); w.writeheader(); w.writerows(rows)
    out={'protocol':{'split':1174,'target_gt':'post-hoc evaluation only','alpha':0.8,'raw_loss_weight':1.0,'shifted_loss':'class-wise lambda','target_gt_used_for_training_or_selection':False},'results':results,'transferability_weights':json.loads((args.root/'transferability_weights.json').read_text())}
    (args.root/'result.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
