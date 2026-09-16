import argparse, hashlib, json, math, random, time
from pathlib import Path
import hdf5storage, numpy as np, torch
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.insert(0, '/home/zhangzj26/TGRS_MLUDA-2024')
from UtilsCMS import ILDA
import utils
from models.sceneshift_net import SceneShiftNet

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); OUT=Path(__file__).resolve().parent/'runs_sceneshiftnet/houston'
def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def patches(cube, centers, w=7):
 h=w//2; p=np.pad(cube,((h,h),(h,h),(0,0))); return np.stack([p[r:r+w,c:c+w].transpose(2,0,1) for r,c in centers]).astype('float32')
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=1341); ap.add_argument('--epochs',type=int,default=100); ap.add_argument('--prepare-only',action='store_true'); a=ap.parse_args(); seed(a.seed)
 out=OUT/f'seed_{a.seed}'; out.mkdir(parents=True,exist_ok=True)
 src,sg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); tgt,tg=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston18.mat'),str(ROOT/'datasets/Houston/Houston18_7gt.mat')); src,tgt=ILDA(src,tgt,2,0.009)
 tr=[]; va=[]
 for k in range(1,8):
  ij=np.argwhere(sg==k); np.random.shuffle(ij); tr+=ij[:180].tolist(); va+=ij[180:].tolist()
 tr=np.array(tr); va=np.array(va); np.random.shuffle(tr); np.random.shuffle(va)
 tx=patches(src,tr); ty=(sg[tr[:,0],tr[:,1]]-1).astype('int64'); vx=patches(src,va); vy=(sg[va[:,0],va[:,1]]-1).astype('int64')
 flat_t=np.argwhere(tg>0); testx=patches(tgt,flat_t); testy=(tg[flat_t[:,0],flat_t[:,1]]-1).astype('int64')
 sm,ss=src.reshape(-1,48).mean(0),src.reshape(-1,48).std(0); tm,ts=tgt.reshape(-1,48).mean(0),tgt.reshape(-1,48).std(0)
 config={'seed':a.seed,'epochs':a.epochs,'source':'Houston13','target':'Houston18','classes':7,'source_per_class':180,'patch_size':7,'bands':48,'alpha':.8,'gamma':.5,'scene_shift_type':'pure_affine','random_scale':False,'smooth_noise':False,'lambda_shift':.5,'lambda_target':.5,'pseudo_threshold':.9,'pseudo_warmup_epochs':10,'preprocessing':'official load_data_houston + ILDA(pca=2,radius=.009)','optimizer':'Adam','lr':1e-3,'target_pool':'GT>0 support mask (labels not used in training)','evaluation':'final epoch target OA/AA/Kappa/per-class','model':'SceneShiftNet shared encoder 128-d + linear classifier','code_hash':sha(__file__),'model_hash':sha(Path(__file__).with_name('models')/'sceneshift_net.py')}
 (out/'config.json').write_text(json.dumps(config,indent=2));
 if a.prepare_only: print('PREPARED',out); return
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=SceneShiftNet().to(dev); opt=torch.optim.Adam(model.parameters(),lr=1e-3); dl=DataLoader(TensorDataset(torch.from_numpy(tx),torch.from_numpy(ty)),32,shuffle=True,drop_last=True); tdl=DataLoader(torch.from_numpy(patches(tgt,np.argwhere(tg>0))),32,shuffle=True,drop_last=True); ce=torch.nn.CrossEntropyLoss(); histlog=[]
 for ep in range(1,a.epochs+1):
  model.train(); it=iter(tdl); sums=[0.,0.,0.]; corr=[0,0]; cov=0; n=0; seen=0; hist=np.zeros(7,dtype=int)
  for x,y in dl:
   try: z=next(it)
   except StopIteration: it=iter(tdl); z=next(it)
   x,y,z=x.to(dev),y.to(dev),z.to(dev); shifted=(x-torch.tensor(sm,device=dev)[None,:,None,None])/(torch.tensor(ss,device=dev)[None,:,None,None]+1e-5)*(0.8*torch.tensor(ts,device=dev)[None,:,None,None]+0.2*torch.tensor(ss,device=dev)[None,:,None,None])+0.8*torch.tensor(tm,device=dev)[None,:,None,None]+0.2*torch.tensor(sm,device=dev)[None,:,None,None]
   _,os=model(x); _,oss=model(shifted); _,ot=model(z); conf,pseudo=ot.softmax(1).max(1); mask=(conf>.9) if ep>10 else torch.zeros_like(conf,dtype=torch.bool); lt=ce(ot[mask],pseudo[mask]) if mask.any() else ot.sum()*0; ls=ce(os,y); lss=ce(oss,y); loss=ls+.5*lss+(.5*lt if ep>10 else 0*lt); opt.zero_grad(); loss.backward(); opt.step(); bs=len(y); sums[0]+=ls.item()*bs; sums[1]+=lss.item()*bs; sums[2]+=lt.item()*bs; corr[0]+=(os.argmax(1)==y).sum().item(); corr[1]+=(oss.argmax(1)==y).sum().item(); cov+=int(mask.sum()); seen+=len(z); hist+=np.bincount(pseudo[mask].detach().cpu().numpy(),minlength=7); n+=bs
  row={'epoch':ep,'L_src':sums[0]/n,'L_shift':sums[1]/n,'L_target':sums[2]/n,'pseudo_label_coverage':cov/seen,'pseudo_label_class_histogram':hist.tolist(),'source_accuracy':corr[0]/n,'shifted_source_accuracy':corr[1]/n}; histlog.append(row); print(json.dumps(row),flush=True)
 torch.save({'model':model.state_dict(),'seed':a.seed},out/'final.pth'); model.eval(); pred=[]
 with torch.no_grad():
  for i in range(0,len(testx),32): pred.extend(model(torch.from_numpy(testx[i:i+32]).to(dev))[1].argmax(1).cpu().numpy())
 pred=np.array(pred); cm=metrics.confusion_matrix(testy,pred,labels=np.arange(7)); pc=np.diag(cm)/cm.sum(1); res={'oa':float((pred==testy).mean()*100),'aa':float(pc.mean()*100),'kappa':float(metrics.cohen_kappa_score(testy,pred)*100),'per_class_accuracy':(pc*100).tolist(),'seed':a.seed}; (out/'history.json').write_text(json.dumps(histlog,indent=2)); (out/'result.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
