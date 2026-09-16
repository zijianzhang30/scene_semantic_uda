import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); from UtilsCMS import ILDA; import utils
from models.sceneshift_net_dcrn import SceneShiftNetDCRN

def mean_dist(a,b): return float(np.linalg.norm(a.mean(0)-b.mean(0)))
def coral(a,b):
 ca=np.cov(a,rowvar=False); cb=np.cov(b,rowvar=False)
 return float(np.linalg.norm(ca-cb,'fro')/(4*a.shape[1]**2))
def mmd(a,b):
 x=torch.from_numpy(a).float(); y=torch.from_numpy(b).float(); z=torch.cat([x,y]);
 with torch.no_grad():
  d=torch.cdist(z,z).square(); med=torch.median(d[d>0]); vals=[]
  for q in (.25,.5,1,2,4):
   k=torch.exp(-d/(2*med*q+1e-12)); n=len(x); vals.append(k[:n,:n].mean()+k[n:,n:].mean()-2*k[:n,n:].mean())
 return float(torch.stack(vals).mean())
def spectral_angle(a,b):
 ma=a.mean(0); mb=b.mean(0); return float(np.arccos(np.clip(ma@mb/(np.linalg.norm(ma)*np.linalg.norm(mb)+1e-12),-1,1)))
def metrics(a,b): return {'mean_distance':mean_dist(a,b),'coral_distance':coral(a,b),'mmd_rbf':mmd(a,b),'mean_sam_radians':spectral_angle(a,b)}
def plot_tsne(groups,path,title):
 x=np.concatenate(groups); y=np.repeat(np.arange(3),[len(g) for g in groups]); emb=TSNE(n_components=2,perplexity=30,init='pca',learning_rate='auto',random_state=1341).fit_transform(x)
 for i,n in enumerate(('Source','Shifted Source','Target')): plt.scatter(emb[y==i,0],emb[y==i,1],s=5,alpha=.5,label=n)
 plt.legend(); plt.title(title); plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--samples',type=int,default=3000); ap.add_argument('--tsne-samples',type=int,default=800); ap.add_argument('--batch-size',type=int,default=128); a=ap.parse_args(); rng=np.random.RandomState(1341)
 out=HERE/'runs_sceneshiftnet_dcrn/houston/seed_1341/domain_gap_analysis'; out.mkdir(parents=True,exist_ok=True)
 s,_=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat')); t,_=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston18.mat'),str(ROOT/'datasets/Houston/Houston18_7gt.mat')); s,t=ILDA(s,t,2,.009)
 sm,ss=s.reshape(-1,48).mean(0),s.reshape(-1,48).std(0); tm,ts=t.reshape(-1,48).mean(0),t.reshape(-1,48).std(0); sp=(s-sm)/(ss+1e-5)*(.8*ts+.2*ss)+.8*tm+.2*sm
 def sample(c,n): z=c.reshape(-1,48); return z[rng.choice(len(z),n,replace=False)].astype('float32')
 si,spi,ti=sample(s,a.samples),sample(sp,a.samples),sample(t,a.samples)
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=SceneShiftNetDCRN().to(dev); ck=torch.load(HERE/'runs_sceneshiftnet_dcrn/houston/seed_1341/final.pth',map_location=dev); model.load_state_dict(ck['model']); model.eval()
 def feature(c,n):
  h=3; p=np.pad(c,((h,h),(h,h),(0,0))); ids=rng.choice(c.shape[0]*c.shape[1],n,replace=False); xs=np.stack([p[i//c.shape[1]:i//c.shape[1]+7,i%c.shape[1]:i%c.shape[1]+7].transpose(2,0,1) for i in ids]).astype('float32'); r=[]
  with torch.no_grad():
   for k in range(0,n,a.batch_size): r.append(model.encoder(torch.from_numpy(xs[k:k+a.batch_size]).to(dev)).cpu().numpy())
  return np.concatenate(r)
 fs,fsp,ft=feature(s,a.samples),feature(sp,a.samples),feature(t,a.samples); assert fs.shape[1]==fsp.shape[1]==ft.shape[1]==288
 rows=[]
 for space,x,xp,y in [('input_spectral',si,spi,ti),('dcrn_feature',fs,fsp,ft)]:
  before,after=metrics(x,y),metrics(xp,y)
  for k in before: rows.append({'space':space,'metric':k,'d_source_target':before[k],'d_shifted_target':after[k],'relative_change_percent':(after[k]-before[k])/before[k]*100 if before[k] else None})
 (out/'domain_gap_metrics.json').write_text(json.dumps(rows,indent=2));
 with (out/'domain_gap_metrics.csv').open('w') as f:
  f.write('Space,Metric,d(S,T),d(Sprime,T),Relative Change (%)\n'); [f.write(f"{r['space']},{r['metric']},{r['d_source_target']},{r['d_shifted_target']},{r['relative_change_percent']}\n") for r in rows]
 plot_tsne([si[:a.tsne_samples],spi[:a.tsne_samples],ti[:a.tsne_samples]],out/'tsne_input.png','Input spectral space'); plot_tsne([fs[:a.tsne_samples],fsp[:a.tsne_samples],ft[:a.tsne_samples]],out/'tsne_feature.png','DCRN feature space')
 print('Space | Metric | d(S,T) | d(Sprime,T) | Relative Change'); [print(r['space'],r['metric'],r['d_source_target'],r['d_shifted_target'],r['relative_change_percent']) for r in rows]
if __name__=='__main__': main()
