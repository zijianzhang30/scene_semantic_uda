"""Offline input-spectral diagnostic. Writes only a new analysis directory.
GT is used only for class grouping; affine statistics use entire raw cubes.
MMD is biased multi-bandwidth RBF MMD^2, with a shared bandwidth before/after.
SAM is angle between class mean spectra (radians), not unpaired pixel angles.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
import train_sceneshiftnet_dcrn_houston_raw as houston
import train_sceneshiftnet_dcrn_pavia as pavia

ROOT = Path(__file__).resolve().parent

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def compare(s, shifted, t):
    s, shifted, t = [x.astype(np.float64) for x in (s, shifted, t)]
    z = np.concatenate((s, t))
    dd = cdist(z, z, 'sqeuclidean')
    positive = dd[dd > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    def metric(x):
        mu, mt = x.mean(0), t.mean(0)
        denom = np.linalg.norm(mu) * np.linalg.norm(mt)
        if denom == 0: raise ValueError('SAM undefined for zero mean spectrum')
        xx, tt, xt = cdist(x,x,'sqeuclidean'), cdist(t,t,'sqeuclidean'), cdist(x,t,'sqeuclidean')
        mmd = np.mean([np.exp(-xx/(2*bandwidth*k)).mean()+np.exp(-tt/(2*bandwidth*k)).mean()-2*np.exp(-xt/(2*bandwidth*k)).mean() for k in (.25,.5,1,2,4)])
        return {'Mean':float(np.linalg.norm(mu-mt)), 'MMD':float(mmd),
                'SAM':float(np.arccos(np.clip(mu@mt/denom,-1,1))),
                'CORAL':float(np.sum((np.cov(x,rowvar=False)-np.cov(t,rowvar=False))**2)/(4*x.shape[1]**2))}
    return metric(s), metric(shifted), bandwidth

def write_csv(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--samples-per-class',type=int,default=1000); ap.add_argument('--seed',type=int,default=1341); ap.add_argument('--output',type=Path,default=ROOT/'runs_sceneshiftnet_dcrn/classwise_gap_raw_seed1341'); args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    details=[]; summary=[]; provenance=[]; dynamics=[]
    for name,module,folder in [('Houston',houston,'houston_raw'),('Pavia',pavia,'pavia')]:
        roots={m:ROOT/'runs_sceneshiftnet_dcrn'/folder/m/'seed_1341' for m in ('baseline','sceneshift')}
        configs={m:json.loads((p/'config.json').read_text()) for m,p in roots.items()}
        cfg=configs['sceneshift']
        for k in ('normalization','ilda','seed','patch_size','bands','classes','source_samples_per_class','epochs','batch_size'):
            assert configs['baseline'][k]==cfg[k], k
        assert cfg['normalization']=='none' and cfg['ilda'] is False
        assert cfg['scene_shift_type']=='pure_affine' and not cfg['random_scale'] and not cfg['smooth_noise']
        s,sg,t,tg,paths=module.load_cubes(cfg['normalization'])
        for p in paths: assert sha(p)==cfg['data_sha256'][str(p)]
        s,t=np.asarray(s,dtype=np.float32),np.asarray(t,dtype=np.float32)
        sm,ss=s.reshape(-1,cfg['bands']).mean(0),s.reshape(-1,cfg['bands']).std(0)
        tm,ts=t.reshape(-1,cfg['bands']).mean(0),t.reshape(-1,cfg['bands']).std(0)
        metrics={m:json.loads((p/'metrics.json').read_text()) for m,p in roots.items()}
        for m,p in roots.items():
            h=json.loads((p/'history.json').read_text())
            dynamics.extend({'Dataset':name,'Method':m,**r} for r in h)
        rng=np.random.RandomState(args.seed)
        samples={}
        for c in range(1,cfg['classes']+1):
            si=np.flatnonzero(sg.reshape(-1)==c); ti=np.flatnonzero(tg.reshape(-1)==c)
            si=rng.choice(si,min(len(si),args.samples_per_class),replace=False); ti=rng.choice(ti,min(len(ti),args.samples_per_class),replace=False)
            samples[f'source_{c}']=si; samples[f'target_{c}']=ti
            x=s.reshape(-1,cfg['bands'])[si]; y=t.reshape(-1,cfg['bands'])[ti]
            a=cfg['alpha']; shifted=(x-sm)/(ss+1e-5)*(a*ts+(1-a)*ss)+a*tm+(1-a)*sm
            before,after,bw=compare(x,shifted,y)
            changes={k:100*(after[k]-before[k])/before[k] if before[k]>1e-15 else None for k in before}
            for k in before:
                assert np.isfinite(before[k]) and np.isfinite(after[k])
                details.append({'Dataset':name,'Class':c,'Metric':k,'d(S_c,T_c)':before[k],"d(Sprime_c,T_c)":after[k],'Relative Change (%)':changes[k],'source_n':len(x),'target_n':len(y),'mmd_bandwidth_sq':bw})
            b,o=[metrics[m]['per_class_accuracy'][c-1] for m in ('baseline','sceneshift')]
            summary.append({'Dataset':name,'Class':c,**{k+' change (%)':changes[k] for k in before},'Baseline Acc':b,'SceneShift Acc':o,'Acc Change (pp)':o-b})
        np.savez(args.output/(name+'_sample_indices.npz'),**samples)
        provenance.append({'dataset':name,'configs':configs,'checkpoint_sha256':{m:sha(p/'final.pth') for m,p in roots.items()},'space':'input raw center-pixel spectra; no feature extraction','statistics':'all cube pixels including unlabeled/background, float32 as trainer','sampling':'up to cap per class, all labeled pixels eligible, same source samples before/after','target_gt':'offline grouping only','metrics':'Mean L2; biased multiscale MMD^2 shared median bandwidth from S/T; SAM class-mean angle; squared covariance CORAL /(4*d*d)','accuracy':'existing final metrics.json; not reevaluated'})
    write_csv(args.output/'distances.csv',details); write_csv(args.output/'summary.csv',summary)
    (args.output/'pseudo_dynamics.json').write_text(json.dumps(dynamics,indent=2))
    (args.output/'provenance.json').write_text(json.dumps({'script_sha256':sha(__file__),'seed':args.seed,'cap':args.samples_per_class,'datasets':provenance},indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
