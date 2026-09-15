"""Post-hoc mask/class-label oracle. Frozen eval only, never a method score."""
import csv,json,os,sys,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent
B=HERE/'runs_mluda_official_reproduction_v1/pavia'
S=HERE/'runs_mluda_official_sceneshift_v1/pavia'
OUT=HERE/'runs_official_shift_diagnostic/pavia_support_oracle'

def moments(a): return a.mean(0),a.std(0)
def distance(a,b):
    return {'mean_profile_mae':float(np.abs(a[0]-b[0]).mean()),
            'std_profile_mae':float(np.abs(a[1]-b[1]).mean())}

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'result.json').exists(): raise RuntimeError('Already completed')
    sys.path.insert(0,str(B/'source_snapshot'))
    import utils
    from net2 import DSANSS
    os.chdir('/home/zhangzj26/TGRS_MLUDA-2024'); utils.set_seed(1622)
    g={'__name__':'oracle_preprocessing_only'}
    exec(compile((B/'executed_entry.py').read_text().split('# Loss Function')[0],'data_only','exec'),g)
    source,target,sg,tg=g['data_s'],g['data_t'],g['label_s'],g['label_t']
    saved=np.load(S/'band_statistics.npz'); sm,ss=saved['sm'],saved['ss']
    full=(saved['tm'],saved['ts']); mask=moments(target[tg>0])
    class_mom={k:moments(target[tg==k]) for k in range(1,8)}
    mus=np.stack([class_mom[k][0].astype(np.float64) for k in range(1,8)])
    sigmas=np.stack([class_mom[k][1].astype(np.float64) for k in range(1,8)])
    mu=mus.mean(0); sigma=np.sqrt(np.maximum((sigmas**2+mus**2).mean(0)-mu**2,0))
    schemes={'Full':full,'Mask':mask,'ClassBalanced':(mu,sigma)}
    result={'oracle_only':True,'not_formal_target_performance':True,'alpha':.8,'source_statistics_unchanged':True,
            'definitions':{'balanced':'equal-class mixture; variance=E(sigma_k^2+mu_k^2)-E(mu_k)^2, not mean std',
              'spectral':'center-pixel spectrum mean/std MAE; source sample label is patch-center GT',
              'feature':'frozen eval target-side pre-classifier feature; L2 normalized; mean top10 cosine distance to fixed1024 target-bank patches',
              'support':'Mask and class-balanced use target annotation strictly post-hoc; class ids shared by official seven-class GT',
              'CORAL_MMD':'not computed; no new estimator introduced; mean/std and feature KNN reported separately'},
            'target_support_comparison':{'Full_vs_Mask':distance(full,mask),'Full_vs_ClassBalanced':distance(full,(mu,sigma)),
                                         'Mask_vs_ClassBalanced':distance(mask,(mu,sigma))},'runs':{}}
    with (OUT/'per_band_stats.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['band','source_mean','source_std','full_mean','full_std','mask_mean','mask_std','balanced_mean','balanced_std','mask_minus_full_mean','mask_minus_full_std'])
        for b in range(102):w.writerow([b+1,sm[b],ss[b],full[0][b],full[1][b],mask[0][b],mask[1][b],mu[b],sigma[b],mask[0][b]-full[0][b],mask[1][b]-full[1][b]])
    pad=np.pad(target,((5,5),(5,5),(0,0)))
    rng=np.random.RandomState(77102)
    pools={'official_mask':np.argwhere(tg>0),'full_cube':None}
    bank_coords={}
    bank_coords['official_mask']=pools['official_mask'][rng.choice(len(pools['official_mask']),1024,replace=False)]
    ii=rng.choice(target.shape[0]*target.shape[1],1024,replace=False)
    bank_coords['full_cube']=np.column_stack(np.unravel_index(ii,target.shape[:2]))
    np.savez(OUT/'target_bank_coords.npz',**bank_coords)
    def patches(coords):
        return torch.tensor(np.stack([pad[r:r+11,c:c+11].transpose(2,0,1) for r,c in coords]),device='cuda')
    class_rows=[]
    for seed in [1622,1256]:
        utils.set_seed(seed);tx,ty=utils.get_sample_data(source,sg,5,180)
        rng=np.random.RandomState(8108)
        ids=[np.concatenate([rng.choice(np.flatnonzero(ty==k),5 if k<4 else 4,replace=False) for k in range(7)]) for j in range(4)]
        np.savez(OUT/f'source_batch_indices_{seed}.npz',indices=np.stack(ids))
        ys=np.concatenate([ty[ix] for ix in ids]); source_x=torch.tensor(np.concatenate([tx[ix] for ix in ids]),device='cuda')
        versions={'Source':source_x}; to=lambda a:torch.as_tensor(a,device='cuda',dtype=source_x.dtype)[None,:,None,None]
        for scheme,(tm,ts) in schemes.items():
            affine=(source_x-to(sm))/(to(ss)+1e-5)*(.8*to(ts)+.2*to(ss))+.8*to(tm)+.2*to(sm)
            versions[scheme+'_affine']=affine
            pieces=[]
            for j in range(4):
                torch.manual_seed(9000+j);torch.cuda.manual_seed_all(9000+j)
                z=affine[j*32:(j+1)*32]
                pieces.append(z*(1+.04*torch.randn(32,1,1,1,device='cuda'))+.015*F.avg_pool2d(torch.randn_like(z),5,1,2))
            versions[scheme+'_noisy']=torch.cat(pieces)
        transport={}
        for name,x in versions.items():
            spectra=x[:,:,5,5].cpu().numpy(); item={'input_mae':float((x-source_x).abs().mean()),
                'spectral_proximity':{'official_mask':distance(moments(spectra),mask),'full_cube':distance(moments(spectra),full)},'per_class':{}}
            for k in range(1,8):
                select=ys==k-1; prof=moments(spectra[select]); d=distance(prof,class_mom[k]); change=float((x[select]-source_x[select]).abs().mean())
                item['per_class'][str(k)]={'n':int(select.sum()),'input_mae':change,'target_class_profile_distance':d,
                    'source_mean_profile':prof[0].tolist(),'source_std_profile':prof[1].tolist(),
                    'target_mean_profile':class_mom[k][0].tolist(),'target_std_profile':class_mom[k][1].tolist()}
                class_rows.append([seed,name,k,int(select.sum()),change,d['mean_profile_mae'],d['std_profile_mae']])
            transport[name]=item
        run={'transport':transport,'models':{}}
        for name,root in [('original',B),('shift',S)]:
            cp=root/f'seed_{seed}/final_epoch100.pth'; h=hashlib.sha256(cp.read_bytes()).hexdigest()
            model=DSANSS(102,11,7).cuda(); model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);model.eval()
            state={k:v.cpu().clone() for k,v in model.state_dict().items()}
            ref=torch.tensor(np.load(root/f'seed_{seed}/evaluation_provenance.npz')['source_reference'],device='cuda')
            banks={}; scores={}
            with torch.no_grad():
                for pool,coords in bank_coords.items():
                    banks[pool]=F.normalize(torch.cat([model(ref,patches(coords[i:i+32]))[5] for i in range(0,1024,32)]),dim=1)
                for version,x in versions.items():
                    feats=[];pseudos=[];raw_preds=[]
                    for i in range(0,128,32):
                        # source/shift counterpart paired exactly; no BN updates in eval.
                        out=model(source_x[i:i+32],x[i:i+32]);feats.append(out[5]);pseudos.append(out[8].argmax(1));raw_preds.append(out[3].argmax(1))
                    feat=F.normalize(torch.cat(feats),dim=1); pred=torch.cat(pseudos).cpu().numpy(); raw_pred=torch.cat(raw_preds).cpu().numpy()
                    proximity={pool:(1-(feat@bank.T).topk(10,dim=1).values.mean(1)).cpu().numpy() for pool,bank in banks.items()}
                    scores[version]={'pseudo_consistency':float((pred==ys).mean()),'raw_source_accuracy':float((raw_pred==ys).mean()),
                        'feature_knn':{pool:float(d.mean()) for pool,d in proximity.items()},
                        'per_class':{str(k):{'consistency':float((pred[ys==k-1]==ys[ys==k-1]).mean()),
                                            'feature_knn':{pool:float(d[ys==k-1].mean()) for pool,d in proximity.items()}} for k in range(1,8)}}
            assert all(torch.equal(v.cpu(),state[k]) for k,v in model.state_dict().items())
            assert hashlib.sha256(cp.read_bytes()).hexdigest()==h
            run['models'][name]={'versions':scores,'state_and_checkpoint_unchanged':True}
        result['runs'][str(seed)]=run
        (OUT/'partial.json').write_text(json.dumps(result,indent=2));print('ORACLE_SEED_COMPLETE',seed,flush=True)
    with (OUT/'per_class_stats.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','version','class','n','input_mae','target_class_mean_profile_mae','target_class_std_profile_mae']);w.writerows(class_rows)
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print('ORACLE_COMPLETE',flush=True)

if __name__=='__main__':main()
