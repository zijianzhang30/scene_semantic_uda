"""Frozen pure-affine checkpoint: fixed-batch parameter and input-view gradients."""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

H=Path(__file__).resolve().parent
B=H/'runs_mluda_official_reproduction_v1/pavia'
P=H/'runs_mluda_official_pure_affine_no_extra_source_scl_v1/pavia'
O=H/'runs_official_shift_diagnostic/pavia_pure_affine_component_seed1256'

def cosine(a,b):
    return float(F.cosine_similarity(a.flatten(),b.flatten(),dim=0)) if a.norm()>1e-12 and b.norm()>1e-12 else None

def metrics(go,gl,gs):
    return dict(original_norm=float(go.norm()), lmmd_norm=float(gl.norm()), scl_norm=float(gs.norm()),
                lmmd_ratio=float(gl.norm()/go.norm().clamp_min(1e-12)),
                scl_ratio=float(gs.norm()/go.norm().clamp_min(1e-12)),
                cos_orig_lmmd=cosine(go,gl),cos_orig_scl=cosine(go,gs),cos_lmmd_scl=cosine(gl,gs))

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    O.mkdir(parents=True,exist_ok=True)
    if (O/'result.json').exists(): raise RuntimeError('Refuse to overwrite completed diagnostic')
    sys.path.insert(0,str(P/'source_snapshot'))
    import utils,mmd
    from net2 import DSANSS
    from contrastive_loss import SupConLoss
    os.chdir('/home/zhangzj26/TGRS_MLUDA-2024')
    utils.set_seed(1622)
    g={'__name__':'diagnostic_data_only'}
    exec(compile((B/'executed_entry.py').read_text().split('# Loss Function')[0],'preprocessing_only','exec'),g)
    utils.set_seed(1256)
    tx,ty=utils.get_sample_data(g['data_s'],g['label_s'],5,180)
    saved=np.load(P/'band_statistics.npz')
    rebuilt=[g['data_s'].mean((0,1)),g['data_s'].std((0,1)),g['data_t'].mean((0,1)),g['data_t'].std((0,1))]
    statdiff={k:float(np.max(np.abs(saved[k]-v))) for k,v in zip(['sm','ss','tm','ts'],rebuilt)}
    assert max(statdiff.values()) < 1e-10, statdiff
    provenance=np.load(B/'seed_1256/evaluation_provenance.npz')
    pad=np.pad(g['data_t'],((5,5),(5,5),(0,0)))
    order,rows,cols=[provenance[k] for k in ['target_order','target_rows','target_cols']]
    rng=np.random.RandomState(8108)
    indices=[np.concatenate([rng.choice(np.flatnonzero(ty==k),5 if k<4 else 4,replace=False) for k in range(7)]) for _ in range(4)]
    np.savez(O/'fixed_batches.npz',source_indices=np.stack(indices),target_indices=order[:128].reshape(4,32))
    cp=P/'seed_1256/final_epoch100.pth'
    digest=hashlib.sha256(cp.read_bytes()).hexdigest()
    model=DSANSS(102,11,7).cuda()
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
    frozen={k:v.detach().clone() for k,v in model.state_dict().items()}
    named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
    params=[p for n,p in named]
    cs=SupConLoss(temperature=.1).cuda(); ct=SupConLoss(temperature=.1).cuda()
    coef=.3*(2/(1+np.exp(-10))-1)
    def leaf(x): return x.detach().float().cuda().requires_grad_(True)
    def path(x,t):
        xv=[leaf(x),leaf(utils.radiation_noise(x.detach().cpu())),leaf(utils.flip_augmentation(x.detach().cpu()))]
        tv=[leaf(t),leaf(utils.radiation_noise(t.detach().cpu())),leaf(utils.flip_augmentation(t.detach().cpu()))]
        raw=model(xv[0],tv[0]); a=model(xv[1],tv[1]); b=model(xv[2],tv[2])
        lm=coef*mmd.lmmd(raw[0],raw[5],y,raw[8].softmax(1),BATCH_SIZE=32,CLASS_NUM=7)
        sl=cs(torch.stack([a[1],b[1]],1),y)
        tl=ct(torch.stack([a[7],b[7]],1),raw[8].detach().argmax(1))
        return raw,lm,sl,tl,xv,tv
    def gradients(loss,ps):
        vs=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
        return [(torch.zeros_like(p) if v is None else v).detach() for p,v in zip(ps,vs)]
    records=[]; samples=[]
    try:
        for j,ix in enumerate(indices):
            model.load_state_dict(frozen); model.train(); utils.set_seed(9000+j)
            x=torch.from_numpy(tx[ix]).cuda(); y=torch.from_numpy(ty[ix])
            ids=order[j*32:(j+1)*32]
            t=torch.from_numpy(np.stack([pad[rows[i]-5:rows[i]+6,cols[i]-5:cols[i]+6].transpose(2,0,1) for i in ids])).cuda()
            sm,ss,tm,ts=[torch.as_tensor(saved[k],device='cuda',dtype=x.dtype)[None,:,None,None] for k in ['sm','ss','tm','ts']]
            shifted=(x-sm)/(ss+1e-5)*(.8*ts+.2*ss)+.8*tm+.2*sm
            raw,lm,sl,tl,ov,ot=path(x,t)
            orig=F.cross_entropy(raw[3],y.cuda())+lm+sl+tl
            sr,elm,esl,est,ev,et=path(x,shifted)
            losses=[orig,.5*elm,.5*est]
            grads=[gradients(loss,params) for loss in losses]
            groups={}
            for group,prefix in [('all',''),('backbone','feature_layers.'),('classifier','fc1.'),('projection_heads','head'),('occupancy_head','fc2.')]:
                vectors=[torch.cat([v.flatten() for (n,p),v in zip(named,vs) if n.startswith(prefix)]) for vs in grads]
                groups[group]=metrics(*vectors)
            oi=gradients(orig,ov)
            li=gradients(.5*elm,ev); si=gradients(.5*est,ev)
            lt=gradients(.5*elm,et); st=gradients(.5*est,et)
            for i in range(32):
                cat=lambda vs: torch.cat([v[i].flatten() for v in vs])
                sample=dict(batch=j,source_index=int(ix[i]),source_class=int(y[i])+1,
                            shifted_pseudo=int(sr[8][i].detach().argmax())+1,
                            shifted_lmmd_view_norm=float(cat(lt).norm()),shifted_scl_view_norm=float(cat(st).norm()))
                sample.update(metrics(cat(oi),cat(li),cat(si)))
                samples.append(sample)
            records.append(dict(batch=j,groups=groups,pure_affine_mae=float((shifted-x).abs().mean()),
                loss_parts=dict(original=float(orig.detach()),original_lmmd=float(lm.detach()),
                                original_source_scl=float(sl.detach()),original_target_scl=float(tl.detach()),
                                weighted_extra_lmmd=float(losses[1].detach()),weighted_extra_scl=float(losses[2].detach()))))
            print('GRADIENT_BATCH',j,json.dumps(groups['all']),flush=True)
    finally:
        model.load_state_dict(frozen)
    assert all(torch.equal(v,frozen[k]) for k,v in model.state_dict().items())
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==digest
    assert all(p.grad is None for p in params)
    aggregate={}
    for group in records[0]['groups']:
        aggregate[group]={}
        for key in records[0]['groups'][group]:
            vals=[r['groups'][group][key] for r in records if r['groups'][group][key] is not None]
            aggregate[group][key]=dict(mean=float(np.mean(vals)),min=float(min(vals)),max=float(max(vals)),
                                      negative_batches=sum(v<0 for v in vals),defined_batches=len(vals),
                                      negative_fraction=sum(v<0 for v in vals)/len(vals)) if vals else None
    flat=[dict(batch=r['batch'],group=group,**v) for r in records for group,v in r['groups'].items()]
    classes=[]
    for k in range(1,8):
        subset=[r for r in samples if r['source_class']==k]
        row=dict(source_class=k,n=len(subset))
        for key in ['lmmd_norm','scl_norm','shifted_lmmd_view_norm','shifted_scl_view_norm','cos_orig_lmmd','cos_orig_scl','cos_lmmd_scl']:
            vals=[s[key] for s in subset if s[key] is not None]
            row[key+'_mean']=float(np.mean(vals)) if vals else None
        classes.append(row)
    report=dict(seed=1256,checkpoint=str(cp),checkpoint_sha256=digest,state_dict_unchanged=True,
        optimizer_steps=0,parameter_grad_buffers_unmodified=True,stats_max_abs_difference=statdiff,
        fixed_batch_policy='Same source indices RNG8108, diagnostic seed9000+j, original saved target order as previous diagnostic. Pure affine consumes no perturbation RNG; previous noisy diagnostic did.',
        class_attribution='Local raw/augmented source-view input gradients; not additive class parameter gradients. Counterpart view norms additionally mapped to source semantic class for post-hoc diagnostic only.',
        target_labels='Not used in objective/pseudo-labels; official historical target ordering reused.',
        gamma=.5,alpha=.8,aggregate=aggregate,batches=records,per_class=classes)
    write_csv(O/'parameter_gradients.csv',flat); write_csv(O/'sample_input_gradients.csv',samples); write_csv(O/'class_input_gradients.csv',classes)
    (O/'result.json').write_text(json.dumps(report,indent=2))
    print('DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__': main()
