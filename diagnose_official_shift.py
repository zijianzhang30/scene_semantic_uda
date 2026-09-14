"""Frozen Pavia seed1622 diagnostic. No optimizer, no checkpoint writes."""
import json, os, sys, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
B=HERE/'runs_mluda_official_reproduction_v1/pavia'
S=HERE/'runs_mluda_official_sceneshift_v1/pavia'
OUT=HERE/'runs_official_shift_diagnostic/pavia_seed1622'

def stats(x):
    x=x.detach().float()
    return dict(min=x.min().item(),max=x.max().item(),mean=x.mean().item(),std=x.std(unbiased=False).item())

def main(seed=1622, output=None):
    OUT=Path(output) if output else HERE/f'runs_official_shift_diagnostic/pavia_seed{seed}'
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'result.json').exists(): raise RuntimeError('Diagnostic already completed')
    sys.path.insert(0,str(B/'source_snapshot'))
    import utils,mmd
    from net2 import DSANSS
    from contrastive_loss import SupConLoss
    os.chdir('/home/zhangzj26/TGRS_MLUDA-2024')
    utils.set_seed(1622)  # Same preprocessing replay as the first diagnostic.
    code=(B/'executed_entry.py').read_text().split('# Loss Function')[0]
    g={'__name__':'diagnostic_data_only'}
    exec(compile(code,'official_preprocessing_only','exec'),g)
    source,target=g['data_s'],g['data_t']
    gt=g['label_s']; mask=g['label_t']>0
    utils.set_seed(seed)
    tx,ty=utils.get_sample_data(source,gt,5,180)
    saved=np.load(S/'band_statistics.npz')
    shift_ns={'torch':torch,'F':F}
    exec((S/'scene_shift_function.py').read_text(),shift_ns)
    shift=shift_ns['shift']
    report={'scope':f'Pavia seed{seed} frozen final checkpoints, not training-time causal proof', 'seed':seed,
            'reconstructed_ilda':'same official call; fixed seed for diagnostic, not original pre-ILDA RNG',
            'target_labels':'only final evaluation; mask used for support diagnostic', 'models':{}}
    full=target.reshape(-1,102); selected=target[mask]
    report['target_support']={'mask_fraction':float(mask.mean()),
      'mean_abs_full_vs_mask_mean':float(np.abs(full.mean(0)-selected.mean(0)).mean()),
      'mean_abs_full_vs_mask_std':float(np.abs(full.std(0)-selected.std(0)).mean()),
      'saved_vs_reconstructed_stats_max':{k:float(np.abs(saved[k]-v).max()) for k,v in
       zip(['sm','ss','tm','ts'],[source.mean((0,1)),source.std((0,1)),full.mean(0),full.std(0)])}}
    provenance=np.load(B/f'seed_{seed}/evaluation_provenance.npz')
    order=provenance['target_order']; rows=provenance['target_rows']; cols=provenance['target_cols']
    pad=np.pad(target,((5,5),(5,5),(0,0)))
    def target_batch(start):
        ids=order[start:start+32]
        return torch.from_numpy(np.stack([pad[rows[i]-5:rows[i]+6,cols[i]-5:cols[i]+6].transpose(2,0,1) for i in ids])).cuda()
    # Four deterministic source batches balanced across classes; no target GT involved.
    rng=np.random.RandomState(8108)
    indices=[]
    for j in range(4):
        ix=np.concatenate([rng.choice(np.flatnonzero(ty==k),5 if k<4 else 4,replace=False) for k in range(7)])
        indices.append(ix)
    for name,root in [('original',B),('shift',S)]:
        cp=root/f'seed_{seed}/final_epoch100.pth'
        checkpoint_hash_before=hashlib.sha256(cp.read_bytes()).hexdigest()
        model=DSANSS(102,11,7).cuda()
        checkpoint=torch.load(cp,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model'])
        frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        cs=SupConLoss(temperature=.1).cuda(); ct=SupConLoss(temperature=.1).cuda()
        params=[p for p in model.parameters() if p.requires_grad]
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        offsets={}; offset=0
        for n,p in named:
            offsets[n]=(offset,offset+p.numel()); offset+=p.numel()
        def buffer_snapshot():
            return {n:v.detach().clone() for n,v in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
        def buffer_delta(before,after):
            per={n:stats((after[n]-v).float()) for n,v in before.items()}
            sums={}
            for field in ['running_mean','running_var']:
                diffs=torch.cat([(after[n]-v).flatten() for n,v in before.items() if field in n])
                sums[field]={'mean_abs':float(diffs.abs().mean()),'max_abs':float(diffs.abs().max()),'l2':float(diffs.norm())}
            return {'aggregate':sums,'per_buffer':per}
        records=[]
        for j,ix in enumerate(indices):
            model.load_state_dict(frozen); model.train(); utils.set_seed(9000+j)
            x=torch.from_numpy(tx[ix]).cuda(); y=torch.from_numpy(ty[ix]); t=target_batch(j*32)
            shifted=shift(x,*(saved[k] for k in ['sm','ss','tm','ts']),alpha=.8)
            sm,ss,tm,ts=[torch.tensor(saved[k],device='cuda')[None,:,None,None] for k in ['sm','ss','tm','ts']]
            transport=(x-sm)/(ss+1e-5)*(.8*ts+.2*ss)+.8*tm+.2*sm
            def path(counter):
                x0=utils.radiation_noise(x.cpu()).float().cuda(); x1=utils.flip_augmentation(x.cpu()).cuda()
                t0=utils.radiation_noise(counter.cpu()).float().cuda(); t1=utils.flip_augmentation(counter.cpu()).cuda()
                raw=model(x,counter); a=model(x0,t0); b=model(x1,t1)
                lm=.3*(2/(1+np.exp(-10))-1)*mmd.lmmd(raw[0],raw[5],y,raw[8].softmax(1),BATCH_SIZE=32,CLASS_NUM=7)
                sl=cs(torch.stack([a[1],b[1]],1),y)
                tl=ct(torch.stack([a[7],b[7]],1),raw[8].detach().argmax(1))
                return raw,lm+sl+tl,dict(lmmd=float(lm.detach()),source_scl=float(sl.detach()),counterpart_scl=float(tl.detach()))
            raw,adapt,parts=path(t)
            original=F.cross_entropy(raw[3],y.cuda())+adapt
            before_extra=buffer_snapshot()
            sr,extra,extra_parts=path(shifted)
            bn_change=buffer_delta(before_extra,buffer_snapshot())
            def grad(loss):
                return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten() for p,v in zip(params,torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True))])
            go=grad(original); ge=grad(.5*extra)
            record={'batch':j,'source':stats(x),'shifted':stats(shifted),
                    'transport_only_abs_delta':float((transport-x).abs().mean()),
                    'total_shift_abs_delta':float((shifted-x).abs().mean()),
                    'original_loss_parts':parts,'extra_loss_parts':extra_parts,
                    'original_grad_norm':float(go.norm()),'weighted_extra_grad_norm':float(ge.norm()),
                    'extra_branch_bn_delta':bn_change,'layer_gradients':{},
                    'weighted_extra_grad_norm_ratio':float(ge.norm()/go.norm().clamp_min(1e-12)),
                    'gradient_cosine':float(F.cosine_similarity(go,ge,dim=0)),
                    'per_class':{}}
            for group,prefixes in {'backbone':['feature_layers.'],'classifier':['fc1.'],
                                  'projection_heads':['head1.','head2.']}.items():
                slices=[offsets[n] for n,p in named if any(n.startswith(pre) for pre in prefixes)]
                u=torch.cat([go[a:b] for a,b in slices]); v=torch.cat([ge[a:b] for a,b in slices])
                record['layer_gradients'][group]={'original_norm':float(u.norm()),'weighted_extra_norm':float(v.norm()),
                    'ratio':float(v.norm()/u.norm().clamp_min(1e-12)),
                    'cosine':float(F.cosine_similarity(u,v,dim=0)) if u.norm()>0 and v.norm()>0 else None}
            pred=sr[8].detach().argmax(1).cpu(); raw_pred=raw[3].detach().argmax(1).cpu()
            for k in range(7):
                mk=y==k
                record['per_class'][str(k+1)]={'n':int(mk.sum()),'raw_source_accuracy':float((raw_pred[mk]==y[mk]).float().mean()),
                    'shift_pseudo_source_agreement':float((pred[mk]==y[mk]).float().mean()),
                    'shift_abs_delta':float((shifted[mk]-x[mk]).abs().mean()),
                    'affine_abs_delta':float((transport[mk]-x[mk]).abs().mean())}
            records.append(record)
            del raw,sr,original,extra,adapt,go,ge
        # Reset all BN buffers before frozen inference. No recalibration performed.
        model.load_state_dict(frozen); model.eval()
        refs={'official_last':torch.tensor(provenance['source_reference']).cuda(),
              'fixed_first32':torch.tensor(tx[:32]).cuda()}
        results={}
        from sklearn.metrics import confusion_matrix,cohen_kappa_score
        labels=provenance['labels']
        for refname,ref in refs.items():
            preds=[]
            with torch.no_grad():
                for start in range(0,len(labels),32):
                    preds.extend(model(ref,target_batch(start))[8].argmax(1).cpu().tolist())
            pred=np.array(preds); cm=confusion_matrix(labels,pred,labels=np.arange(7)); pc=np.diag(cm)/cm.sum(1)
            results[refname]={'oa_official_denominator':float((pred==labels).sum()/len(order)*100),
                             'oa_evaluated_subset':float((pred==labels).mean()*100),'aa':float(pc.mean()*100),
                             'kappa':float(cohen_kappa_score(labels,pred)*100),'per_class':(pc*100).tolist(),
                             'original_saved_prediction_agreement':float((pred==np.load(root/f'seed_{seed}/evaluation_provenance.npz')['prediction']).mean())}
        assert all(torch.equal(v.cpu(),frozen[k]) for k,v in model.state_dict().items())
        assert hashlib.sha256(cp.read_bytes()).hexdigest()==checkpoint_hash_before
        report['models'][name]={'batches':records,'reference_evaluation':results,'state_restored_verified':True,
                               'checkpoint_sha256_unchanged':checkpoint_hash_before}
        (OUT/'partial.json').write_text(json.dumps(report,indent=2))
        print(name,results,flush=True)
    (OUT/'result.json').write_text(json.dumps(report,indent=2))
    print('DIAGNOSTIC_COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=1622); p.add_argument('--output')
    a=p.parse_args(); main(a.seed,a.output)
