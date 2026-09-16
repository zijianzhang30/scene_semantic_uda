"""Deterministic Houston F0--F4: SceneShift, residual GRL alignment, and soft MCC."""
import argparse, csv, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_dcrn import SceneShiftNetDCRN
from sceneshiftnet_dcrn_train import PatchDataset, evaluate, file_sha256, sample_source
from train_sceneshiftnet_dcrn_houston_raw import load_cubes

HERE = Path(__file__).resolve().parent
ROOT_OUT = HERE / 'runs_sceneshiftnet_dcrn/houston_raw/residual_mcc'
SETTINGS = {
    'F0': dict(adaptive=False, adv=False, mcc=False),
    'F1': dict(adaptive=False, adv=True,  mcc=False),
    'F2': dict(adaptive=True,  adv=True,  mcc=False),
    'F3': dict(adaptive=True,  adv=False, mcc=True),
    'F4': dict(adaptive=True,  adv=True,  mcc=True),
    'G0': dict(adaptive=True,  adv=False, mcc=True, selective=False, w_max=1.0),
    'G1': dict(adaptive=True,  adv=False, mcc=True, selective=True,  w_max=2.0),
    'G2': dict(adaptive=True,  adv=False, mcc=True, selective=True,  w_max=1.5),
    'H0': dict(adaptive=True,  adv=False, mcc=True, selective=False, normalized_selective=False, w_max=1.0),
    'H1': dict(adaptive=True,  adv=False, mcc=True, selective=False, normalized_selective=True,  w_max=1.5),
    'H2': dict(adaptive=True,  adv=False, mcc=True, selective=False, normalized_selective=True,  w_max=1.25),
}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient): ctx.coefficient = coefficient; return x.view_as(x)
    @staticmethod
    def backward(ctx, grad): return -ctx.coefficient * grad, None


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(288,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(), nn.Linear(64,2))
    def forward(self, x, coefficient): return self.net(GRL.apply(x, coefficient))


def mcc_loss(logits, temperature=2.5):
    probabilities = (logits / temperature).softmax(1)
    entropy = -(probabilities * (probabilities + 1e-5).log()).sum(1)
    weights = 1.0 + torch.exp(-entropy)
    weights = weights * (len(weights) / weights.sum())
    confusion = (probabilities * weights[:, None]).T @ probabilities
    confusion = confusion / (confusion.sum(1, keepdim=True) + 1e-5)
    off_diagonal = (confusion.sum() - confusion.diag().sum()) / confusion.shape[0]
    return off_diagonal, probabilities, confusion


def selective_mcc(confusion, gamma=.5, w_min=.5, w_max=2.0):
    off_mask = ~torch.eye(confusion.shape[0], dtype=torch.bool, device=confusion.device)
    off_values = confusion[off_mask]
    with torch.no_grad():
        weights = (off_values / (off_values.mean() + 1e-5)).pow(gamma).clamp(w_min, w_max)
    loss = (weights * off_values).sum() / (weights.sum() + 1e-5)
    weight_matrix = torch.zeros_like(confusion)
    weight_matrix[off_mask] = weights
    weighted_matrix = weight_matrix * confusion
    return loss, weight_matrix, weighted_matrix


def normalized_selective_mcc(confusion, gamma=.5, w_min=.5, w_max=1.5):
    classes = confusion.shape[0]
    off_mask = ~torch.eye(classes, dtype=torch.bool, device=confusion.device)
    off_values = confusion[off_mask]
    with torch.no_grad():
        raw_weights = (off_values / (off_values.mean() + 1e-5)).pow(gamma).clamp(w_min, w_max)
        normalized_weights = raw_weights / (raw_weights.mean() + 1e-5)
        # Cancel the epsilon-induced common scale so mean(w_hat) is exactly 1.
        normalized_weights = normalized_weights / normalized_weights.mean()
    loss = (normalized_weights * off_values).sum() / classes
    raw_weight_matrix = torch.zeros_like(confusion)
    normalized_weight_matrix = torch.zeros_like(confusion)
    raw_weight_matrix[off_mask] = raw_weights
    normalized_weight_matrix[off_mask] = normalized_weights
    weighted_matrix = normalized_weight_matrix * confusion
    return loss, raw_weight_matrix, normalized_weight_matrix, weighted_matrix


def distances(a, b):
    mean = float(np.linalg.norm(a.mean(0)-b.mean(0)))
    ca, cb = np.cov(a,rowvar=False), np.cov(b,rowvar=False)
    coral = float(np.linalg.norm(ca-cb,'fro')/(4*a.shape[1]**2))
    x,y=torch.from_numpy(a),torch.from_numpy(b); z=torch.cat((x,y)); d=torch.cdist(z,z).square()
    med=torch.median(d[d>0]); vals=[]; n=len(x)
    for scale in (.25,.5,1,2,4):
        k=torch.exp(-d/(2*med*scale+1e-12)); vals.append(k[:n,:n].mean()+k[n:,n:].mean()-2*k[:n,n:].mean())
    return {'mean_distance':mean,'coral':coral,'mmd_rbf':float(torch.stack(vals).mean())}


@torch.no_grad()
def feature_gap(model, shifted_ds, target_ds, indices, device):
    model.eval(); results=[]
    for dataset in (shifted_ds,target_ds):
        values=[]
        for begin in range(0,len(indices),128):
            batch=torch.stack([dataset[int(i)] for i in indices[begin:begin+128]]).to(device)
            values.append(model.encoder(batch).cpu().numpy())
        results.append(np.concatenate(values))
    return distances(*results)


def run(args):
    setting=SETTINGS[args.experiment]; seed_all(args.seed)
    suffix=f'smoke_{args.epochs}ep' if args.epochs != 100 else 'seed_1341'
    if args.experiment.startswith('G'):
        output_root = HERE/'runs_sceneshiftnet_dcrn/houston_raw/selective_mcc'
    elif args.experiment.startswith('H'):
        output_root = HERE/'runs_sceneshiftnet_dcrn/houston_raw/normalized_selective_mcc'
    else:
        output_root = ROOT_OUT
    out=output_root/args.experiment/suffix; out.mkdir(parents=True,exist_ok=False)
    source,sg,target,tg,paths=load_cubes('none'); source=source.astype('float32');target=target.astype('float32')
    centers,labels=sample_source(sg,7,180,np.random.RandomState(args.seed)); target_centers=np.argwhere(tg>0)
    generator=torch.Generator().manual_seed(args.seed)
    source_loader=DataLoader(PatchDataset(source,centers,7,labels),32,shuffle=True,drop_last=True,generator=generator)
    target_loader=DataLoader(PatchDataset(target,target_centers,7),32,shuffle=True,drop_last=True,generator=generator)
    eval_loader=DataLoader(PatchDataset(target,target_centers,7,tg[target_centers[:,0],target_centers[:,1]].astype('int64')-1),32,shuffle=False)
    sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0)
    discrepancy=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5)
    adaptive=.4+.5*(discrepancy-discrepancy.min())/(discrepancy.max()-discrepancy.min()+1e-5)
    alpha=adaptive if setting['adaptive'] else np.full(48,.8,dtype='float32')
    shifted_cube=(source-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
    # Fixed, GT-free diagnostic sample positions. Same positions in S' and T are not semantically paired.
    gap_rng=np.random.RandomState(args.seed+99); gap_n=1000
    gap_indices=gap_rng.choice(min(source.shape[0]*source.shape[1],target.shape[0]*target.shape[1]),gap_n,replace=False)
    width=min(source.shape[1],target.shape[1]); gap_centers=np.stack((gap_indices//width,gap_indices%width),1)
    gap_centers[:,0]=np.minimum(gap_centers[:,0],np.minimum(source.shape[0],target.shape[0])-1)
    gap_centers[:,1]=np.minimum(gap_centers[:,1],np.minimum(source.shape[1],target.shape[1])-1)
    shifted_diag=PatchDataset(shifted_cube,gap_centers,7); target_diag=PatchDataset(target,gap_centers,7)
    config=dict(experiment=args.experiment,seed=args.seed,epochs=args.epochs,batch_size=32,patch_size=7,source_per_class=180,
      optimizer='Adam',lr=.001,weight_decay=0,normalization='none',use_ilda=False,deterministic=True,
      forward_order='source -> shifted-source -> target; exactly once each per iteration for every experiment',
      scene_shift='adaptive' if setting['adaptive'] else 'fixed',alpha=.8 if not setting['adaptive'] else None,
      alpha_min=.4 if setting['adaptive'] else None,alpha_max=.9 if setting['adaptive'] else None,
      lambda_shift=.5,lambda_adv=.01 if setting['adv'] else 0,lambda_mcc=.1 if setting['mcc'] else 0,
      lambda_smcc=.1 if setting.get('selective',False) else 0,
      lambda_nsmcc=.1 if setting.get('normalized_selective',False) else 0,
      mcc_variant='normalized_selective' if setting.get('normalized_selective',False) else ('selective_weighted' if setting.get('selective',False) else 'original'),
      selective_gamma=.5 if setting.get('selective',False) else None,
      selective_w_min=.5 if setting.get('selective',False) else None,
      selective_w_max=setting.get('w_max') if setting.get('selective',False) else None,
      selective_weight_gradient='detached' if setting.get('selective',False) else None,
      normalized_selective=setting.get('normalized_selective',False),
      normalized_selective_gamma=.5 if setting.get('normalized_selective',False) else None,
      normalized_selective_w_min=.5 if setting.get('normalized_selective',False) else None,
      normalized_selective_w_max=setting.get('w_max') if setting.get('normalized_selective',False) else None,
      normalized_selective_weight_gradient='detached' if setting.get('normalized_selective',False) else None,
      normalized_selective_scale='sum(weight_hat * offdiag_M) / C' if setting.get('normalized_selective',False) else None,
      grl_schedule='2/(1+exp(-10*global_progress))-1',temperature=2.5,target_labels_in_training=False,
      target_hard_pseudo_labels=False,code_sha256=file_sha256(__file__),model_sha256=file_sha256(HERE/'models/sceneshift_net_dcrn.py'),
      data_hash={str(p):file_sha256(p) for p in paths})
    (out/'config.json').write_text(json.dumps(config,indent=2));np.savez(out/'indices.npz',source=centers,source_labels=labels,target=target_centers,gap_centers=gap_centers)
    (out/'band_statistics.json').write_text(json.dumps({'d_b':discrepancy.tolist(),'alpha_b':alpha.tolist()},indent=2))
    assert torch.cuda.is_available();device='cuda';model=SceneShiftNetDCRN(48,7,7).to(device);disc=Discriminator().to(device)
    parameters=list(model.parameters())+(list(disc.parameters()) if setting['adv'] else [])
    optimizer=torch.optim.Adam(parameters,lr=.001)
    stat=[torch.tensor(v,device=device)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)];sm_t,ss_t,tm_t,ts_t,alpha_t=stat
    initial_gap=feature_gap(model,shifted_diag,target_diag,np.arange(gap_n),device) if setting['adv'] else None
    history=[]; total_steps=args.epochs*len(source_loader);global_step=0
    for epoch in range(1,args.epochs+1):
        model.train();disc.train();target_it=iter(target_loader);sums=np.zeros(6);correct=np.zeros(2);n=domain_n=domain_ok=0;soft_sum=np.zeros(7);conf_sum=np.zeros((7,7));weighted_sum=np.zeros((7,7));weight_sum=np.zeros((7,7));normalized_weight_sum=np.zeros((7,7));
        for x,y in source_loader:
            try:z=next(target_it)
            except StopIteration:target_it=iter(target_loader);z=next(target_it)
            x,y,z=x.to(device),y.to(device),z.to(device)
            fs,ls=model(x)
            shifted=(x-sm_t)/(ss_t+1e-5)*(alpha_t*ts_t+(1-alpha_t)*ss_t)+alpha_t*tm_t+(1-alpha_t)*sm_t
            fss,lss=model(shifted);ft,lt=model(z)
            progress=global_step/max(total_steps-1,1);coefficient=2/(1+math.exp(-10*progress))-1
            adv=ls.sum()*0
            if setting['adv']:
                dl=torch.cat((disc(fss,coefficient),disc(ft,coefficient)));dy=torch.cat((torch.zeros(len(x),dtype=torch.long,device=device),torch.ones(len(z),dtype=torch.long,device=device)))
                adv=F.cross_entropy(dl,dy);domain_ok+=int((dl.argmax(1)==dy).sum());domain_n+=len(dy)
            mcc,prob,conf=mcc_loss(lt)
            selective,weight_matrix,weighted_matrix=selective_mcc(conf,w_max=setting.get('w_max',2.0))
            normalized_selective,raw_weight_matrix,normalized_weight_matrix,normalized_weighted_matrix=normalized_selective_mcc(conf,w_max=setting.get('w_max',1.5))
            if setting.get('normalized_selective',False):
                target_regularizer=normalized_selective
                weight_matrix=raw_weight_matrix
                weighted_matrix=normalized_weighted_matrix
            else:
                target_regularizer=selective if setting.get('selective',False) else mcc
            if not setting.get('selective',False):
                if not setting.get('normalized_selective',False):
                    weight_matrix=(~torch.eye(7,dtype=torch.bool,device=device)).to(conf.dtype)
                    normalized_weight_matrix=weight_matrix
                    weighted_matrix=weight_matrix*conf
            src_loss=F.cross_entropy(ls,y);shift_loss=F.cross_entropy(lss,y)
            loss=src_loss+.5*shift_loss+.01*adv+(0.1*target_regularizer if setting['mcc'] else 0)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            optimizer.zero_grad();loss.backward()
            if setting['adv'] and args.epochs==1 and global_step==0:
                grad_adv=torch.autograd.grad(adv,fs,allow_unused=True,retain_graph=True)[0] if False else None
            optimizer.step();bs=len(y);n+=bs;global_step+=1
            sums+=np.array([src_loss.item(),shift_loss.item(),adv.item(),mcc.item(),target_regularizer.item(),normalized_selective.item()])*bs;correct+=[int((ls.argmax(1)==y).sum()),int((lss.argmax(1)==y).sum())]
            soft_sum+=prob.detach().sum(0).cpu().numpy();conf_sum+=conf.detach().cpu().numpy()*bs;weighted_sum+=weighted_matrix.detach().cpu().numpy()*bs;weight_sum+=weight_matrix.detach().cpu().numpy()*bs;normalized_weight_sum+=normalized_weight_matrix.detach().cpu().numpy()*bs
        normalized_epoch_weights=normalized_weight_sum/n;off_epoch=normalized_epoch_weights[~np.eye(7,dtype=bool)]
        row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_adv=sums[2]/n if setting['adv'] else 0,L_mcc=sums[3]/n if setting['mcc'] else 0,L_smcc=sums[4]/n if setting['mcc'] else 0,L_nsmcc=sums[4]/n if setting.get('normalized_selective',False) else sums[3]/n,
          domain_accuracy=domain_ok/domain_n if domain_n else None,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,
          target_mean_soft_distribution=(soft_sum/n).tolist(),class_confusion_matrix=(conf_sum/n).tolist(),raw_confusion_matrix=(conf_sum/n).tolist(),weighted_confusion_matrix=(weighted_sum/n).tolist(),w_ij_matrix=(weight_sum/n).tolist(),raw_w_ij_matrix=(weight_sum/n).tolist(),normalized_w_hat_ij_matrix=normalized_epoch_weights.tolist(),normalized_weight_mean=float(off_epoch.mean()),normalized_weight_min=float(off_epoch.min()),normalized_weight_max=float(off_epoch.max()),off_diagonal_confusion_score=sums[3]/n,grl_coefficient=coefficient,finite=True)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    torch.save({'model':model.state_dict(),'discriminator':disc.state_dict() if setting['adv'] else None,'config':config},out/'final.pth')
    result=evaluate(model,eval_loader,device,7);result.update({k:history[-1][k] for k in ('source_accuracy','shifted_source_accuracy','L_adv','domain_accuracy','L_mcc','L_smcc','L_nsmcc','target_mean_soft_distribution','raw_confusion_matrix','weighted_confusion_matrix','w_ij_matrix','raw_w_ij_matrix','normalized_w_hat_ij_matrix','normalized_weight_mean','normalized_weight_min','normalized_weight_max')})
    if setting['adv']:result['feature_gap']={'before':initial_gap,'after':feature_gap(model,shifted_diag,target_diag,np.arange(gap_n),device)}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',choices=SETTINGS,required=True);p.add_argument('--seed',type=int,default=1341);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
