"""Houston Adaptive SceneShift with scheduled Original MCC and gradient diagnostics."""
import argparse, csv, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_dcrn import SceneShiftNetDCRN
from sceneshiftnet_dcrn_train import PatchDataset, evaluate, file_sha256, sample_source
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from train_sceneshiftnet_residual_mcc import Discriminator, mcc_loss

HERE = Path(__file__).resolve().parent
OUT = HERE / 'runs_sceneshiftnet_dcrn/houston_raw/scheduled_mcc_gradient'
METHODS = {
    'K0': 'fixed MCC 0.1',
    'K1': 'delayed mild MCC',
    'K2': 'delayed full MCC',
    'K3': 'cosine ramp MCC',
}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def lambda_mcc(method, epoch):
    if method == 'K0': return .1
    if method in ('K1', 'K2'):
        top = .05 if method == 'K1' else .1
        if epoch <= 30: return 0.
        if epoch <= 70: return top * (epoch - 31) / 39.
        return top
    if epoch <= 20: return 0.
    progress = (epoch - 21) / 79.
    return .1 * .5 * (1. - math.cos(math.pi * progress))


def grad_diagnostic(model, dx, dy, dz, stats):
    was_training = model.training
    model.eval()
    params = [p for p in model.encoder.parameters() if p.requires_grad]
    sm, ss, tm, ts, alpha = stats
    fs, ls = model(dx)
    shifted = (dx-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
    fss, lss = model(shifted)
    sup = F.cross_entropy(ls, dy) + .5 * F.cross_entropy(lss, dy)
    gs = torch.autograd.grad(sup, params, allow_unused=True)
    ft, lt = model(dz)
    mcc, _, _ = mcc_loss(lt)
    gm = torch.autograd.grad(mcc, params, allow_unused=True)
    dot = sum((a.detach()*b.detach()).sum() for a,b in zip(gs,gm) if a is not None and b is not None)
    ns2 = sum(a.detach().square().sum() for a in gs if a is not None)
    nm2 = sum(b.detach().square().sum() for b in gm if b is not None)
    ns, nm = ns2.sqrt(), nm2.sqrt()
    cosine = dot/(ns*nm+1e-12)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return dict(grad_cosine=float(cosine), grad_sup_norm=float(ns), grad_mcc_norm=float(nm),
                grad_norm_ratio=float(nm/(ns+1e-12)), diagnostic_L_sup=float(sup), diagnostic_L_mcc=float(mcc))


def run(args):
    seed_all(args.seed)
    suffix = f'seed_{args.seed}' if args.epochs == 100 else f'seed_{args.seed}_smoke_{args.epochs}ep'
    out = OUT / args.method / suffix; out.mkdir(parents=True, exist_ok=False)
    source, sg, target, tg, paths = load_cubes('none'); source=source.astype('float32'); target=target.astype('float32')
    centers, labels = sample_source(sg, 7, 180, np.random.RandomState(args.seed)); tc=np.argwhere(tg>0)
    generator=torch.Generator().manual_seed(args.seed)
    sl=DataLoader(PatchDataset(source,centers,7,labels),32,shuffle=True,drop_last=True,generator=generator)
    tl=DataLoader(PatchDataset(target,tc,7),32,shuffle=True,drop_last=True,generator=generator)
    el=DataLoader(PatchDataset(target,tc,7,tg[tc[:,0],tc[:,1]].astype('int64')-1),32,shuffle=False)
    # Fixed diagnostic data, assembled without a DataLoader iterator so no
    # global RNG is consumed before model initialization.
    diag_s=PatchDataset(source,centers,7,labels);diag_t=PatchDataset(target,tc,7)
    diag_s_items=[diag_s[i] for i in range(32)];dx=torch.stack([v[0] for v in diag_s_items]);dy=torch.stack([v[1] for v in diag_s_items])
    dz=torch.stack([diag_t[i] for i in range(32)])
    sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0)
    discrepancy=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5)
    alpha=(.4+.5*(discrepancy-discrepancy.min())/(discrepancy.max()-discrepancy.min()+1e-5)).astype('float32')
    config=dict(method=args.method,method_name=METHODS[args.method],seed=args.seed,epochs=args.epochs,batch_size=32,patch_size=7,
      source_per_class=180,optimizer='Adam',lr=.001,weight_decay=0,normalization='none',use_ilda=False,deterministic=True,
      forward_order='source -> shifted-source -> target, once each',scene_shift='adaptive',alpha_min=.4,alpha_max=.9,
      original_mcc=True,temperature=2.5,lambda_schedule=args.method,loss='L_src + .5 L_shift + lambda_mcc(epoch) L_MCC',
      gradient_diagnostic='every 5 epochs; fixed batch; eval mode; encoder only; independent autograd.grad; no step',
      target_gt_usage='support mask and final evaluation only',target_hard_pseudo_labels=False,
      code_sha256=file_sha256(__file__),model_sha256=file_sha256(HERE/'models/sceneshift_net_dcrn.py'),
      mcc_dependency_sha256=file_sha256(HERE/'train_sceneshiftnet_residual_mcc.py'),data_hash={str(p):file_sha256(p) for p in paths})
    (out/'config.json').write_text(json.dumps(config,indent=2));np.savez(out/'indices.npz',source=centers,source_labels=labels,target=tc)
    (out/'band_statistics.json').write_text(json.dumps({'d_b':discrepancy.tolist(),'alpha_b':alpha.tolist()},indent=2))
    assert torch.cuda.is_available();device='cuda';model=SceneShiftNetDCRN(48,7,7).to(device)
    unused_discriminator=Discriminator().to(device)  # preserve the verified unified-runner initialization stream
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    tensors=[torch.tensor(v,device=device)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)]
    sm,ss,tm,ts,alpha=tensors;dx,dy,dz=dx.to(device),dy.to(device),dz.to(device);history=[];diagnostics=[]
    for epoch in range(1,args.epochs+1):
        model.train(); target_it=iter(tl); sums=np.zeros(3); correct=np.zeros(2,dtype=int); n=0; soft_sum=np.zeros(7); lam=lambda_mcc(args.method,epoch)
        for x,y in sl:
            try:z=next(target_it)
            except StopIteration:target_it=iter(tl);z=next(target_it)
            x,y,z=x.to(device),y.to(device),z.to(device);fs,ls=model(x)
            shifted=(x-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
            fss,lss=model(shifted);ft,lt=model(z);raw_mcc,prob,_=mcc_loss(lt)
            src=F.cross_entropy(ls,y);shift_loss=F.cross_entropy(lss,y);sup=src+.5*shift_loss;loss=sup+lam*raw_mcc
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss epoch {epoch}')
            optimizer.zero_grad();loss.backward();optimizer.step();bs=len(y);n+=bs
            sums+=np.array([src.item(),shift_loss.item(),raw_mcc.item()])*bs
            correct+=[int((ls.argmax(1)==y).sum()),int((lss.argmax(1)==y).sum())];soft_sum+=prob.detach().sum(0).cpu().numpy()
        row=dict(epoch=epoch,lambda_mcc=lam,L_src=sums[0]/n,L_shift=sums[1]/n,L_mcc=sums[2]/n,
          source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,
          target_mean_soft_distribution=(soft_sum/n).tolist(),finite=True)
        if epoch%5==0:
            diag={'epoch':epoch,**grad_diagnostic(model,dx,dy,dz,(sm,ss,tm,ts,alpha))};diagnostics.append(diag);row.update(diag)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));(out/'gradient_diagnostics.json').write_text(json.dumps(diagnostics,indent=2));print(json.dumps(row),flush=True)
    torch.save({'model':model.state_dict(),'config':config,'seed':args.seed},out/'final.pth');result=evaluate(model,el,device,7)
    cos=np.array([d['grad_cosine'] for d in diagnostics]);epochs=np.array([d['epoch'] for d in diagnostics])
    grad_summary={'negative_cosine_ratio':float((cos<0).mean()),'mean_cosine':float(cos.mean()),
      'early_mean_cosine':float(cos[epochs<=30].mean()),'middle_mean_cosine':float(cos[(epochs>30)&(epochs<=70)].mean()),'late_mean_cosine':float(cos[epochs>70].mean())}
    result.update(seed=args.seed,method=args.method,method_name=METHODS[args.method],source_accuracy=history[-1]['source_accuracy'],shifted_source_accuracy=history[-1]['shifted_source_accuracy'],L_mcc=history[-1]['L_mcc'],target_mean_soft_distribution=history[-1]['target_mean_soft_distribution'],gradient_summary=grad_summary)
    (out/'metrics.json').write_text(json.dumps(result,indent=2));(out/'gradient_summary.json').write_text(json.dumps(grad_summary,indent=2))
    with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=sorted({k for r in history for k in r}));w.writeheader();w.writerows(history)
    with (out/'gradient_diagnostics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=diagnostics[0]);w.writeheader();w.writerows(diagnostics)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,choices=[1341,2024,3407],required=True);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
