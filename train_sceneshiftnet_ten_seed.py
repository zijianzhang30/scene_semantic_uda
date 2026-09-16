"""Unified deterministic Houston A/B/C ten-seed stability runner."""
import argparse, csv, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_dcrn import SceneShiftNetDCRN
from sceneshiftnet_dcrn_train import PatchDataset, evaluate, file_sha256, sample_source
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from train_sceneshiftnet_residual_mcc import Discriminator, mcc_loss

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs_sceneshiftnet_dcrn/houston_raw/ten_seed_final'
METHODS={
    'A': dict(name='Original SceneShift',adaptive=False,target_ce=True,mcc=False),
    'B': dict(name='Adaptive SceneShift',adaptive=True,target_ce=True,mcc=False),
    'C': dict(name='Adaptive SceneShift + Original MCC',adaptive=True,target_ce=False,mcc=True),
}


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def run(args):
    spec=METHODS[args.method];seed_all(args.seed)
    suffix=f'seed_{args.seed}' if args.epochs==100 else f'seed_{args.seed}_smoke_{args.epochs}ep'
    out=OUT/args.method/suffix;out.mkdir(parents=True,exist_ok=False)
    source,sg,target,tg,paths=load_cubes('none');source=source.astype('float32');target=target.astype('float32')
    centers,labels=sample_source(sg,7,180,np.random.RandomState(args.seed));tc=np.argwhere(tg>0)
    # A fresh generator with the same seed in every process fixes identical A/B/C order.
    generator=torch.Generator().manual_seed(args.seed)
    sl=DataLoader(PatchDataset(source,centers,7,labels),32,shuffle=True,drop_last=True,generator=generator)
    tl=DataLoader(PatchDataset(target,tc,7),32,shuffle=True,drop_last=True,generator=generator)
    el=DataLoader(PatchDataset(target,tc,7,tg[tc[:,0],tc[:,1]].astype('int64')-1),32,shuffle=False)
    sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0)
    discrepancy=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5)
    adaptive=.4+.5*(discrepancy-discrepancy.min())/(discrepancy.max()-discrepancy.min()+1e-5)
    alpha=adaptive if spec['adaptive'] else np.full(48,.8,dtype='float32')
    config=dict(method=args.method,method_name=spec['name'],seed=args.seed,epochs=args.epochs,batch_size=32,patch_size=7,
      source_per_class=180,optimizer='Adam',lr=.001,weight_decay=0,normalization='none',use_ilda=False,
      deterministic=True,forward_order='source -> shifted-source -> target, once each',scene_shift='adaptive' if spec['adaptive'] else 'fixed',
      alpha=None if spec['adaptive'] else .8,alpha_min=.4 if spec['adaptive'] else None,alpha_max=.9 if spec['adaptive'] else None,
      target_pseudo_label_ce=spec['target_ce'],pseudo_threshold=.9 if spec['target_ce'] else None,pseudo_warmup_epochs=10 if spec['target_ce'] else None,
      original_mcc=spec['mcc'],temperature=2.5 if spec['mcc'] else None,lambda_mcc=.1 if spec['mcc'] else 0,
      loss='L_src + .5 L_shift + .5 L_target' if spec['target_ce'] else 'L_src + .5 L_shift + .1 L_MCC',
      target_gt_usage='support mask and final evaluation only',code_sha256=file_sha256(__file__),
      model_sha256=file_sha256(HERE/'models/sceneshift_net_dcrn.py'),mcc_dependency_sha256=file_sha256(HERE/'train_sceneshiftnet_residual_mcc.py'),
      data_hash={str(p):file_sha256(p) for p in paths})
    (out/'config.json').write_text(json.dumps(config,indent=2));np.savez(out/'indices.npz',source=centers,source_labels=labels,target=tc)
    (out/'band_statistics.json').write_text(json.dumps({'d_b':discrepancy.tolist(),'alpha_b':alpha.tolist()},indent=2))
    assert torch.cuda.is_available();device='cuda';model=SceneShiftNetDCRN(48,7,7).to(device)
    # Match the F3 initialization stream for all methods; this module is never forwarded or optimized.
    unused_discriminator=Discriminator().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    sm,ss,tm,ts,alpha=[torch.tensor(v,device=device)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)]
    history=[]
    for epoch in range(1,args.epochs+1):
        model.train();target_it=iter(tl);sums=np.zeros(4);correct=np.zeros(2,dtype=int);n=seen=selected=0;hist=np.zeros(7,dtype=int);soft_sum=np.zeros(7)
        for x,y in sl:
            try:z=next(target_it)
            except StopIteration:target_it=iter(tl);z=next(target_it)
            x,y,z=x.to(device),y.to(device),z.to(device)
            fs,ls=model(x)
            shifted=(x-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
            fss,lss=model(shifted);ft,lt=model(z)
            raw_mcc,prob,conf=mcc_loss(lt)
            confidence,pseudo=lt.softmax(1).max(1);mask=(confidence>.9)&(epoch>10)
            target_loss=F.cross_entropy(lt[mask],pseudo[mask]) if spec['target_ce'] and mask.any() else lt.sum()*0
            src_loss=F.cross_entropy(ls,y);shift_loss=F.cross_entropy(lss,y)
            loss=src_loss+.5*shift_loss+(.5*target_loss if spec['target_ce'] else .1*raw_mcc)
            if not torch.isfinite(loss):raise RuntimeError(f'nonfinite loss epoch {epoch}')
            optimizer.zero_grad();loss.backward();optimizer.step()
            bs=len(y);n+=bs;seen+=len(z);selected+=int(mask.sum()) if spec['target_ce'] else 0
            sums+=np.array([src_loss.item(),shift_loss.item(),target_loss.item(),raw_mcc.item()])*bs
            correct+=[int((ls.argmax(1)==y).sum()),int((lss.argmax(1)==y).sum())]
            if spec['target_ce']:hist+=np.bincount(pseudo[mask].detach().cpu().numpy(),minlength=7)
            soft_sum+=prob.detach().sum(0).cpu().numpy()
        row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_target=sums[2]/n if spec['target_ce'] else 0,
          L_mcc=sums[3]/n if spec['mcc'] else 0,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,
          pseudo_label_coverage=selected/seen if spec['target_ce'] else 0,pseudo_label_histogram=hist.tolist(),
          target_mean_soft_distribution=(soft_sum/n).tolist(),finite=True)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    torch.save({'model':model.state_dict(),'config':config,'seed':args.seed},out/'final.pth')
    result=evaluate(model,el,device,7);result.update(seed=args.seed,method=args.method,method_name=spec['name'],source_accuracy=history[-1]['source_accuracy'],shifted_source_accuracy=history[-1]['shifted_source_accuracy'],pseudo_label_coverage=history[-1]['pseudo_label_coverage'])
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,choices=[1341,2024,3407,1174,1622,42,340,777,1024,2026],required=True);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
