"""Houston DAMamba backbone generalization: M0/M1/M2, three matched seeds."""
import argparse,csv,json,random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_damamba import SceneShiftNetDAMamba
from sceneshiftnet_dcrn_train import PatchDataset,evaluate,file_sha256,sample_source
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from train_sceneshiftnet_residual_mcc import mcc_loss

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs_sceneshiftnet_damamba/houston_raw/three_seed'
METHODS={
 'M0':dict(name='DAMamba baseline',shift=False,mcc=False),
 'M1':dict(name='DAMamba + Adaptive SceneShift',shift=True,mcc=False),
 'M2':dict(name='DAMamba + Adaptive SceneShift + Original MCC',shift=True,mcc=True),
}


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def optimizer_for(model):
 backbone=list(model.encoder.backbone.parameters())
 head=list(model.encoder.channel_attention.parameters())+list(model.encoder.spatial_attention.parameters())+list(model.encoder.bottleneck.parameters())+list(model.classifier.parameters())
 optimizer=torch.optim.SGD([{'params':backbone,'lr':.1},{'params':head,'lr':1.0}],lr=.01,momentum=.9,weight_decay=5e-4)
 scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:.01*(1+.0003*step)**(-.75))
 expected={id(p) for p in model.parameters() if p.requires_grad};actual={id(p) for g in optimizer.param_groups for p in g['params']};assert expected==actual
 return optimizer,scheduler


def run(args):
 spec=METHODS[args.method];seed_all(args.seed);suffix=f'seed_{args.seed}' if args.epochs==100 else f'seed_{args.seed}_smoke_{args.epochs}ep';out=OUT/args.method/suffix;out.mkdir(parents=True,exist_ok=False)
 source,sg,target,tg,paths=load_cubes('none');source=source.astype('float32');target=target.astype('float32');centers,labels=sample_source(sg,7,180,np.random.RandomState(args.seed));tc=np.argwhere(tg>0);target_labels=tg[tc[:,0],tc[:,1]].astype('int64')-1
 generator=torch.Generator().manual_seed(args.seed);sl=DataLoader(PatchDataset(source,centers,7,labels),32,shuffle=True,drop_last=True,generator=generator);tl=DataLoader(PatchDataset(target,tc,7),32,shuffle=True,drop_last=True,generator=generator);el=DataLoader(PatchDataset(target,tc,7,target_labels),32,shuffle=False)
 sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0);d=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5);alpha=(.4+.5*(d-d.min())/(d.max()-d.min()+1e-5)).astype(np.float32)
 config={
  'method':args.method,'method_name':spec['name'],'seed':args.seed,'epochs':args.epochs,
  'external_patch_size':7,'internal_damamba_grid':12,
  'patch_adapter':'reflect pad external 7x7 by (left=2,right=3,top=2,bottom=3); no extra cube pixels',
  'source_per_class':180,'batch_size':32,'normalization':'none','use_ilda':False,'deterministic':True,
  'backbone':'official DAMamba MambaFeature + official CA/SA + 4608->256 bottleneck',
  'feature_dim':256,'classifier':'Linear(256,7)','damamba_adaptation_modules':False,
  'optimizer':'SGD','base_lr':.01,'momentum':.9,'weight_decay':5e-4,
  'differential_lr':'backbone .1x; CA/SA/bottleneck/classifier 1x',
  'scheduler':'LambdaLR .01*(1+.0003*step)^(-.75)',
  'optimizer_note':'stable prior DAMamba Houston form retained; requested batch32/100 epochs retained',
  'forward_order':'source -> auxiliary-source -> target, once each; M0 auxiliary input is an unshifted source duplicate, M1/M2 use adaptive-shifted source',
  'adaptive_scene_shift':spec['shift'],'alpha_min':.4,'alpha_max':.9,
  'original_mcc':spec['mcc'],'temperature':2.5,
  'lambda_mcc':.1 if spec['mcc'] else 0,'lambda_shift':.5 if spec['shift'] else 0,
  'target_hard_pseudo_labels':False,'target_gt_usage':'support mask and final evaluation only',
  'loss':'L_src'+(' + .5 L_shift' if spec['shift'] else '')+(' + .1 L_mcc' if spec['mcc'] else ''),
  'code_sha256':file_sha256(__file__),'model_sha256':file_sha256(HERE/'models/sceneshift_net_damamba.py'),
  'damamba_source_sha256':file_sha256('/home/zhangzj26/DAMamba/DAMamba_basenet.py'),
  'data_hash':{str(p):file_sha256(p) for p in paths},
 }
 (out/'config.json').write_text(json.dumps(config,indent=2));np.savez(out/'indices.npz',source=centers,source_labels=labels,target=tc);(out/'band_statistics.json').write_text(json.dumps({'d_b':d.tolist(),'alpha_b':alpha.tolist()},indent=2))
 assert torch.cuda.is_available();device='cuda';model=SceneShiftNetDAMamba(48,7).to(device);optimizer,scheduler=optimizer_for(model);sm,ss,tm,ts,alpha=[torch.tensor(v,device=device)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)];history=[]
 for epoch in range(1,args.epochs+1):
  model.train();target_it=iter(tl);sums=np.zeros(3);correct=np.zeros(2,dtype=int);n=0;soft_sum=np.zeros(7)
  for x,y in sl:
   try:z=next(target_it)
   except StopIteration:target_it=iter(tl);z=next(target_it)
   x,y,z=x.to(device),y.to(device),z.to(device);fs,ls=model(x);shifted=(x-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm if spec['shift'] else x;fss,lss=model(shifted);ft,lt=model(z);raw_mcc,prob,conf=mcc_loss(lt);src=F.cross_entropy(ls,y);shift_loss=F.cross_entropy(lss,y);loss=src+(.5*shift_loss if spec['shift'] else 0)+(.1*raw_mcc if spec['mcc'] else 0)
   if not torch.isfinite(loss) or not torch.isfinite(fs).all() or not torch.isfinite(fss).all() or not torch.isfinite(ft).all():raise RuntimeError(f'nonfinite epoch {epoch}')
   optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step();bs=len(y);n+=bs;sums+=np.array([src.item(),shift_loss.item(),raw_mcc.item()])*bs;correct+=[int((ls.argmax(1)==y).sum()),int((lss.argmax(1)==y).sum())];soft_sum+=prob.detach().sum(0).cpu().numpy()
  row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_mcc=sums[2]/n if spec['mcc'] else 0,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,target_mean_soft_class_distribution=(soft_sum/n).tolist(),backbone_lr=optimizer.param_groups[0]['lr'],head_lr=optimizer.param_groups[1]['lr'],finite=True);history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 torch.save({'model':model.state_dict(),'config':config,'seed':args.seed},out/'final.pth');metrics=evaluate(model,el,device,7);metrics.update(seed=args.seed,method=args.method,method_name=spec['name'],source_accuracy=history[-1]['source_accuracy'],shifted_source_accuracy=history[-1]['shifted_source_accuracy'],L_mcc=history[-1]['L_mcc'],target_mean_soft_class_distribution=history[-1]['target_mean_soft_class_distribution']);(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
 with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 print(json.dumps(metrics),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,choices=[1341,2024,3407],required=True);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
