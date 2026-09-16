"""Houston Adaptive SceneShift / MCC / Semantic Relation Preservation ablation."""
import argparse,csv,json,random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.sceneshift_net_dcrn import SceneShiftNetDCRN
from sceneshiftnet_dcrn_train import PatchDataset,evaluate,file_sha256,sample_source
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from train_sceneshiftnet_residual_mcc import Discriminator,mcc_loss

HERE=Path(__file__).resolve().parent
OUT=HERE/'runs_sceneshiftnet_dcrn/houston_raw/srp'
METHODS={
 'I0':dict(name='Adaptive + Original MCC',mcc=True,srp=False),
 'I1':dict(name='Adaptive + SRP',mcc=False,srp=True),
 'I2':dict(name='Adaptive + Original MCC + SRP',mcc=True,srp=True),
}


def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False


def relation_matrix(prototypes):
 z=F.normalize(prototypes,p=2,dim=1);return 1-z@z.T


def srp_loss(shifted_features,target_features,target_probabilities,labels,rho=.8):
 classes=labels.unique(sorted=True)
 shifted_norm=F.normalize(shifted_features,p=2,dim=1)
 target_norm=F.normalize(target_features,p=2,dim=1)
 source_proto=torch.stack([shifted_norm[labels==c].mean(0) for c in classes])
 source_proto=F.normalize(source_proto,p=2,dim=1)
 target_proto=(target_probabilities.T@target_norm)/(target_probabilities.sum(0)[:,None]+1e-5)
 target_proto=F.normalize(target_proto,p=2,dim=1)
 source_relation=relation_matrix(source_proto)
 target_relation=relation_matrix(target_proto)
 pair=torch.triu(torch.ones(len(classes),len(classes),dtype=torch.bool,device=labels.device),diagonal=1)
 target_selected=target_relation[classes[:,None],classes[None,:]][pair]
 source_selected=source_relation[pair]
 violation=F.relu(rho*source_selected-target_selected)
 loss=violation.square().mean() if len(violation) else shifted_features.sum()*0
 full_source=torch.full((7,7),float('nan'),device=labels.device);full_source[classes[:,None],classes[None,:]]=source_relation
 return loss,full_source,target_relation,int((violation>0).sum()),len(violation),violation.mean() if len(violation) else loss


@torch.no_grad()
def offline_diagnostic(model,loader,device):
 model.eval();features=[];probs=[];truth=[]
 for x,y in loader:
  f,z=model(x.to(device));features.append(f.cpu().numpy());probs.append(z.softmax(1).cpu().numpy());truth.append(y.numpy())
 f,p,y=np.concatenate(features),np.concatenate(probs),np.concatenate(truth);pred=p.argmax(1);cm=np.zeros((7,7),dtype=np.int64);np.add.at(cm,(y,pred),1);norm=cm/cm.sum(1,keepdims=True)
 centroids=np.stack([f[y==c].mean(0) for c in range(7)]);radii=np.array([np.linalg.norm(f[y==c]-centroids[c],axis=1).mean() for c in range(7)])
 geometry={}
 for a,b in ((1,2),(2,6),(6,5),(6,4)):
  x,z=centroids[a-1],centroids[b-1];cos=float(x@z/(np.linalg.norm(x)*np.linalg.norm(z)+1e-12));eu=float(np.linalg.norm(x-z));geometry[f'C{a}-C{b}']={'cosine':cos,'euclidean':eu,'inter_intra_ratio':eu/((radii[a-1]+radii[b-1])/2+1e-12)}
 return {'hard_confusion_matrix':cm.tolist(),'normalized_confusion_matrix':norm.tolist(),'per_class_mean_soft_probability':np.stack([p[y==c].mean(0) for c in range(7)]).tolist(),'geometry':geometry,'C6_to_C5':float(norm[5,4]),'C6_to_C4':float(norm[5,3])}


def run(args):
 spec=METHODS[args.method];seed_all(args.seed);suffix=f'seed_{args.seed}' if args.epochs==100 else f'seed_{args.seed}_smoke_{args.epochs}ep';out=OUT/args.method/suffix;out.mkdir(parents=True,exist_ok=False)
 source,sg,target,tg,paths=load_cubes('none');source=source.astype('float32');target=target.astype('float32');centers,labels=sample_source(sg,7,180,np.random.RandomState(args.seed));tc=np.argwhere(tg>0);target_labels=tg[tc[:,0],tc[:,1]].astype('int64')-1
 generator=torch.Generator().manual_seed(args.seed);sl=DataLoader(PatchDataset(source,centers,7,labels),32,shuffle=True,drop_last=True,generator=generator);tl=DataLoader(PatchDataset(target,tc,7),32,shuffle=True,drop_last=True,generator=generator);el=DataLoader(PatchDataset(target,tc,7,target_labels),32,shuffle=False)
 sm,ss=source.reshape(-1,48).mean(0),source.reshape(-1,48).std(0);tm,ts=target.reshape(-1,48).mean(0),target.reshape(-1,48).std(0);d=(np.abs(tm-sm)+np.abs(ts-ss))/(ss+1e-5);alpha=.4+.5*(d-d.min())/(d.max()-d.min()+1e-5)
 config=dict(method=args.method,method_name=spec['name'],seed=args.seed,epochs=args.epochs,batch_size=32,patch_size=7,source_per_class=180,optimizer='Adam',lr=.001,weight_decay=0,normalization='none',use_ilda=False,deterministic=True,forward_order='source -> shifted-source -> target, once each',scene_shift='adaptive',alpha_min=.4,alpha_max=.9,temperature=2.5,original_mcc=spec['mcc'],lambda_mcc=.1 if spec['mcc'] else 0,srp=spec['srp'],rho=.8 if spec['srp'] else None,lambda_rel=.05 if spec['srp'] else 0,target_hard_pseudo_labels=False,target_gt_usage='support mask and final offline evaluation only',loss='L_src + .5 L_shift'+(' + .1 L_mcc' if spec['mcc'] else '')+(' + .05 L_rel' if spec['srp'] else ''),code_sha256=file_sha256(__file__),model_sha256=file_sha256(HERE/'models/sceneshift_net_dcrn.py'),data_hash={str(p):file_sha256(p) for p in paths})
 (out/'config.json').write_text(json.dumps(config,indent=2));np.savez(out/'indices.npz',source=centers,source_labels=labels,target=tc);(out/'band_statistics.json').write_text(json.dumps({'d_b':d.tolist(),'alpha_b':alpha.tolist()},indent=2))
 assert torch.cuda.is_available();device='cuda';model=SceneShiftNetDCRN(48,7,7).to(device);unused_discriminator=Discriminator().to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001);sm,ss,tm,ts,alpha=[torch.tensor(v,device=device)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)];history=[]
 for epoch in range(1,args.epochs+1):
  model.train();target_it=iter(tl);sums=np.zeros(4);correct=np.zeros(2,dtype=int);n=0;soft_sum=np.zeros(7);source_rel_sum=np.zeros((7,7));source_rel_count=np.zeros((7,7));target_rel_sum=np.zeros((7,7));active=candidates=0;violation_sum=0
  for x,y in sl:
   try:z=next(target_it)
   except StopIteration:target_it=iter(tl);z=next(target_it)
   x,y,z=x.to(device),y.to(device),z.to(device);fs,ls=model(x);shifted=(x-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm;fss,lss=model(shifted);ft,lt=model(z);raw_mcc,prob,conf=mcc_loss(lt);rel,srel,trel,na,npair,vmean=srp_loss(fss,ft,prob,y);src_loss=F.cross_entropy(ls,y);shift_loss=F.cross_entropy(lss,y);loss=src_loss+.5*shift_loss+(.1*raw_mcc if spec['mcc'] else 0)+(.05*rel if spec['srp'] else 0)
   if not torch.isfinite(loss) or not torch.isfinite(srel[~torch.isnan(srel)]).all() or not torch.isfinite(trel).all():raise RuntimeError(f'nonfinite epoch {epoch}')
   optimizer.zero_grad();loss.backward();optimizer.step();bs=len(y);n+=bs;sums+=np.array([src_loss.item(),shift_loss.item(),raw_mcc.item(),rel.item()])*bs;correct+=[int((ls.argmax(1)==y).sum()),int((lss.argmax(1)==y).sum())];soft_sum+=prob.detach().sum(0).cpu().numpy();sr=srel.detach().cpu().numpy();valid=~np.isnan(sr);source_rel_sum[valid]+=sr[valid];source_rel_count[valid]+=1;target_rel_sum+=trel.detach().cpu().numpy();active+=na;candidates+=npair;violation_sum+=float(vmean)*npair
  source_epoch=np.divide(source_rel_sum,source_rel_count,out=np.zeros_like(source_rel_sum),where=source_rel_count>0);target_epoch=target_rel_sum/len(sl);row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_mcc=sums[2]/n if spec['mcc'] else 0,L_rel=sums[3]/n if spec['srp'] else 0,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,shifted_source_relation_matrix=source_epoch.tolist(),target_soft_relation_matrix=target_epoch.tolist(),active_relation_pairs=active,candidate_relation_pairs=candidates,active_relation_ratio=active/candidates if candidates else 0,mean_relation_violation=violation_sum/candidates if candidates else 0,target_mean_soft_distribution=(soft_sum/n).tolist(),finite=True)
  history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
 torch.save({'model':model.state_dict(),'config':config,'seed':args.seed},out/'final.pth');metrics=evaluate(model,el,device,7);metrics.update(seed=args.seed,method=args.method,method_name=spec['name'],source_accuracy=history[-1]['source_accuracy'],shifted_source_accuracy=history[-1]['shifted_source_accuracy'],L_mcc=history[-1]['L_mcc'],L_rel=history[-1]['L_rel'],active_relation_ratio=history[-1]['active_relation_ratio']);metrics['offline_diagnostic']=offline_diagnostic(model,el,device);(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
 with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 print(json.dumps(metrics),flush=True)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--seed',type=int,default=1341);p.add_argument('--epochs',type=int,default=100);run(p.parse_args())
