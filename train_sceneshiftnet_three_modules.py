"""Houston raw ablations; no target labels in training or reliability selection."""
import argparse, json, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sceneshiftnet_dcrn_train import PatchDataset, sample_source, set_seed, evaluate, file_sha256
from train_sceneshiftnet_dcrn_houston_raw import load_cubes
from models.sceneshift_net_dcrn import SceneShiftNetDCRN
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs_sceneshiftnet_dcrn/houston_raw'
class IndexedTarget(PatchDataset):
    def __getitem__(self,i): return super().__getitem__(i),i

def reliable(features, shifted, labels, logits, enabled):
    with torch.no_grad():
        classes=labels.unique(sorted=True)
        prototypes=torch.stack([shifted[labels==c].mean(0) for c in classes])
        sim=F.normalize(features,dim=1)@F.normalize(prototypes,dim=1).T
        top=sim.topk(min(2,len(classes)),dim=1)
        proto=classes[top.indices[:,0]]
        conf,pseudo=logits.softmax(1).max(1)
        qualified=(conf>.9) & enabled
        agreement=pseudo==proto
        mask=qualified & agreement
        gap=top.values[:,0]-top.values[:,1] if len(classes)>1 else torch.zeros_like(conf)
    return qualified,agreement,mask,pseudo,proto,top.values[:,0],gap

def target_objective(logits,pseudo,mask,pi,correction):
    if not mask.any(): return logits.sum()*0,pi
    if correction:
        with torch.no_grad():
            distribution=torch.bincount(pseudo[mask],minlength=7).float()/mask.sum()
            pi=.9*pi+.1*distribution
        weights=(pi.mean()/(pi+1e-5)).sqrt().clamp(.5,2)[pseudo[mask]]
        loss=(weights*F.cross_entropy(logits[mask],pseudo[mask],reduction='none')).sum()/(weights.sum()+1e-5)
    else: loss=F.cross_entropy(logits[mask],pseudo[mask])
    return loss,pi

def summary():
    rows=[]
    for tag,folder in [('A','sceneshift'),('B','adaptive_shift'),('C','adaptive_reliable'),('D','adaptive_reliable_confusion')]:
        p=OUT/folder/'seed_1341/metrics.json'
        if p.exists(): rows.append({'Method':tag,**json.loads(p.read_text())})
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/'three_module_ablation.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def run(a):
    assert a.adaptive_shift and (not a.confusion_correction or a.reliability_target)
    tag='adaptive_reliable_confusion' if a.confusion_correction else 'adaptive_reliable' if a.reliability_target else 'adaptive_shift'
    out=OUT/tag/f'seed_{a.seed}'
    out.mkdir(parents=True,exist_ok=False)
    set_seed(a.seed)
    s,sg,t,tg,paths=load_cubes('none');s=s.astype('float32');t=t.astype('float32')
    centers,y=sample_source(sg,7,180,np.random.RandomState(a.seed))
    tc=np.argwhere(tg>0)
    # Keep original loader creation/iteration order and random draws.
    sl=DataLoader(PatchDataset(s,centers,7,y),32,shuffle=True,drop_last=True)
    tl=DataLoader(IndexedTarget(t,tc,7),32,shuffle=True,drop_last=True)
    el=DataLoader(PatchDataset(t,tc,7,tg[tc[:,0],tc[:,1]].astype('int64')-1),32,shuffle=False)
    sm,ss=s.reshape(-1,48).mean(0),s.reshape(-1,48).std(0)
    tm,ts=t.reshape(-1,48).mean(0),t.reshape(-1,48).std(0)
    d=(abs(tm-sm)+abs(ts-ss))/(ss+1e-5);alpha=.4+.5*(d-d.min())/(d.max()-d.min()+1e-5)
    config=dict(vars(a),normalization='none',use_ilda=False,optimizer='Adam',lr=.001,weight_decay=0,batch_size=32,epochs=100,source_per_class=180,patch_size=7,warmup=10,threshold=.9,alpha_min=.4,alpha_max=.9,epsilon=1e-5,beta=.9,gamma=.5,ema_initialization='uniform',ema_update='each nonempty reliable batch, before weighted loss',loss='L_src + .5 L_shift + .5 L_target',code_sha256=file_sha256(__file__),model_sha256=file_sha256(ROOT/'models/sceneshift_net_dcrn.py'),trainer_dependency_sha256=file_sha256(ROOT/'sceneshiftnet_dcrn_train.py'),data_hash={str(p):file_sha256(p) for p in paths})
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'band_statistics.json').write_text(json.dumps(dict(d_b=d.tolist(),alpha_b=alpha.tolist(),mean=float(alpha.mean()),min=float(alpha.min()),max=float(alpha.max())),indent=2))
    np.savez(out/'indices.npz',source=centers,source_labels=y,target=tc)
    assert torch.cuda.is_available(),'CUDA required'
    dev='cuda';model=SceneShiftNetDCRN(48,7,7).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.001)
    sm,ss,tm,ts,alpha=[torch.tensor(v,device=dev)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)]
    pi=torch.ones(7,device=dev)/7;history=[]
    for epoch in range(1,101):
        model.train();it=iter(tl);sums=np.zeros(3);correct=np.zeros(2);seen=n=conf_n=agree_n=rel_n=0
        hist=np.zeros((3,7),dtype=int);cos_sum=gap_sum=0.;records=[]
        for x,yb in sl:
            try:z,ids=next(it)
            except StopIteration:it=iter(tl);z,ids=next(it)
            x,yb,z=x.to(dev),yb.to(dev),z.to(dev)
            fs,ls=model(x)
            shifted=(x-sm)/(ss+1e-5)*(alpha*ts+(1-alpha)*ss)+alpha*tm+(1-alpha)*sm
            fss,lss=model(shifted);ft,lt=model(z)
            q,ag,rel,pseudo,proto,cos,gap=reliable(ft,fss,yb,lt,epoch>10)
            mask=rel if a.reliability_target else q
            target_loss,pi=target_objective(lt,pseudo,mask,pi,a.confusion_correction)
            src_loss=F.cross_entropy(ls,yb);shift_loss=F.cross_entropy(lss,yb)
            loss=src_loss+.5*shift_loss+.5*target_loss
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            bs=len(yb);sums+=np.array([src_loss.item(),shift_loss.item(),target_loss.item()])*bs;n+=bs;seen+=len(z)
            correct+=np.array([(ls.argmax(1)==yb).sum().item(),(lss.argmax(1)==yb).sum().item()])
            conf_n+=q.sum().item();agree_n+=ag.sum().item();rel_n+=rel.sum().item()
            for j,v in enumerate((pseudo[q],proto,pseudo[rel])):hist[j]+=np.bincount(v.cpu().numpy(),minlength=7)
            cos_sum+=cos.sum().item();gap_sum+=gap.sum().item()
            records.append((ids.numpy(),pseudo.cpu().numpy(),rel.cpu().numpy()))
        # Offline GT diagnostics on the exact observed samples; never passed to loss.
        ids,pred,chosen=[np.concatenate(v) for v in zip(*records)]
        truth=tg[tc[ids,0],tc[ids,1]].astype('int64')-1
        precision=[];recall=[]
        for c in range(7):
            tp=int(((pred==c)&(truth==c)&chosen).sum());den=int(((pred==c)&chosen).sum());gt=int((truth==c).sum())
            precision.append(tp/den if den else None);recall.append(tp/gt if gt else None)
        weights=(pi.mean()/(pi+1e-5)).sqrt().clamp(.5,2)
        row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_target=sums[2]/n,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,confidence_qualified_count=conf_n,classifier_prototype_agreement_count=agree_n,reliable_target_count=rel_n,reliable_target_ratio=rel_n/seen,pseudo_label_coverage=conf_n/seen,pseudo_label_class_histogram=hist[0].tolist(),prototype_label_histogram=hist[1].tolist(),reliable_pseudo_label_histogram=hist[2].tolist(),mean_prototype_top1_cosine=cos_sum/seen,mean_prototype_top1_top2_margin=gap_sum/seen,reliable_precision_per_class=precision,reliable_recall_per_class=recall,ema_distribution=pi.tolist(),class_weights=weights.tolist(),min_class_weight=weights.min().item(),max_class_weight=weights.max().item())
        history.append(row);print(json.dumps(row),flush=True);(out/'history.json').write_text(json.dumps(history,indent=2))
    torch.save(dict(model=model.state_dict(),config=config,seed=a.seed),out/'final.pth')
    result=evaluate(model,el,dev,7)
    for k in ('source_accuracy','shifted_source_accuracy','pseudo_label_coverage','reliable_target_ratio','pseudo_label_class_histogram','reliable_pseudo_label_histogram'):result[k]=history[-1][k]
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    print(json.dumps(result),flush=True);summary()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1341);p.add_argument('--adaptive_shift',action='store_true');p.add_argument('--reliability_target',action='store_true');p.add_argument('--confusion_correction',action='store_true');run(p.parse_args())
