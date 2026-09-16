"""Fixed-alpha Houston E1/E2/E3. Target GT is offline diagnostic only."""
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

def supcon(features, labels):
    z=F.normalize(features,dim=1)
    sim=z@z.T/.1
    off=~torch.eye(len(z),dtype=torch.bool,device=z.device)
    positive=(labels[:,None]==labels[None,:]) & off
    counts=positive.sum(1);valid=counts>0
    if not valid.any():return features.sum()*0,0,0,int(off.sum())
    logprob=sim-torch.logsumexp(sim.masked_fill(~off,float('-inf')),dim=1,keepdim=True)
    loss=-(logprob.masked_fill(~positive,0).sum(1)[valid]/counts[valid]).mean()
    return loss,int(valid.sum()),int(positive.sum()),int((off & ~positive).sum())

def augment(x,rng):
    h=torch.rand((len(x),1,1,1),device=x.device,generator=rng)<.5
    v=torch.rand((len(x),1,1,1),device=x.device,generator=rng)<.5
    return torch.where(v,torch.where(h,x.flip(-1),x).flip(-2),torch.where(h,x.flip(-1),x))

def run(a):
    tag={'E1':'gap_partition_diag','E2':'smallgap_inter','E3':'customized_gap_learning'}[a.experiment]
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
    alpha=np.full(48,.8,dtype=np.float32)
    config=dict(vars(a),normalization='none',use_ilda=False,optimizer='Adam',lr=.001,weight_decay=0,batch_size=32,epochs=100,source_per_class=180,patch_size=7,warmup=10,threshold=.9,alpha=.8,epsilon=1e-5,tau=.1,lambda_inter=.05 if a.experiment!='E1' else 0,lambda_intra=.05 if a.experiment=='E3' else 0,noise_std='0.01 * full target cube per-band std',extra_view_bn='running buffers restored after weak/strong; train-mode batch statistics',augmentation_rng='independent CUDA generator seed+10000',loss='L_src + .5 L_shift + .5 L_target',code_sha256=file_sha256(__file__),model_sha256=file_sha256(ROOT/'models/sceneshift_net_dcrn.py'),trainer_dependency_sha256=file_sha256(ROOT/'sceneshiftnet_dcrn_train.py'),data_hash={str(p):file_sha256(p) for p in paths})
    config['loss']={'E1':'L_src + .5 L_shift + .5 CE_confident','E2':'L_src + .5 L_shift + .5 CE_small + .05 L_inter','E3':'L_src + .5 L_shift + .5 CE_small + .05 L_inter + .05 L_intra'}[a.experiment]
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'source_code.py').write_text(Path(__file__).read_text())
    np.savez(out/'indices.npz',source=centers,source_labels=y,target=tc)
    assert torch.cuda.is_available(),'CUDA required'
    dev='cuda';model=SceneShiftNetDCRN(48,7,7).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.001)
    sm,ss,tm,ts,alpha=[torch.tensor(v,device=dev)[None,:,None,None] for v in (sm,ss,tm,ts,alpha)]
    aug_rng=torch.Generator(device=dev).manual_seed(a.seed+10000);history=[]
    for epoch in range(1,101):
        model.train();it=iter(tl);sums=np.zeros(5);correct=np.zeros(2);seen=n=conf_n=agree_n=rel_n=0
        hist=np.zeros((4,7),dtype=int);cos_sum=gap_sum=0.;records=[];anchors=positives=negatives=intra_n=view_agree=view_n=0
        for x,yb in sl:
            try:z,ids=next(it)
            except StopIteration:it=iter(tl);z,ids=next(it)
            x,yb,z=x.to(dev),yb.to(dev),z.to(dev)
            fs,ls=model(x)
            shifted=(x-sm)/(ss+1e-5)*(.8*ts+.2*ss)+.8*tm+.2*sm
            fss,lss=model(shifted);ft,lt=model(z)
            q,ag,small,pseudo,proto,cos,gap=reliable(ft,fss,yb,lt,True)
            large=q & ~ag
            mask=q if a.experiment=='E1' else small
            target_loss=F.cross_entropy(lt[mask],pseudo[mask]) if epoch>10 and mask.any() else lt.sum()*0
            inter=ft.sum()*0;intra=ft.sum()*0
            if epoch>10 and a.experiment!='E1':
                inter,na,npair,nneg=supcon(torch.cat((fss,ft[small])),torch.cat((yb,pseudo[small])))
                anchors+=na;positives+=npair;negatives+=nneg
            if epoch>10 and a.experiment=='E3' and large.any():
                zl=z[large];weak=augment(zl,aug_rng);strong=augment(zl,aug_rng)+torch.randn(zl.shape,device=dev,generator=aug_rng)*(.01*ts)
                # Additional views must not change the base three-forward BN buffers.
                buffers={n:v.clone() for n,v in model.named_buffers()}
                with torch.no_grad():pw=model(weak)[1].softmax(1)
                strong_logits=model(strong)[1]
                valid=pw.max(1).values>.9
                view_agree+=int((pw.argmax(1)==strong_logits.argmax(1)).sum());view_n+=len(zl)
                if valid.any():
                    intra=F.kl_div(strong_logits[valid].log_softmax(1),pw[valid].detach(),reduction='batchmean');intra_n+=int(valid.sum())
            src_loss=F.cross_entropy(ls,yb);shift_loss=F.cross_entropy(lss,yb)
            loss=src_loss+.5*shift_loss+.5*target_loss+.05*inter+.05*intra
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();opt.step()
            if epoch>10 and a.experiment=='E3' and large.any():
                with torch.no_grad():
                    for name,v in model.named_buffers():v.copy_(buffers[name])
            bs=len(yb);sums+=np.array([src_loss.item(),shift_loss.item(),target_loss.item(),inter.item(),intra.item()])*bs;n+=bs;seen+=len(z)
            correct+=np.array([(ls.argmax(1)==yb).sum().item(),(lss.argmax(1)==yb).sum().item()])
            conf_n+=q.sum().item();agree_n+=ag.sum().item();rel_n+=small.sum().item()
            for j,v in enumerate((pseudo[q],proto,pseudo[small],pseudo[large])):hist[j]+=np.bincount(v.cpu().numpy(),minlength=7)
            cos_sum+=cos.sum().item();gap_sum+=gap.sum().item()
            records.append((ids.numpy(),pseudo.cpu().numpy(),small.cpu().numpy(),large.cpu().numpy()))
        # Offline GT diagnostics on the exact observed samples; never passed to loss.
        ids,pred,small_np,large_np=[np.concatenate(v) for v in zip(*records)]
        truth=tg[tc[ids,0],tc[ids,1]].astype('int64')-1
        offline={}
        for group,chosen in [('small',small_np),('large',large_np)]:
            precision=[];recall=[]
            for c in range(7):
                tp=int(((pred==c)&(truth==c)&chosen).sum());den=int(((pred==c)&chosen).sum());gt=int((truth==c).sum())
                precision.append(tp/den if den else None);recall.append(tp/gt if gt else None)
            offline[group+'_precision']=float((pred[chosen]==truth[chosen]).mean()) if chosen.any() else None
            offline[group+'_precision_per_class']=precision;offline[group+'_recall_per_class']=recall
        row=dict(epoch=epoch,L_src=sums[0]/n,L_shift=sums[1]/n,L_target=sums[2]/n,L_inter=sums[3]/n,L_intra=sums[4]/n,source_accuracy=correct[0]/n,shifted_source_accuracy=correct[1]/n,confidence_qualified_count=conf_n,small_gap_count=rel_n,large_gap_count=conf_n-rel_n,small_gap_ratio=rel_n/seen,large_gap_ratio=(conf_n-rel_n)/seen,pseudo_label_coverage=conf_n/seen,pseudo_label_class_histogram=hist[0].tolist(),prototype_label_histogram=hist[1].tolist(),small_gap_histogram=hist[2].tolist(),large_gap_histogram=hist[3].tolist(),mean_prototype_top1_cosine=cos_sum/seen,mean_prototype_top1_top2_margin=gap_sum/seen,valid_contrastive_anchors=anchors,positive_pairs=positives,negative_pairs=negatives,large_gap_consistency_samples=intra_n,weak_strong_agreement=view_agree/view_n if view_n else None,**offline)
        history.append(row);print(json.dumps(row),flush=True);(out/'history.json').write_text(json.dumps(history,indent=2))
    torch.save(dict(model=model.state_dict(),config=config,seed=a.seed),out/'final.pth')
    result=evaluate(model,el,dev,7)
    for k in ('source_accuracy','shifted_source_accuracy','pseudo_label_coverage','small_gap_ratio','large_gap_ratio','pseudo_label_class_histogram','small_gap_histogram','large_gap_histogram'):result[k]=history[-1][k]
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    with (out/'history.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
    print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=1341);p.add_argument('--experiment',choices=['E1','E2','E3'],required=True);run(p.parse_args())
