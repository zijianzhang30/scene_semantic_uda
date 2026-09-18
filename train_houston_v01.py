"""Houston13 -> Houston18 smoke experiment for class-conditional transport.

The legacy project is imported read-only through PYTHONPATH; this script does
not edit it.  Target labels are loaded only for post-training evaluation.
"""
from __future__ import annotations
import csv, json, random, sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
LEGACY = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(LEGACY))
import utils  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
from net2 import DSANSS  # noqa: E402
from class_conditional_flow_uda import (compute_source_prototypes, compute_soft_membership,
    uncertainty_filter, classwise_sinkhorn_ot, sample_ot_pairs, ConditionalFlowMLP,
    flow_matching_loss, bridge_classification_loss)  # noqa: E402

SEED, EPOCHS, CLASSES, BATCH, HALF_WIDTH = 1341, 100, 7, 32, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def metrics(pred, y):
    pred, y = np.asarray(pred), np.asarray(y); cm = np.zeros((CLASSES, CLASSES), int)
    for a,b in zip(y,pred):
        if 0 <= b < CLASSES and 0 <= a < CLASSES: cm[a,b] += 1
    oa = float((pred == y).mean()); per = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(CLASSES), where=cm.sum(1)>0)
    aa = float(per.mean()); total=cm.sum(); po=np.trace(cm)/max(total,1); pe=(cm.sum(0)*cm.sum(1)).sum()/max(total*total,1)
    return {"oa":oa, "aa":aa, "kappa":float((po-pe)/(1-pe+1e-12)), "per_class":per.tolist()}

def encode(model, x, device):
    # DSANSS uses the paired forward API; a zero-sized partner is not supported.
    return model.feature_layers(x, x)[0]

def run(transport, out_dir, epochs=EPOCHS):
    seed_all(); out_dir.mkdir(parents=True, exist_ok=True)
    src, src_gt = utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston13.mat'), str(LEGACY/'datasets/Houston/Houston13_7gt.mat'))
    tgt, tgt_gt = utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston18.mat'), str(LEGACY/'datasets/Houston/Houston18_7gt.mat'))
    src, tgt = ILDA(src, tgt, 2, 0.009)
    sx, sy = utils.get_sample_data(src, src_gt, HALF_WIDTH, 180)
    _, tx, ty, *_ = utils.get_all_data(tgt, tgt_gt, HALF_WIDTH)
    sx=torch.tensor(sx).float(); sy=torch.tensor(sy).long(); tx=torch.tensor(tx).float(); ty=torch.tensor(ty).long()
    # Keep the smoke test tractable; full target remains used for evaluation.
    tx_ot = tx[:2048]
    model=DSANSS(48, 7, CLASSES).to(DEVICE); flow=ConditionalFlowMLP(288, CLASSES).to(DEVICE)
    opt=torch.optim.SGD(list(model.parameters())+list(flow.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loader=DataLoader(TensorDataset(sx,sy), batch_size=BATCH, shuffle=True, drop_last=False); history=[]; best=-1
    for epoch in range(1,epochs+1):
        model.train(); flow.train(); sums=np.zeros(3); n=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); fs=encode(model,xb,DEVICE); logits=model.fc1(fs); lsrc=nn.functional.cross_entropy(logits,yb); lfm=fs.sum()*0; lbr=fs.sum()*0
            if transport:
                with torch.no_grad():
                    ft=encode(model,tx_ot.to(DEVICE),DEVICE); probs=model.fc1(ft).softmax(-1); proto,_=compute_source_prototypes(fs.detach(),yb,CLASSES); r=compute_soft_membership(ft,probs,proto,beta=1.0,temperature=1.0); keep=uncertainty_filter(r,0.45); ot=classwise_sinkhorn_ot(fs.detach(),yb,ft.detach(),r,CLASSES,filtered_mask=keep)
                    # OT decisions and r are detached; pair tensors retain graph for bridge loss.
                    ps,pt,pc=sample_ot_pairs(ot,fs,ft,64)
                if ps.numel(): lfm=flow_matching_loss(flow,ps.detach(),pt.detach(),pc,0.3); lbr=bridge_classification_loss(model.fc1,ps,pt,pc,0.3)
            loss=lsrc+0.5*lfm+0.5*lbr; loss.backward(); opt.step(); sums += [lsrc.item(),lfm.item(),lbr.item()]; n+=1
        model.eval();
        with torch.no_grad(): pred=[]
        for batch in DataLoader(tx,batch_size=256):
            with torch.no_grad(): pred.extend(model.fc1(encode(model,batch.to(DEVICE),DEVICE)).argmax(-1).cpu().tolist())
        met=metrics(pred,ty.numpy()); row={"epoch":epoch,"source_loss":sums[0]/n,"fm_loss":sums[1]/n,"bridge_loss":sums[2]/n,**met}
        if transport:
            with torch.no_grad(): f=encode(model,tx.to(DEVICE),DEVICE); p=model.fc1(f).softmax(-1); proto,_=compute_source_prototypes(encode(model,sx.to(DEVICE),DEVICE),sy.to(DEVICE),CLASSES); r=compute_soft_membership(f,p,proto); row.update({"membership_max_mean":float(r.max(1).values.mean()),"membership_max_min":float(r.max(1).values.min()),"membership_max_max":float(r.max(1).values.max()),"membership_entropy":float(-(r*r.clamp_min(1e-8).log()).sum(1).mean()),"filter_ratio":float(uncertainty_filter(r,.45).float().mean()),"class_mass":r.sum(0).cpu().tolist()})
        history.append(row); print(json.dumps(row))
        if met["oa"]>best: best=met["oa"]; torch.save({"model":model.state_dict(),"flow":flow.state_dict(),"bridge":bridge.state_dict(),"epoch":epoch,"metrics":met},out_dir/'best_target_oa.pth')
    (out_dir/'history.json').write_text(json.dumps(history,indent=2)); return history

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['source_only','transport'],default='transport'); p.add_argument('--epochs',type=int,default=EPOCHS); p.add_argument('--out',default=str(ROOT/'runs_houston_v01')); a=p.parse_args(); run(a.mode=='transport',Path(a.out)/a.mode,a.epochs)
