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
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(LEGACY))
import utils  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
from net2 import DSANSS  # noqa: E402
from class_conditional_flow_uda import (compute_source_prototypes, compute_soft_membership,
    uncertainty_filter, classwise_sinkhorn_ot, sample_ot_pairs, ConditionalFlowMLP,
    flow_matching_loss, bridge_classification_loss)  # noqa: E402

SEED, EPOCHS, CLASSES, BATCH, HALF_WIDTH = 1341, 100, 7, 32, 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_FEATURE_CHUNK = 128

def print_cuda_mem(tag):
    if DEVICE.type != 'cuda': return
    gb = 1024 ** 3
    print(f"[CUDA {tag}] allocated={torch.cuda.memory_allocated()/gb:.3f} GB, reserved={torch.cuda.memory_reserved()/gb:.3f} GB, max_allocated={torch.cuda.max_memory_allocated()/gb:.3f} GB, max_reserved={torch.cuda.max_memory_reserved()/gb:.3f} GB", flush=True)

def encode_chunked(model, x, chunk_size=TARGET_FEATURE_CHUNK):
    parts=[]
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            parts.append(encode(model, x[start:start+chunk_size], DEVICE).detach())
    out=torch.cat(parts, dim=0)
    del parts
    return out

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

def run(transport, out_dir, epochs=EPOCHS, bridge_mode='flow_bridge', config='default'):
    conservative = config == 'conservative'
    warmup_epochs = 10 if conservative else 0
    tau_r = 0.90 if conservative else 0.45
    lambda_fm = 0.1 if conservative else 0.5
    lambda_bridge = 0.1 if conservative else 0.5
    pairs_per_class = 32 if conservative else 64
    seed_all(); out_dir.mkdir(parents=True, exist_ok=True)
    src, src_gt = utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston13.mat'), str(LEGACY/'datasets/Houston/Houston13_7gt.mat'))
    tgt, tgt_gt = utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston18.mat'), str(LEGACY/'datasets/Houston/Houston18_7gt.mat'))
    src, tgt = ILDA(src, tgt, 2, 0.009)
    sx, sy = utils.get_sample_data(src, src_gt, HALF_WIDTH, 180)
    _, tx, ty, *_ = utils.get_all_data(tgt, tgt_gt, HALF_WIDTH)
    sx=torch.tensor(sx).float(); sy=torch.tensor(sy).long(); tx=torch.tensor(tx).float(); ty=torch.tensor(ty).long()
    # Keep the smoke test tractable; full target remains used for evaluation.
    g = torch.Generator().manual_seed(SEED)
    tx_ot = tx[torch.randperm(tx.shape[0], generator=g)[:2048]]
    model=DSANSS(48, 7, CLASSES).to(DEVICE); flow=ConditionalFlowMLP(288, CLASSES).to(DEVICE)
    opt=torch.optim.SGD(list(model.parameters())+list(flow.parameters()), lr=0.01, momentum=0.9, weight_decay=5e-4)
    loader=DataLoader(TensorDataset(sx,sy), batch_size=BATCH, shuffle=True, drop_last=False); history=[]; best=-1
    for epoch in range(1,epochs+1):
        if DEVICE.type == 'cuda': torch.cuda.reset_peak_memory_stats()
        model.train(); flow.train(); sums=np.zeros(3); n=0
        for xb,yb in loader:
            xb,yb=xb.to(DEVICE),yb.to(DEVICE); opt.zero_grad(); fs=encode(model,xb,DEVICE); logits=model.fc1(fs); lsrc=nn.functional.cross_entropy(logits,yb); lfm=fs.sum()*0; lbr=fs.sum()*0
            if transport and epoch > warmup_epochs:
                with torch.no_grad():
                    print_cuda_mem('before_target_ot')
                    ft=encode_chunked(model,tx_ot.to(DEVICE)); print_cuda_mem('after_target_ot')
                    probs=model.fc1(ft).softmax(-1); proto,valid=compute_source_prototypes(fs.detach(),yb,CLASSES); r=compute_soft_membership(ft,probs,proto,beta=1.0,temperature=1.0,valid_classes=valid); keep=uncertainty_filter(r,tau_r); ot=classwise_sinkhorn_ot(fs.detach(),yb,ft.detach(),r,CLASSES,filtered_mask=keep); print_cuda_mem('after_ot')
                # Re-gather source features outside no_grad: bridge can update encoder.
                ps,pt,pc=sample_ot_pairs(ot,fs,ft.detach(),pairs_per_class)
                if ps.numel():
                    tau_max=0.3 if epoch<=20 else (0.5 if epoch<=50 else 0.8)
                    lfm=flow_matching_loss(flow,ps.detach(),pt.detach(),pc,tau_max)
                    lbr=bridge_classification_loss(model.fc1,ps,pt,pc,tau_max,flow_model=(flow if bridge_mode=='flow_bridge' else None))
            loss=lsrc+lambda_fm*lfm+lambda_bridge*lbr; loss.backward(); opt.step(); sums += [lsrc.item(),lfm.item(),lbr.item()]; n+=1
            if transport and epoch > warmup_epochs: print_cuda_mem('after_backward')
        model.eval(); print_cuda_mem('before_evaluation')
        with torch.no_grad(): pred=[]
        for batch in DataLoader(tx,batch_size=256):
            with torch.no_grad(): pred.extend(model.fc1(encode(model,batch.to(DEVICE),DEVICE)).argmax(-1).cpu().tolist())
        print_cuda_mem('after_evaluation')
        met=metrics(pred,ty.numpy()); row={"epoch":epoch,"source_loss":sums[0]/n,"fm_loss":sums[1]/n,"bridge_loss":sums[2]/n,**met}
        if transport:
            with torch.no_grad():
                f=encode_chunked(model,tx.to(DEVICE)); p=model.fc1(f).softmax(-1); proto,valid=compute_source_prototypes(encode_chunked(model,sx.to(DEVICE)),sy,CLASSES); r=compute_soft_membership(f,p,proto,valid_classes=valid); keep=uncertainty_filter(r,tau_r); rp=r.argmax(1); row.update({"membership_max_mean":float(r.max(1).values.mean()),"membership_max_min":float(r.max(1).values.min()),"membership_max_max":float(r.max(1).values.max()),"membership_entropy":float(-(r*r.clamp_min(1e-8).log()).sum(1).mean()),"filter_ratio":float(keep.float().mean()),"retained_membership_accuracy":float((rp[keep]==ty.to(DEVICE)[keep]).float().mean()) if keep.any() else 0.0,"class_mass":r.sum(0).cpu().tolist(),"retained_predicted_class_counts":torch.bincount(rp[keep],minlength=CLASSES).cpu().tolist(),"valid_classes":valid.cpu().tolist(),"tau_max":0.3 if epoch<=20 else (0.5 if epoch<=50 else 0.8)})
                del f,p,proto,r,keep,rp
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        row.update({'config':config,'warmup_epochs':warmup_epochs,'tau_r':tau_r,'lambda_fm':lambda_fm,'lambda_bridge':lambda_bridge,'pairs_per_class':pairs_per_class,'transport_active':bool(transport and epoch>warmup_epochs)})
        history.append(row); print(json.dumps(row))
        if met["oa"]>best: best=met["oa"]; torch.save({"model":model.state_dict(),"flow":flow.state_dict(),"epoch":epoch,"metrics":met},out_dir/'best_target_oa.pth')
    (out_dir/'history.json').write_text(json.dumps(history,indent=2)); return history

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('--mode',choices=['source_only','linear_bridge','flow_bridge'],default='flow_bridge'); p.add_argument('--config',choices=['default','conservative'],default='default'); p.add_argument('--epochs',type=int,default=EPOCHS); p.add_argument('--out',default=str(ROOT/'runs_houston_v01')); a=p.parse_args(); run(a.mode!='source_only',Path(a.out)/a.mode,a.epochs,'flow_bridge' if a.mode=='flow_bridge' else 'linear_bridge',a.config)
