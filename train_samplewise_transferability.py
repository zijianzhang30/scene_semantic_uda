"""Sample-wise Transferability-Aware Scene Shift (1174 clean diagnostic).

The discriminator is an independent patch-level source/target classifier. Its
loss never updates DCRN, and the classifier loss never updates the discriminator.
No target labels are loaded.
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import train as clean
import utils
from model import DCRNClassifier

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent


class PatchDomainDiscriminator(nn.Module):
    def __init__(self, bands=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(bands, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x.mean(dim=(-1, -2))).squeeze(1)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def margin_for_class(logits, labels):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone(); masked.scatter_(1, labels[:, None], -torch.inf)
    return true - masked.max(1).values


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()), "max": float(values.max())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=HERE/"runs_samplewise_transferability_1174")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.002)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    set_seed(1174); device=torch.device(args.device)
    source, source_gt=utils.load_data_houston(str(ROOT/"datasets/Houston/Houston13.mat"),str(ROOT/"datasets/Houston/Houston13_7gt.mat")); target=hdf5storage.loadmat(str(ROOT/"datasets/Houston/Houston18.mat"))["ori_data"]
    source=source.astype(np.float32); target=target.astype(np.float32)
    tc,ty,vc,vy=clean.source_split(source_gt,1174); train_x=clean.center_patches(source,tc); val_x=clean.center_patches(source,vc)
    target_centers=np.stack(np.meshgrid(np.arange(target.shape[0]),np.arange(target.shape[1]),indexing="ij"),-1).reshape(-1,2); target_x=clean.center_patches(target,target_centers)
    sf,tf=source.reshape(-1,source.shape[-1]),target.reshape(-1,target.shape[-1]); sm,ss=sf.mean(0),sf.std(0); tm,ts=tf.mean(0),tf.std(0)
    source_loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(ty)),batch_size=args.batch_size,shuffle=True,drop_last=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(val_x),torch.from_numpy(vy)),batch_size=args.batch_size,shuffle=False)
    target_loader=DataLoader(TensorDataset(torch.from_numpy(target_x)),batch_size=args.batch_size,shuffle=True,drop_last=True)
    model=DCRNClassifier().to(device); discriminator=PatchDomainDiscriminator(source.shape[-1]).to(device)
    opt_cls=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); opt_d=torch.optim.AdamW(discriminator.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); bce=nn.BCEWithLogitsLoss(); best={"val_acc":-1.0}; history=[]; target_iter=None
    lambdas_last=None
    for epoch in range(1,args.epochs+1):
        model.train(); discriminator.train(); loss_sum=correct=seen=0; dloss_sum=dcorrect=dseen=0; a_vals=[]; r_vals=[]; w_vals=[]; class_stats={c:{"a":[],"r":[],"w":[]} for c in range(7)}
        target_iter=iter(target_loader)
        for x,y in source_loader:
            x,y=x.to(device),y.to(device)
            try: xt=next(target_iter)[0].to(device)
            except StopIteration: target_iter=iter(target_loader); xt=next(target_iter)[0].to(device)
            # Independent discriminator update: source=0, target=1.
            d_source=discriminator(x); d_target=discriminator(xt); d_loss=bce(d_source,torch.zeros_like(d_source))+bce(d_target,torch.ones_like(d_target)); opt_d.zero_grad(); d_loss.backward(); opt_d.step()
            dcorrect += int(((torch.sigmoid(d_source.detach())<0.5).sum() + (torch.sigmoid(d_target.detach())>=0.5).sum()).item()); dseen += 2*len(x); dloss_sum += float(d_loss.detach())
            # Classifier update with fixed alpha=0.8 Scene Shift.
            raw_input=clean.augment(x); shifted=clean.scene_shift(x,sm,ss,tm,ts,strength=0.8); shifted_input=clean.augment(shifted)
            raw_logits=model(raw_input); shifted_logits=model(shifted_input)
            with torch.no_grad():
                affinity=torch.sigmoid(discriminator(shifted)).detach()
                delta=(margin_for_class(raw_logits.detach(),y)-margin_for_class(shifted_logits.detach(),y)).relu(); reliability=torch.exp(-delta); weights=affinity*reliability; weights=weights.clamp(0.2,1.0)
            raw_loss=nn.functional.cross_entropy(raw_logits,y,reduction="none"); shift_loss=nn.functional.cross_entropy(shifted_logits,y,reduction="none"); loss=(raw_loss+weights*shift_loss).mean(); opt_cls.zero_grad(); loss.backward(); opt_cls.step()
            loss_sum+=float(loss.detach())*len(y); correct+=(raw_logits.argmax(1)==y).sum().item(); seen+=len(y)
            a_vals.extend(affinity.cpu().numpy()); r_vals.extend(reliability.cpu().numpy()); w_vals.extend(weights.cpu().numpy()); lambdas_last=weights
            for c in range(7):
                mask=(y==c).detach().cpu().numpy()
                if mask.any(): class_stats[c]["a"].extend(affinity.detach().cpu().numpy()[mask]); class_stats[c]["r"].extend(reliability.detach().cpu().numpy()[mask]); class_stats[c]["w"].extend(weights.detach().cpu().numpy()[mask])
        model.eval(); val_loss=val_correct=val_seen=0
        with torch.no_grad():
            for x,y in val_loader:
                z=model(x.to(device)); yy=y.to(device); val_loss+=ce(z,yy).item()*len(y); val_correct+=(z.argmax(1)==yy).sum().item(); val_seen+=len(y)
        row={"epoch":epoch,"train_loss":loss_sum/seen,"train_acc":correct/seen,"val_loss":val_loss/val_seen,"val_acc":val_correct/val_seen,"discriminator_loss":dloss_sum/len(source_loader),"discriminator_accuracy":dcorrect/max(dseen,1),"affinity":summarize(a_vals),"reliability":summarize(r_vals),"weight":summarize(w_vals),"class_stats":{str(c+1):{k:summarize(v) for k,v in class_stats[c].items()} for c in range(7)}}; history.append(row); print(json.dumps(row),flush=True)
        if row["val_acc"]>best["val_acc"]:
            best=row.copy(); torch.save({"model":model.state_dict(),"discriminator_state":discriminator.state_dict(),"group":"SAMPLEWISE_TRANSFERABILITY","group_description":"DCRN + global Scene Shift alpha=0.8 + sample-wise affinity/reliability weighting","split_seed":1174,"optimization_seed":1174,"use_ilda":False,"use_scene_shift":True,"scene_shift_strength":0.8,"target_gt_used_for_training_or_selection":False,"discriminator_arch":"spatial mean -> Linear(48,64) -> GELU -> Linear(64,1)","disabled_losses":["prototype","pseudo_label","LMMD","FixMatch","intra","inter","foundation","semantic","neighborhood","modulation"],"best":best},args.output/"best.pth")
    (args.output/"history.json").write_text(json.dumps(history,indent=2)); (args.output/"summary.json").write_text(json.dumps({"config":{"split_seed":1174,"optimization_seed":1174,"epochs":args.epochs,"batch_size":args.batch_size,"lr_classifier":args.lr,"lr_discriminator":args.lr,"alpha":0.8,"target_gt_used_for_training_or_selection":False},"best":best},indent=2))

if __name__=='__main__': main()
