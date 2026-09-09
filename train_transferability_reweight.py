"""Clean DCRN + fixed alpha=0.8 Scene Shift with class-wise shifted-loss weights.

Weights are calibrated only from source labels, unlabeled target statistics, and
source-validation semantic safety. Target GT is never loaded in this script.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
import train as clean  # noqa: E402
import utils  # noqa: E402
from model import DCRNClassifier  # noqa: E402


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_transferability_weights(diagnostic_root, gamma=1.0, lambda_min=0.2, lambda_max=1.0):
    need = json.loads((diagnostic_root / "shift_need_per_class.json").read_text())["need_normalized"]
    records = json.loads((diagnostic_root / "semantic_safety_per_class.json").read_text())["records"]
    retention = [next(row["confidence_retention"] for row in records if row["class"] == c and abs(row["alpha"] - 0.8) < 1e-6) for c in range(1, 8)]
    sensitivity = [max(0.0, 1.0 - float(value)) for value in retention]
    raw = np.asarray(need, dtype=np.float32) + gamma * np.asarray(sensitivity, dtype=np.float32)
    t = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-8)
    lambdas = lambda_min + t * (lambda_max - lambda_min)
    return {
        "alpha": 0.8, "gamma": gamma, "lambda_min": lambda_min, "lambda_max": lambda_max,
        "need": [float(x) for x in need], "confidence_retention_alpha_0_8": retention,
        "shift_sensitivity": sensitivity, "combined_raw": raw.tolist(), "T_normalized": t.tolist(), "lambda": lambdas.tolist(),
        "source_only_calibration": True, "target_gt_used_for_training_or_selection": False,
    }


def main():
    p = argparse.ArgumentParser(); p.add_argument("--diagnostic-root", type=Path, default=HERE/"runs_transferability_aware_1174"); p.add_argument("--output", type=Path, default=HERE/"runs_transferability_reweight_1174"); p.add_argument("--device", default="cuda:0"); p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=0.002); args=p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    set_seed(1174); device=torch.device(args.device)
    weights=load_transferability_weights(args.diagnostic_root); (args.output/"transferability_weights.json").write_text(json.dumps(weights,indent=2)); lambdas=np.asarray(weights["lambda"],dtype=np.float32)
    source, source_gt=utils.load_data_houston(str(ROOT/"datasets/Houston/Houston13.mat"),str(ROOT/"datasets/Houston/Houston13_7gt.mat")); target=hdf5storage.loadmat(str(ROOT/"datasets/Houston/Houston18.mat"))["ori_data"]; source=source.astype(np.float32); target=target.astype(np.float32)
    tc,ty,vc,vy=clean.source_split(source_gt,1174); train_x=clean.center_patches(source,tc); val_x=clean.center_patches(source,vc)
    sf,tf=source.reshape(-1,source.shape[-1]),target.reshape(-1,target.shape[-1]); sm,ss=sf.mean(0),sf.std(0); tm,ts=tf.mean(0),tf.std(0)
    loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(ty)),batch_size=args.batch_size,shuffle=True,drop_last=True); val_loader=DataLoader(TensorDataset(torch.from_numpy(val_x),torch.from_numpy(vy)),batch_size=args.batch_size,shuffle=False)
    model=DCRNClassifier().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); best={"val_acc":-1.0}; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=correct=seen=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits=model(clean.augment(x)); shifted=clean.scene_shift(x,sm,ss,tm,ts,strength=0.8); shifted_logits=model(clean.augment(shifted)); raw_loss=ce(logits,y); shifted_loss=nn.functional.cross_entropy(shifted_logits,y,reduction="none"); loss=raw_loss+(torch.as_tensor(lambdas,device=device)[y]*shifted_loss).mean(); opt.zero_grad(); loss.backward(); opt.step(); loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); seen+=len(y)
        model.eval(); vl=vcnt=vs=0
        with torch.no_grad():
            for x,y in val_loader:
                z=model(x.to(device)); yy=y.to(device); vl+=ce(z,yy).item()*len(y); vcnt+=(z.argmax(1)==yy).sum().item(); vs+=len(y)
        row={"epoch":epoch,"train_loss":loss_sum/seen,"train_acc":correct/seen,"val_loss":vl/vs,"val_acc":vcnt/vs}; history.append(row); print(json.dumps(row),flush=True)
        if row["val_acc"]>best["val_acc"]:
            best=row.copy(); torch.save({"model":model.state_dict(),"group":"TRANSFERABILITY_REWEIGHT","group_description":"DCRN + global Scene Shift alpha=0.8 + class-wise shifted-loss reweighting","split_seed":1174,"optimization_seed":1174,"use_ilda":False,"use_scene_shift":True,"scene_shift_strength":0.8,"transferability_weights":weights,"target_gt_used_for_training_or_selection":False,"best":best,"backbone":{"name":"DCRN_02","call":"DCRN_02(x,x)","cross_attention_source_target_interaction":False}},args.output/"best.pth")
    (args.output/"history.json").write_text(json.dumps(history,indent=2)); (args.output/"summary.json").write_text(json.dumps({"config":{"alpha":0.8,"raw_loss_weight":1.0,"shifted_loss_weight":"lambda_class","epochs":args.epochs,"batch_size":args.batch_size,"lr":args.lr,"target_gt_used_for_training_or_selection":False},"best":best},indent=2))

if __name__ == "__main__": main()
