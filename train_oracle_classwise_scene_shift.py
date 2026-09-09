"""Upper-bound oracle class-wise Scene Shift training for split 1174.

The alpha vector is intentionally read from target-GT oracle analysis. This is
not a valid target-label-free method and is only an upper-bound experiment.
"""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import hdf5storage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
import train as clean  # noqa: E402
from model import DCRNClassifier  # noqa: E402
import utils  # noqa: E402


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def classwise_scene_shift(x, y, source_mean, source_std, target_mean, target_std, alpha):
    sm = torch.as_tensor(source_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ss = torch.as_tensor(source_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    tm = torch.as_tensor(target_mean, device=x.device, dtype=x.dtype)[None, :, None, None]
    ts = torch.as_tensor(target_std, device=x.device, dtype=x.dtype)[None, :, None, None]
    a = torch.as_tensor(alpha, device=x.device, dtype=x.dtype)[y][:, None, None, None]
    shifted = (x - sm) / (ss + 1e-5)
    shifted = shifted * (a * ts + (1.0 - a) * ss) + a * tm + (1.0 - a) * sm
    scale = 1.0 + 0.04 * torch.randn(x.size(0), 1, 1, 1, device=x.device)
    noise = F.avg_pool2d(torch.randn_like(shifted), 5, 1, 2)
    return (shifted * scale + 0.015 * noise).clamp(0, 1)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--oracle-json", type=Path, default=HERE/"runs_transferability_aware_1174/oracle_alpha_per_class.json"); p.add_argument("--output", type=Path, default=HERE/"runs_transferability_aware_1174/oracle_classwise_train"); p.add_argument("--device", default="cuda:0"); p.add_argument("--epochs", type=int, default=100); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--lr", type=float, default=0.002); args=p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    set_seed(1174); device=torch.device(args.device)
    oracle=json.loads(args.oracle_json.read_text()); alpha=np.asarray(oracle["oracle_alpha_per_class"], dtype=np.float32)
    source, source_gt=utils.load_data_houston(str(ROOT/"datasets/Houston/Houston13.mat"), str(ROOT/"datasets/Houston/Houston13_7gt.mat")); target=hdf5storage.loadmat(str(ROOT/"datasets/Houston/Houston18.mat"))["ori_data"]; source=source.astype(np.float32); target=target.astype(np.float32)
    tc,ty,vc,vy=clean.source_split(source_gt,1174); train_x=clean.center_patches(source,tc); val_x=clean.center_patches(source,vc)
    sf,tf=source.reshape(-1,source.shape[-1]),target.reshape(-1,target.shape[-1]); sm,ss=sf.mean(0),sf.std(0); tm,ts=tf.mean(0),tf.std(0)
    loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(ty)),batch_size=args.batch_size,shuffle=True,drop_last=True); val_loader=DataLoader(TensorDataset(torch.from_numpy(val_x),torch.from_numpy(vy)),batch_size=args.batch_size,shuffle=False)
    model=DCRNClassifier().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); ce=nn.CrossEntropyLoss(); best={"val_acc":-1.0}; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=correct=seen=0
        for x,y in loader:
            x,y=x.to(device),y.to(device); logits=model(clean.augment(x)); shifted=classwise_scene_shift(x,y,sm,ss,tm,ts,alpha); loss=ce(logits,y)+0.5*ce(model(clean.augment(shifted)),y); opt.zero_grad(); loss.backward(); opt.step(); loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); seen+=len(y)
        model.eval(); vl=vcnt=vs=0
        with torch.no_grad():
            for x,y in val_loader:
                z=model(x.to(device)); yy=y.to(device); vl+=ce(z,yy).item()*len(y); vcnt+=(z.argmax(1)==yy).sum().item(); vs+=len(y)
        row={"epoch":epoch,"train_loss":loss_sum/seen,"train_acc":correct/seen,"val_loss":vl/vs,"val_acc":vcnt/vs}; history.append(row); print(json.dumps(row),flush=True)
        if row["val_acc"]>best["val_acc"]:
            best=row.copy(); torch.save({"model":model.state_dict(),"group":"ORACLE","group_description":"Oracle class-wise Target-Guided Scene Shift (upper bound)","split_seed":1174,"optimization_seed":1174,"use_ilda":False,"use_scene_shift":True,"scene_shift_strength":"class-wise oracle","oracle_alpha_per_class":alpha.tolist(),"oracle_source":"target GT post-hoc diagnostic","target_gt_used_for_training_or_selection":True,"best":best,"backbone":{"name":"DCRN_02","call":"DCRN_02(x,x)","cross_attention_source_target_interaction":False}},args.output/"best.pth")
    (args.output/"history.json").write_text(json.dumps(history,indent=2)); (args.output/"summary.json").write_text(json.dumps({"upper_bound_oracle":True,"oracle_alpha_per_class":alpha.tolist(),"target_gt_used_for_training_or_selection":True,"best":best},indent=2))

if __name__ == "__main__": main()
