"""Clean DAMamba-style Mamba backbone audit (no UDA adaptation losses)."""
from __future__ import annotations
import argparse, json, random, sys, importlib.util
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
from config_Houston import HalfWidth  # noqa: E402
from mamba_model import MambaBackboneClassifier  # noqa: E402
from own_backbone import SpectralSpatialGatedMambaClassifier  # noqa: E402
_spec = importlib.util.spec_from_file_location("hsi_utils", ROOT / "utils.py")
utils = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(utils)

SPLITS = (1174, 1703, 2141)

def center_patches(cube, centers, width=12):
    half = width // 2
    # Official DAMamba pads by int(width/2)+1 with symmetric reflection, then
    # slices x-half:x+half. This preserves the even-patch center convention.
    pad = half + 1
    padded = np.pad(cube, ((pad, pad), (pad, pad), (0, 0)), mode="symmetric")
    out = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        out[i] = padded[row + 1:row + 1 + width, col + 1:col + 1 + width].transpose(2, 0, 1)
    return out

def source_split(gt, seed):
    rng = np.random.RandomState(seed)
    padded = np.pad(gt, HalfWidth)
    rows, cols = np.nonzero(padded)
    train, val = [], []
    for cls in range(int(padded.max())):
        ids = [i for i in range(len(rows)) if padded[rows[i], cols[i]] == cls + 1]
        rng.shuffle(ids); train.extend(ids[:180]); val.extend(ids[180:])
    rng.shuffle(train); rng.shuffle(val)
    tc = np.asarray([(rows[i] - HalfWidth, cols[i] - HalfWidth) for i in train], np.int64)
    vc = np.asarray([(rows[i] - HalfWidth, cols[i] - HalfWidth) for i in val], np.int64)
    return tc, gt[tc[:,0], tc[:,1]].astype(np.int64)-1, vc, gt[vc[:,0],vc[:,1]].astype(np.int64)-1

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def augment(x):
    if torch.rand(()) < 0.5: x = x.flip(-1)
    if torch.rand(()) < 0.5: x = x.flip(-2)
    return x

def scene_shift(x, sm, ss, tm, ts, strength=0.7):
    sm = torch.as_tensor(sm, device=x.device, dtype=x.dtype)[None,:,None,None]
    ss = torch.as_tensor(ss, device=x.device, dtype=x.dtype)[None,:,None,None]
    tm = torch.as_tensor(tm, device=x.device, dtype=x.dtype)[None,:,None,None]
    ts = torch.as_tensor(ts, device=x.device, dtype=x.dtype)[None,:,None,None]
    shifted = (x-sm)/(ss+1e-5)
    shifted = shifted*(strength*ts+(1-strength)*ss) + strength*tm + (1-strength)*sm
    scale = 1.0 + 0.04*torch.randn(x.size(0),1,1,1,device=x.device)
    noise = F.avg_pool2d(torch.randn_like(shifted), 5, 1, 2)
    return (shifted*scale + 0.015*noise).clamp(0,1)

def train_one(args):
    set_seed(args.optimization_seed); device=torch.device(args.device)
    source, source_gt = utils.load_data_houston(str(ROOT/"datasets/Houston/Houston13.mat"), str(ROOT/"datasets/Houston/Houston13_7gt.mat"))
    target = hdf5storage.loadmat(str(ROOT/"datasets/Houston/Houston18.mat"))["ori_data"]
    source, target = source.astype(np.float32), target.astype(np.float32)
    tc, ty, vc, vy = source_split(source_gt, args.split_seed)
    train_x, val_x = center_patches(source,tc), center_patches(source,vc)
    sf, tf = source.reshape(-1,source.shape[-1]), target.reshape(-1,target.shape[-1])
    sm, ss, tm, ts = sf.mean(0), sf.std(0), tf.mean(0), tf.std(0)
    train_loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(ty)),batch_size=args.batch_size,shuffle=True,drop_last=True)
    val_loader=DataLoader(TensorDataset(torch.from_numpy(val_x),torch.from_numpy(vy)),batch_size=args.batch_size,shuffle=False)
    model=(SpectralSpatialGatedMambaClassifier() if args.model_type == "own" else MambaBackboneClassifier()).to(device)
    if args.optimizer == "sgd":
        if args.official_recipe and args.model_type != "own":
            # DAMamba's scheduler applies args.lr as the first multiplier.  Its
            # parameter groups therefore start at 0.1/1.0 and become
            # 0.1*args.lr/args.lr after the first scheduler step.
            params=[{"params": model.backbone.parameters(), "lr": 0.1},
                    {"params": list(model.bottleneck.parameters()) + list(model.classifier.parameters()), "lr": 1.0}]
            opt=torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4)
        else:
            # For the own backbone use one uniform group.  Keep the official
            # scheduler parameterization (lambda(0)=args.lr), hence initialize
            # the group at 1.0 so its effective first rate is args.lr.
            opt=torch.optim.SGD(model.parameters(), lr=(1.0 if args.official_recipe else args.lr), momentum=0.9, weight_decay=5e-4)
        if args.lr_scheduler:
            scheduler=torch.optim.lr_scheduler.LambdaLR(
                opt, lambda step: (args.lr if args.official_recipe else 1.0)
                * (1.0 + args.lr_gamma * step) ** (-args.lr_decay)
            )
        else:
            scheduler=None
    else:
        opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4); scheduler=None
    ce=nn.CrossEntropyLoss(); best={"val_acc":-1.0}; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=correct=seen=0
        grad_norm=0.0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); logits=model(augment(x)); loss=ce(logits,y)
            if args.use_scene_shift: loss=loss+0.5*ce(model(augment(scene_shift(x,sm,ss,tm,ts))),y)
            opt.zero_grad(); loss.backward()
            grad_norm = float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in model.parameters() if p.grad is not None)))
            opt.step()
            if scheduler is not None: scheduler.step()
            loss_sum+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); seen+=len(y)
        model.eval(); vl=vcnt=vs=0
        with torch.no_grad():
            for x,y in val_loader:
                z=model(x.to(device)); yt=y.to(device); vl+=ce(z,yt).item()*len(y); vcnt+=(z.argmax(1)==yt).sum().item(); vs+=len(y)
            probe = next(iter(val_loader))[0][:min(32, args.batch_size)].to(device)
            feature_norm=float(model.forward_features(probe).norm(dim=1).mean())
        cls_module = model.head if args.model_type == "own" else model.classifier
        cls_norm=float(torch.sqrt(sum(p.detach().norm()**2 for p in cls_module.parameters())))
        row={"epoch":epoch,"train_loss":loss_sum/seen,"train_acc":correct/seen,"val_loss":vl/vs,"val_acc":vcnt/vs,"lr":opt.param_groups[0]["lr"],"feature_norm":feature_norm,"classifier_weight_norm":cls_norm,"gradient_norm":grad_norm}; history.append(row); print(json.dumps(row),flush=True)
        if row["val_acc"]>best["val_acc"]:
            best=row.copy(); torch.save({"model":model.state_dict(),"model_type":args.model_type,"patch_size":12,"backbone":"SpectralSpatialGatedMamba" if args.model_type == "own" else "DAMamba MambaFeature","backbone_output_dim":getattr(model,"representation_dim",4608),"split_seed":args.split_seed,"optimization_seed":args.optimization_seed,"use_scene_shift":args.use_scene_shift,"target_gt_used_for_training_or_selection":False,"disabled_losses":["prototype","pseudo_label","LMMD","FixMatch","intra","inter","foundation","semantic","neighborhood","modulation"] ,"best":best},args.output/"best.pth")
    cfg={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}; cfg.update({"model_type":args.model_type,"patch_size":12,"backbone_output_dim":getattr(model,"representation_dim",4608),"target_gt_used_for_training_or_selection":False}); (args.output/"history.json").write_text(json.dumps(history,indent=2)); (args.output/"summary.json").write_text(json.dumps({"config":cfg,"best":best},indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--split-seed",type=int,choices=SPLITS,required=True); p.add_argument("--optimization-seed",type=int,default=1174); p.add_argument("--epochs",type=int,default=100); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--lr",type=float,default=0.001); p.add_argument("--optimizer",choices=("sgd","adamw"),default="sgd"); p.add_argument("--official-recipe",action="store_true"); p.add_argument("--lr-scheduler",action="store_true"); p.add_argument("--lr-gamma",type=float,default=0.0003); p.add_argument("--lr-decay",type=float,default=0.75); p.add_argument("--device",default="cuda:0"); p.add_argument("--use-scene-shift",action="store_true"); p.add_argument("--model-type",choices=("mamba","own"),default="mamba"); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); train_one(a)
if __name__=="__main__": main()
