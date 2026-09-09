"""Post-hoc evaluation for alpha={0,0.4,0.8,1.0} matched 3-seed sweep."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import hdf5storage, matplotlib.pyplot as plt, numpy as np, torch
from sklearn import metrics

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from model import DCRNClassifier  # noqa: E402

ALPHAS = (0.0, 0.4, 0.8, 1.0)
SPLITS = (1174, 1703, 2141)

def patches(cube, centers, width=7):
    h = width // 2
    p = np.pad(cube, ((h, h), (h, h), (0, 0)), mode="constant")
    out = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (r, c) in enumerate(centers): out[i] = p[r:r+width, c:c+width].transpose(2, 0, 1)
    return out

def main():
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, default=HERE/"runs_scene_shift_strength_3seed"); p.add_argument("--device", default="cuda:0"); args = p.parse_args()
    target = hdf5storage.loadmat(str(ROOT/"datasets/Houston/Houston18.mat"))["ori_data"]
    gt = hdf5storage.loadmat(str(ROOT/"datasets/Houston/Houston18_7gt.mat"))["map"]
    centers = np.argwhere(gt > 0).astype(np.int64); y = gt[centers[:,0], centers[:,1]].astype(np.int64)-1; x = patches(target, centers)
    runs = []
    for alpha in ALPHAS:
        for split in SPLITS:
            tag = str(alpha).replace('.', '_'); ck = torch.load(args.root/f"alpha_{tag}"/f"split_{split}"/"best.pth", map_location="cpu")
            m = DCRNClassifier().to(args.device); m.load_state_dict(ck["model"], strict=True); m.eval(); pred=[]
            with torch.no_grad():
                for i in range(0, len(x), 32): pred.append(m(torch.from_numpy(x[i:i+32]).to(args.device)).argmax(1).cpu().numpy())
            pred=np.concatenate(pred); cm=metrics.confusion_matrix(y,pred,labels=np.arange(7)); pc=np.diag(cm)/np.maximum(cm.sum(1),1)
            runs.append({"alpha":alpha,"split":split,"oa":float((y==pred).mean()),"aa":float(pc.mean()),"kappa":float(metrics.cohen_kappa_score(y,pred,labels=np.arange(7))),"per_class_accuracy":pc.tolist(),"best_epoch":ck["best"]["epoch"],"source_val_accuracy":ck["best"]["val_acc"],"target_gt_used_for_training_or_selection":False})
    aggregate={}
    for alpha in ALPHAS:
        rows=[r for r in runs if r["alpha"]==alpha]; pcs=np.asarray([r["per_class_accuracy"] for r in rows])
        aggregate[str(alpha)]={k:{"mean":float(np.mean([r[k] for r in rows])),"std":float(np.std([r[k] for r in rows]))} for k in ("oa","aa","kappa")}
        aggregate[str(alpha)]["per_class_accuracy"]={"mean":pcs.mean(0).tolist(),"std":pcs.std(0).tolist()}
        aggregate[str(alpha)]["best_epoch"]=[r["best_epoch"] for r in rows]
    indexed={(r["alpha"],r["split"]):r for r in runs}; deltas={}
    for alpha in ALPHAS[1:]:
        rows=[]
        for split in SPLITS:
            a=indexed[(alpha,split)]; b=indexed[(0.0,split)]
            rows.append({"split":split,**{k:a[k]-b[k] for k in ("oa","aa","kappa")},"per_class_accuracy":(np.asarray(a["per_class_accuracy"])-np.asarray(b["per_class_accuracy"])).tolist()})
        deltas[str(alpha)]={"by_split":rows,"mean":{k:float(np.mean([r[k] for r in rows])) for k in ("oa","aa","kappa")},"win_rate":{k:float(np.mean([r[k]>0 for r in rows])) for k in ("oa","aa","kappa")}}
    out={"protocol":{"splits":list(SPLITS),"alphas":list(ALPHAS),"target_gt":"post-hoc evaluation only","backbone":"DCRN_02(x,x)","scene_shift":"global per-band transfer; only strength changed"},"runs":runs,"aggregate":aggregate,"paired_delta_vs_alpha0":deltas}
    (args.root/"summary_3seed.json").write_text(json.dumps(out,indent=2))
    aa=np.asarray(ALPHAS)
    plt.figure(figsize=(7,4.5))
    for k in ("oa","aa","kappa"):
        mu=[aggregate[str(a)][k]["mean"]*100 for a in ALPHAS]; sd=[aggregate[str(a)][k]["std"]*100 for a in ALPHAS]; plt.errorbar(aa,mu,yerr=sd,marker='o',capsize=3,label=k.upper())
    plt.xlabel('Scene Shift strength α'); plt.ylabel('Target metric (%)'); plt.xticks(aa); plt.grid(alpha=.3); plt.legend(); plt.tight_layout(); plt.savefig(args.root/'overall_metrics_vs_alpha_3seed.png',dpi=180); plt.close()
    plt.figure(figsize=(8,5))
    for c in range(7):
        mu=[aggregate[str(a)]["per_class_accuracy"]["mean"][c]*100 for a in ALPHAS]; sd=[aggregate[str(a)]["per_class_accuracy"]["std"][c]*100 for a in ALPHAS]; plt.errorbar(aa,mu,yerr=sd,marker='o',capsize=2,label=f'C{c+1}')
    plt.xlabel('Scene Shift strength α'); plt.ylabel('Per-class accuracy (%)'); plt.xticks(aa); plt.ylim(0,105); plt.grid(alpha=.3); plt.legend(ncol=4); plt.tight_layout(); plt.savefig(args.root/'per_class_accuracy_vs_alpha_3seed.png',dpi=180); plt.close(); print(json.dumps(out,indent=2))

if __name__ == '__main__': main()
