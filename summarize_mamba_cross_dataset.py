"""Aggregate completed cross-dataset clean Mamba runs."""
import csv, json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "final_quantitative/mamba_cross_dataset"
rows=[]
for dataset in ("pavia", "shanghai_hangzhou"):
    for method in ("ce", "scene_shift"):
        for seed in (1174,1703,2141):
            f=ROOT/dataset/method/f"seed_{seed}"/"result.json"
            if f.exists():
                d=json.loads(f.read_text()); rows.append({"dataset":dataset,"method":method,"seed":seed,**d})
summary={}
for dataset in ("pavia","shanghai_hangzhou"):
    summary[dataset]={}
    for method in ("ce","scene_shift"):
        a=[r for r in rows if r["dataset"]==dataset and r["method"]==method]
        summary[dataset][method]={k:{"mean":statistics.mean([100*r[k] for r in a]),"std":statistics.pstdev([100*r[k] for r in a])} for k in ("oa","aa","kappa")} if a else None
    if summary[dataset].get("ce") and summary[dataset].get("scene_shift"):
        summary[dataset]["delta"]={k:summary[dataset]["scene_shift"][k]["mean"]-summary[dataset]["ce"][k]["mean"] for k in ("oa","aa","kappa")}
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2))
with (ROOT/"per_seed.csv").open("w",newline="") as f:
    w=csv.writer(f); w.writerow(["dataset","method","seed","oa","aa","kappa","best_epoch"])
    for r in rows: w.writerow([r["dataset"],r["method"],r["seed"],100*r["oa"],100*r["aa"],100*r["kappa"],r.get("best_epoch")])
lines=["# Cross-dataset clean Mamba + SceneShift", "", "MambaFeature architecture and Houston recipe are unchanged; Pavia/Shanghai use 12×12 symmetric patches, z-score normalization and no clamp.", "", "| Dataset | Method | OA | AA | Kappa | Wins |", "|---|---|---:|---:|---:|---:|"]
for d in ("pavia","shanghai_hangzhou"):
    for m in ("ce","scene_shift"):
        s=summary[d].get(m)
        if s: lines.append(f"| {d} | {m} | {s['oa']['mean']:.2f}±{s['oa']['std']:.2f} | {s['aa']['mean']:.2f}±{s['aa']['std']:.2f} | {s['kappa']['mean']:.2f}±{s['kappa']['std']:.2f} | TODO |")
        else: lines.append(f"| {d} | {m} | TODO | TODO | TODO | TODO |")
lines += ["", "## Per-seed OA/AA/Kappa (%)", ""]
for r in rows: lines.append(f"- {r['dataset']} / {r['method']} / {r['seed']}: {100*r['oa']:.2f} / {100*r['aa']:.2f} / {100*r['kappa']:.2f} (best epoch {r.get('best_epoch','TODO')})")
(ROOT/"mamba_cross_dataset_summary.md").write_text("\n".join(lines)+"\n")
print((ROOT/"mamba_cross_dataset_summary.md").resolve())
