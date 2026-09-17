"""Summarize completed Houston DAMamba backbone-generalization runs."""
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent / "runs_sceneshiftnet_damamba/houston_raw/three_seed_500epoch"
METHODS = {
    "M0": "DAMamba baseline",
    "M1": "DAMamba + Adaptive SceneShift",
    "M2": "DAMamba + Adaptive SceneShift + Original MCC",
}
SEEDS = (1341, 2024, 3407)


def fmt(values):
    values = np.asarray(values, dtype=float)
    return f"{values.mean():.2f} ± {values.std(ddof=0):.2f}"


def main():
    records = []
    for method, name in METHODS.items():
        for seed in SEEDS:
            run = ROOT / method / f"seed_{seed}"
            metrics = json.loads((run / "metrics.json").read_text())
            history = json.loads((run / "history.json").read_text())
            records.append({"method": method, "method_name": name, "seed": seed,
                            "metrics": metrics, "last": history[-1]})

    with (ROOT / "per_seed_results.csv").open("w", newline="") as f:
        fields = ["method", "method_name", "seed", "OA", "AA", "Kappa",
                  "source_accuracy", "shifted_source_accuracy", "L_mcc",
                  "target_mean_soft_class_distribution"] + [f"class_{i}_accuracy" for i in range(1, 8)]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in records:
            m, last = r["metrics"], r["last"]
            row = {"method": r["method"], "method_name": r["method_name"], "seed": r["seed"],
                   "OA": m["OA"], "AA": m["AA"], "Kappa": m["Kappa"],
                   "source_accuracy": last["source_accuracy"],
                   "shifted_source_accuracy": last["shifted_source_accuracy"],
                   "L_mcc": last["L_mcc"],
                   "target_mean_soft_class_distribution": json.dumps(last["target_mean_soft_class_distribution"])}
            row.update({f"class_{i}_accuracy": x for i, x in enumerate(m["per_class_accuracy"], 1)})
            w.writerow(row)

    summary = []
    for method, name in METHODS.items():
        rs = [r for r in records if r["method"] == method]
        row = {"method": method, "method_name": name}
        for metric in ("OA", "AA", "Kappa"):
            x = np.array([r["metrics"][metric] for r in rs])
            row.update({f"{metric}_mean": x.mean(), f"{metric}_std": x.std(ddof=0), f"{metric}_mean_std": fmt(x)})
        summary.append(row)
    with (ROOT / "mean_std_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    pc_rows = []
    for method, name in METHODS.items():
        rs = [r for r in records if r["method"] == method]
        a = np.array([r["metrics"]["per_class_accuracy"] for r in rs])
        for c in range(7):
            pc_rows.append({"method": method, "method_name": name, "class": c + 1,
                            "accuracy_mean": a[:, c].mean(), "accuracy_std": a[:, c].std(ddof=0),
                            "accuracy_mean_std": fmt(a[:, c])})
    with (ROOT / "per_class_mean_std.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pc_rows[0])); w.writeheader(); w.writerows(pc_rows)

    by = {(r["method"], r["seed"]): r for r in records}
    def deltas(a, b, metric): return [by[(b, s)]["metrics"][metric] - by[(a, s)]["metrics"][metric] for s in SEEDS]
    lines = ["# Houston DAMamba backbone generalization (500 epochs)", "",
             "All values use the final epoch. Standard deviations use population `ddof=0`.", "",
             "| Method | OA mean±std | AA mean±std | Kappa mean±std |", "|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['method']} {row['method_name']} | {row['OA_mean_std']} | {row['AA_mean_std']} | {row['Kappa_mean_std']} |")
    lines += ["", "## Per-seed results", "", "| Method | Seed | OA | AA | Kappa |", "|---|---:|---:|---:|---:|"]
    for r in records:
        m=r["metrics"]; lines.append(f"| {r['method']} | {r['seed']} | {m['OA']:.2f} | {m['AA']:.2f} | {m['Kappa']:.2f} |")
    lines += ["", "## Paired changes", "",
              f"- M1 − M0 OA by seed: {', '.join(f'{x:+.2f}' for x in deltas('M0','M1','OA'))}; mean {np.mean(deltas('M0','M1','OA')):+.2f}.",
              f"- M1 − M0 AA by seed: {', '.join(f'{x:+.2f}' for x in deltas('M0','M1','AA'))}; mean {np.mean(deltas('M0','M1','AA')):+.2f}.",
              f"- M2 − M1 OA by seed: {', '.join(f'{x:+.2f}' for x in deltas('M1','M2','OA'))}; mean {np.mean(deltas('M1','M2','OA')):+.2f}.",
              f"- M2 − M1 AA by seed: {', '.join(f'{x:+.2f}' for x in deltas('M1','M2','AA'))}; mean {np.mean(deltas('M1','M2','AA')):+.2f}.",
              "", "## Interpretation", "",
              "Adaptive SceneShift lowers DAMamba OA at seeds 1341 and 2024 and raises it slightly at seed 3407, while improving AA in all three seeds. It changes the OA/AA tradeoff rather than producing a stable overall gain.",
              "",
              "Original MCC lowers OA in all three seeds. It raises AA at seeds 1341 and 3407 but slightly lowers AA at seed 2024; the mean AA gain over M1 is small.",
              "",
              "Five hundred epochs reduce variance but do not reverse the main conclusion: Adaptive SceneShift improves AA while lowering OA, and MCC further trades OA for weak-class/AA performance. This does not establish the same overall improvement observed with DCRN.",
              "",
              "The DAMamba optimizer is the previously validated SGD recipe (base LR 0.01, momentum 0.9, weight decay 5e-4, backbone/head differential rates), rather than the DCRN Adam optimizer. External patches remain 7×7 and are reflect-padded inside the encoder to the official DAMamba backbone's fixed 12×12 grid."]
    (ROOT / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
