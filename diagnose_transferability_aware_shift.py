"""Diagnostics for class-wise Scene Shift transferability on split 1174.

Target GT is used only for the explicitly labelled oracle/post-hoc analyses.
The semantic safety calculation uses source validation labels only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn import metrics

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
import train as clean  # noqa: E402
from model import DCRNClassifier  # noqa: E402
import utils  # noqa: E402

ALPHAS = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
SPLIT = 1174


def batched_logits(model, x, device, batch_size=32):
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            output.append(model(torch.from_numpy(x[start:start + batch_size]).to(device)).cpu())
    return torch.cat(output, 0)


def class_metrics(labels, predictions):
    cm = metrics.confusion_matrix(labels, predictions, labels=np.arange(7))
    return np.diag(cm) / np.maximum(cm.sum(1), 1)


def safe_corr(x, y):
    return {
        "pearson_r": float(pearsonr(x, y).statistic),
        "pearson_p": float(pearsonr(x, y).pvalue),
        "spearman_r": float(spearmanr(x, y).statistic),
        "spearman_p": float(spearmanr(x, y).pvalue),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=HERE / "runs_scene_shift_strength_sweep_1174")
    parser.add_argument("--output", type=Path, default=HERE / "runs_transferability_aware_1174")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    source, source_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    target_gt = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18_7gt.mat"))["map"]
    source = source.astype(np.float32)
    target = target.astype(np.float32)

    # 1) Oracle curves from already completed matched alpha sweep.
    target_centers = np.argwhere(target_gt > 0).astype(np.int64)
    target_y = target_gt[target_centers[:, 0], target_centers[:, 1]].astype(np.int64) - 1
    oracle_curves = []
    for alpha in ALPHAS:
        tag = f"{float(alpha):.1f}".replace(".", "_")
        checkpoint = torch.load(args.sweep_root / f"alpha_{tag}" / "best.pth", map_location="cpu")
        model = DCRNClassifier().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        target_x = clean.center_patches(target, target_centers)
        predictions = batched_logits(model, target_x, device).argmax(1).numpy()
        oracle_curves.append(class_metrics(target_y, predictions))
    target_curve = np.asarray(oracle_curves)
    oracle_index = target_curve.argmax(0)
    oracle_alpha = ALPHAS[oracle_index]
    target_counts = np.bincount(target_y, minlength=7)
    global08 = target_curve[4]
    best_per_class = target_curve.max(0)
    oracle_aa = float(best_per_class.mean())
    global08_aa = float(global08.mean())
    oracle_oa_upper = float((best_per_class * target_counts).sum() / target_counts.sum())
    global08_oa = float((global08 * target_counts).sum() / target_counts.sum())
    oracle_json = {
        "split": SPLIT,
        "alphas": ALPHAS.tolist(),
        "target_gt": "used only for oracle/post-hoc analysis",
        "per_class_accuracy_curve": target_curve.tolist(),
        "oracle_alpha_per_class": oracle_alpha.tolist(),
        "oracle_accuracy_per_class": best_per_class.tolist(),
        "global_alpha_0_8_accuracy_per_class": global08.tolist(),
        "oracle_aa": oracle_aa,
        "global_alpha_0_8_aa": global08_aa,
        "theoretical_aa_headroom": oracle_aa - global08_aa,
        "oracle_weighted_oa_upper_bound": oracle_oa_upper,
        "global_alpha_0_8_oa": global08_oa,
        "theoretical_weighted_oa_headroom": oracle_oa_upper - global08_oa,
        "target_class_counts": target_counts.tolist(),
    }
    (args.output / "oracle_alpha_per_class.json").write_text(json.dumps(oracle_json, indent=2))

    # 2) Class-wise shift need from labelled source pixels vs global target stats.
    source_flat = source.reshape(-1, source.shape[-1])
    target_flat = target.reshape(-1, target.shape[-1])
    target_mean, target_std = target_flat.mean(0), target_flat.std(0)
    need_raw = []
    class_stats = []
    source_pixel_labels = source_gt.reshape(-1)
    for cls in range(1, 8):
        pixels = source_flat[source_pixel_labels == cls]
        sm, ss = pixels.mean(0), pixels.std(0)
        d = np.mean(np.abs(sm - target_mean) / (target_std + 1e-5) + np.abs(np.log((ss + 1e-5) / (target_std + 1e-5))))
        need_raw.append(float(d))
        class_stats.append({"class": cls, "source_count": int(len(pixels)), "source_mean": sm.tolist(), "source_std": ss.tolist()})
    need_raw = np.asarray(need_raw)
    need = (need_raw - need_raw.min()) / max(float(need_raw.max() - need_raw.min()), 1e-8)
    need_json = {"beta": 1.0, "target_statistics": "global unlabeled target cube", "d_raw": need_raw.tolist(), "need_normalized": need.tolist(), "class_stats": class_stats}
    (args.output / "shift_need_per_class.json").write_text(json.dumps(need_json, indent=2))

    # 3) Source-val semantic safety under each alpha, using only alpha=0 model.
    _, _, val_centers, val_y = clean.source_split(source_gt, SPLIT)
    val_x = clean.center_patches(source, val_centers)
    base_ck = torch.load(args.sweep_root / "alpha_0_0" / "best.pth", map_location="cpu")
    base_model = DCRNClassifier().to(device)
    base_model.load_state_dict(base_ck["model"], strict=True)
    base_model.eval()
    original_logits = batched_logits(base_model, val_x, device)
    original_prob = original_logits.softmax(1)
    original_margin = []
    for cls in val_y:
        row = original_logits[len(original_margin)]
        other = torch.cat([row[:cls], row[cls + 1:]])
        original_margin.append(float(row[cls] - other.max()))
    original_margin = np.asarray(original_margin)
    safety = []
    source_mean, source_std = source_flat.mean(0), source_flat.std(0)
    for alpha_i, alpha in enumerate(ALPHAS):
        torch.manual_seed(20260909 + alpha_i)
        shifted_parts = []
        for start in range(0, len(val_x), 32):
            batch = torch.from_numpy(val_x[start:start + 32]).to(device)
            shifted_parts.append(clean.scene_shift(batch, source_mean, source_std, target_mean, target_std, strength=float(alpha)).cpu())
        shifted_x = torch.cat(shifted_parts, 0).numpy()
        shifted_logits = batched_logits(base_model, shifted_x, device)
        shifted_prob = shifted_logits.softmax(1)
        per_class = []
        for cls in range(7):
            mask = val_y == cls
            orig_conf = original_prob[mask, cls].mean().item()
            shift_conf = shifted_prob[mask, cls].mean().item()
            shifts_margin = []
            for row in shifted_logits[mask]:
                other = torch.cat([row[:cls], row[cls + 1:]])
                shifts_margin.append(float(row[cls] - other.max()))
            shifts_margin = np.asarray(shifts_margin)
            orig_m = original_margin[mask]
            per_class.append({
                "class": cls + 1,
                "alpha": float(alpha),
                "mean_p_true_original": orig_conf,
                "mean_p_true_shifted": shift_conf,
                "confidence_retention": float(shift_conf / (orig_conf + 1e-8)),
                "mean_margin_original": float(orig_m.mean()),
                "mean_margin_shifted": float(shifts_margin.mean()),
                "margin_drop": float((orig_m - shifts_margin).mean()),
            })
        safety.extend(per_class)
    safety_json = {"split": SPLIT, "classifier": "alpha=0 source-only checkpoint", "labels": "source validation only", "records": safety}
    (args.output / "semantic_safety_per_class.json").write_text(json.dumps(safety_json, indent=2))

    # 4) Correlation and compact summary table.
    safety_by_class = {c: {r["alpha"]: r for r in safety if r["class"] == c} for c in range(1, 8)}
    summary_rows = []
    for c in range(1, 8):
        a = float(oracle_alpha[c - 1])
        rec = safety_by_class[c][a]
        gains = target_curve[:, c - 1] - target_curve[0, c - 1]
        summary_rows.append({"class": c, "oracle_alpha": a, "need_raw": float(need_raw[c - 1]), "need": float(need[c - 1]), "confidence_retention_at_oracle": rec["confidence_retention"], "margin_drop_at_oracle": rec["margin_drop"], "target_best_gain_vs_alpha0": float(gains.max()), "target_gain_curve": gains.tolist()})
    summary = {
        "split": SPLIT,
        "rows": summary_rows,
        "correlation": {
            "need_vs_oracle_alpha": safe_corr(need, oracle_alpha),
            "confidence_retention_at_oracle_vs_oracle_alpha": safe_corr(np.asarray([r["confidence_retention_at_oracle"] for r in summary_rows]), oracle_alpha),
            "negative_margin_drop_at_oracle_vs_oracle_alpha": safe_corr(-np.asarray([r["margin_drop_at_oracle"] for r in summary_rows]), oracle_alpha),
        },
    }
    (args.output / "transferability_summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output / "transferability_summary.csv").open("w") as f:
        f.write("class,oracle_alpha,need_raw,need,confidence_retention_at_oracle,margin_drop_at_oracle,target_best_gain_vs_alpha0\n")
        for r in summary_rows:
            f.write(f"{r['class']},{r['oracle_alpha']},{r['need_raw']},{r['need']},{r['confidence_retention_at_oracle']},{r['margin_drop_at_oracle']},{r['target_best_gain_vs_alpha0']}\n")

    # Plots.
    plt.figure(figsize=(6, 5)); plt.scatter(need, oracle_alpha, s=60)
    for c in range(7): plt.annotate(f"C{c+1}", (need[c], oracle_alpha[c]), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Shift Need (normalized)"); plt.ylabel("Oracle α"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(args.output/"need_vs_oracle_alpha.png", dpi=180); plt.close()
    safety_score = np.asarray([r["confidence_retention_at_oracle"] for r in summary_rows])
    plt.figure(figsize=(6, 5)); plt.scatter(safety_score, oracle_alpha, s=60)
    for c in range(7): plt.annotate(f"C{c+1}", (safety_score[c], oracle_alpha[c]), xytext=(4, 4), textcoords="offset points")
    plt.xlabel("Confidence retention at oracle α"); plt.ylabel("Oracle α"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig(args.output/"safety_vs_oracle_alpha.png", dpi=180); plt.close()
    fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)
    axes[0].bar(np.arange(1, 8), need); axes[0].set_ylabel("Need"); axes[0].set_xticks(range(1, 8)); axes[0].grid(axis="y", alpha=.3)
    axes[1].bar(np.arange(1, 8), safety_score); axes[1].set_ylabel("Safety\n(conf retention)"); axes[1].set_xticks(range(1, 8)); axes[1].grid(axis="y", alpha=.3)
    for c in range(7): axes[2].plot(ALPHAS, target_curve[:, c] * 100, marker="o", label=f"C{c+1}")
    axes[2].set_ylabel("Target accuracy (%)"); axes[2].set_xlabel("α"); axes[2].set_xticks(ALPHAS); axes[2].grid(alpha=.3); axes[2].legend(ncol=4)
    fig.tight_layout(); fig.savefig(args.output/"need_safety_target_response.png", dpi=180); plt.close(fig)

    # Human-readable summary.
    lines = ["# Transferability-Aware Scene Shift diagnostic (split 1174)", "", "Target GT is used only for oracle/post-hoc analysis; source validation labels are used for safety.", "", f"Oracle AA: {oracle_aa*100:.2f}% vs global α=0.8 AA {global08_aa*100:.2f}% (theoretical headroom {((oracle_aa-global08_aa)*100):+.2f} pp).", f"Weighted OA upper bound: {oracle_oa_upper*100:.2f}% vs global α=0.8 OA {global08_oa*100:.2f}% (headroom {((oracle_oa_upper-global08_oa)*100):+.2f} pp).", "", "|Class|Oracle α|Need|Safety|Best target gain vs α=0|", "|---:|---:|---:|---:|---:|"]
    for r in summary_rows: lines.append(f"|C{r['class']}|{r['oracle_alpha']:.1f}|{r['need']:.3f}|{r['confidence_retention_at_oracle']:.3f}|{r['target_best_gain_vs_alpha0']*100:+.2f} pp|")
    lines += ["", f"Need vs oracle α Spearman r={summary['correlation']['need_vs_oracle_alpha']['spearman_r']:.3f} (p={summary['correlation']['need_vs_oracle_alpha']['spearman_p']:.3f}).", "", "Interpretation:", "- Oracle class-wise selection quantifies available upper-bound headroom, not a valid method.", "- Need/Safety correlations are diagnostic only; weak correlation means the simple discrepancy is insufficient for automatic α selection.", "- C6 should be checked for low tolerance: its target curve and safety metrics determine whether large shifts are unsafe."]
    (args.output/"transferability_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
