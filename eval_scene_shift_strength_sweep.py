"""Post-hoc evaluation and plots for the single-split Scene Shift strength sweep."""
from __future__ import annotations

import json
from pathlib import Path

import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
import sys
sys.path[:0] = [str(ROOT), str(HERE)]
from model import DCRNClassifier  # noqa: E402


ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def center_patches(cube, centers, width=7):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    output = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        output[i] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "runs_scene_shift_strength_sweep_1174")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    target_gt = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18_7gt.mat"))["map"]
    centers = np.argwhere(target_gt > 0).astype(np.int64)
    labels = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    x = center_patches(target, centers)
    runs = []
    for alpha in ALPHAS:
        directory = args.root / f"alpha_{str(alpha).replace('.', '_')}"
        checkpoint = torch.load(directory / "best.pth", map_location="cpu")
        model = DCRNClassifier().to(args.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(x), 32):
                predictions.append(model(torch.from_numpy(x[start:start + 32]).to(args.device)).argmax(1).cpu().numpy())
        predictions = np.concatenate(predictions)
        cm = metrics.confusion_matrix(labels, predictions, labels=np.arange(7))
        per_class = np.diag(cm) / np.maximum(cm.sum(1), 1)
        runs.append({
            "alpha": alpha,
            "oa": float((labels == predictions).mean()),
            "aa": float(per_class.mean()),
            "kappa": float(metrics.cohen_kappa_score(labels, predictions, labels=np.arange(7))),
            "per_class_accuracy": per_class.tolist(),
            "best_epoch": checkpoint["best"]["epoch"],
            "source_val_accuracy": checkpoint["best"]["val_acc"],
            "target_gt_used_for_training_or_selection": checkpoint["target_gt_used_for_training_or_selection"],
            "checkpoint": str(directory / "best.pth"),
        })
    summary = {
        "protocol": {
            "split": 1174,
            "optimization_seed": 1174,
            "alphas": list(ALPHAS),
            "alpha_zero": "pure DCRN + CE path (exact no-shift equivalent)",
            "target_gt": "post-hoc evaluation only",
            "backbone": "DCRN_02(x, x)",
            "scene_shift": "global per-band statistical transfer; only strength changed",
            "other_settings": "7x7 patch, AdamW lr=0.002, batch=32, 100 epochs, source-val checkpoint",
        },
        "runs": runs,
    }
    summary["best_by_metric"] = {
        metric: runs[int(np.argmax([r[metric] for r in runs]))]["alpha"]
        for metric in ("oa", "aa", "kappa")
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2))

    alpha = np.asarray(ALPHAS)
    plt.figure(figsize=(7, 4.5))
    for metric in ("oa", "aa", "kappa"):
        plt.plot(alpha, [r[metric] * 100 for r in runs], marker="o", label=metric.upper())
    plt.xlabel("Scene Shift strength α"); plt.ylabel("Target metric (%)")
    plt.xticks(alpha); plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(args.root / "overall_metrics_vs_alpha.png", dpi=180); plt.close()

    plt.figure(figsize=(8, 5))
    for cls in range(7):
        plt.plot(alpha, [r["per_class_accuracy"][cls] * 100 for r in runs], marker="o", label=f"C{cls + 1}")
    plt.xlabel("Scene Shift strength α"); plt.ylabel("Per-class accuracy (%)")
    plt.xticks(alpha); plt.ylim(0, 105); plt.grid(alpha=0.3); plt.legend(ncol=4); plt.tight_layout()
    plt.savefig(args.root / "per_class_accuracy_vs_alpha.png", dpi=180); plt.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
