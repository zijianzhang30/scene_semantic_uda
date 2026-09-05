"""Post-hoc target evaluation for the clean A/B/C/D audit."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
from sklearn import metrics

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]

from UtilsCMS import ILDA  # noqa: E402
from model import DCRNClassifier  # noqa: E402
import utils  # noqa: E402

GROUPS = ("A", "B")
DESCRIPTIONS = {
    "A": "DCRN + CE",
    "B": "DCRN + Scene Shift",
    "C": "GuidedPGC(ILDA) + DCRN + CE",
    "D": "GuidedPGC(ILDA) + DCRN + Scene Shift",
}
# Clean Step-1 matched splits. Historical C/D checkpoints are intentionally
# not scanned by this evaluator.
SPLITS = (1174, 1370, 1417, 1418, 1421, 1535, 1546, 1599, 1610, 1631)
METRICS = ("oa", "aa", "kappa")


def center_patches(cube, centers, width=7):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    output = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        output[i] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return output


def metrics_for(labels, predictions):
    confusion = metrics.confusion_matrix(labels, predictions, labels=np.arange(7))
    per_class = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "oa": float((labels == predictions).mean()),
        "aa": float(per_class.mean()),
        "kappa": float(metrics.cohen_kappa_score(labels, predictions, labels=np.arange(7))),
        "per_class_accuracy": per_class.tolist(),
        "confusion_matrix": confusion.tolist(),
        "prediction_distribution": np.bincount(predictions, minlength=7).tolist(),
    }


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std())}


def aggregate(runs):
    result = {}
    for group in GROUPS:
        selected = [run for run in runs if run["group"] == group]
        result[group] = {metric: mean_std([run[metric] for run in selected]) for metric in METRICS}
        per_class = np.asarray([run["per_class_accuracy"] for run in selected])
        result[group]["per_class_accuracy"] = {
            "mean": per_class.mean(0).tolist(), "std": per_class.std(0).tolist()
        }
    return result


def pairwise_deltas(runs, splits):
    indexed = {(run["split"], run["group"]): run for run in runs}
    comparisons = {"B-A": ("B", "A")}
    output = {}
    for name, (left, right) in comparisons.items():
        rows = []
        for split in splits:
            a, b = indexed[(split, left)], indexed[(split, right)]
            rows.append({
                "split": split,
                **{metric: a[metric] - b[metric] for metric in METRICS},
                "per_class_accuracy": (
                    np.asarray(a["per_class_accuracy"])
                    - np.asarray(b["per_class_accuracy"])
                ).tolist(),
            })
        output[name] = {
            "by_split": rows,
            "aggregate": {
                metric: mean_std([row[metric] for row in rows]) for metric in METRICS
            },
            "win_count": {
                metric: int(sum(row[metric] > 0 for row in rows)) for metric in METRICS
            },
        }
        per_class = np.asarray([row["per_class_accuracy"] for row in rows])
        output[name]["aggregate"]["per_class_accuracy"] = {
            "mean": per_class.mean(0).tolist(), "std": per_class.std(0).tolist()
        }
        output[name]["win_rate"] = {
            metric: float(np.mean([row[metric] > 0 for row in rows])) for metric in METRICS
        }
        output[name]["best_delta"] = {metric: float(max(row[metric] for row in rows)) for metric in METRICS}
        output[name]["worst_delta"] = {metric: float(min(row[metric] for row in rows)) for metric in METRICS}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "runs_clean_audit")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--splits", nargs="+", type=int, default=list(SPLITS))
    args = parser.parse_args()
    device = torch.device(args.device)

    # This is the only script in the clean pipeline that opens target GT.
    source, _source_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    target_gt = hdf5storage.loadmat(
        str(ROOT / "datasets/Houston/Houston18_7gt.mat")
    )["map"]
    centers = np.argwhere(target_gt > 0).astype(np.int64)
    labels = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1

    runs = []
    splits = tuple(args.splits)
    for split in splits:
        for group in GROUPS:
            checkpoint_path = args.root / f"split_{split}" / f"group_{group}" / "best.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            group_target = target
            if checkpoint["use_ilda"]:
                _adapted_source, group_target = ILDA(source, target, 2, 0.009)
            target_x = center_patches(group_target, centers)
            model = DCRNClassifier().to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            predictions = []
            with torch.no_grad():
                for start in range(0, len(target_x), 32):
                    predictions.append(
                        model(torch.from_numpy(target_x[start:start + 32]).to(device))
                        .argmax(1).cpu().numpy()
                    )
            result = metrics_for(labels, np.concatenate(predictions))
            runs.append({
                "split": split,
                "group": group,
                "description": DESCRIPTIONS[group],
                **result,
                "best_epoch": checkpoint["best"]["epoch"],
                "source_val_accuracy": checkpoint["best"]["val_acc"],
                "use_ilda": checkpoint["use_ilda"],
                "use_scene_shift": checkpoint["use_scene_shift"],
                "checkpoint": str(checkpoint_path),
            })

    output = {
        "protocol": {
            "splits": list(splits),
            "groups": {group: DESCRIPTIONS[group] for group in GROUPS},
            "target_gt": "post-hoc evaluation only",
            "checkpoint_selection": "source validation only",
            "backbone": "DCRN_02(x, x)",
            "cross_attention_source_target_interaction": False,
            "disabled_losses": ["foundation", "LMMD", "SCL", "prototype", "semantic", "orth", "modulation"],
        },
        "runs": runs,
        "aggregate": aggregate(runs),
        "pairwise_deltas": pairwise_deltas(runs, splits),
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "summary.json").write_text(json.dumps(output, indent=2))
    with (args.root / "summary.csv").open("w", newline="") as stream:
        fields = ["split", "group", "description", "oa", "aa", "kappa", *[f"class_{i}" for i in range(1, 8)], "best_epoch", "source_val_accuracy", "use_ilda", "use_scene_shift", "checkpoint"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            row = {key: run[key] for key in fields if key in run}
            for cls, value in enumerate(run["per_class_accuracy"], 1):
                row[f"class_{cls}"] = value
            writer.writerow(row)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
