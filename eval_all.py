"""Post-hoc target evaluation for the three registered current stages."""
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

from model import SemanticSceneUDA  # noqa: E402

SPLITS = (1174, 1703, 2141)
STAGES = ("baseline", "scene_shift", "scene_shift_foundation_reliability")
CORE_METRICS = ("oa", "aa", "kappa")


def patches(cube, centers, width=7):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)))
    output = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        output[i] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return output


def compute_metrics(labels, predictions):
    confusion = metrics.confusion_matrix(labels, predictions, labels=np.arange(7))
    per_class = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "oa": float((labels == predictions).mean()),
        "aa": float(per_class.mean()),
        "kappa": float(metrics.cohen_kappa_score(labels, predictions, labels=np.arange(7))),
        "per_class_accuracy": per_class.tolist(),
        "prediction_distribution": np.bincount(predictions, minlength=7).tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    result = {"mean": array.mean(axis=0), "std": array.std(axis=0)}
    return {
        key: value.tolist() if np.ndim(value) else float(value)
        for key, value in result.items()
    }


def aggregate_runs(runs):
    aggregate = {}
    for stage in STAGES:
        selected = [run for run in runs if run["stage"] == stage]
        aggregate[stage] = {
            metric: mean_std([run[metric] for run in selected]) for metric in CORE_METRICS
        }
        aggregate[stage]["per_class_accuracy"] = mean_std(
            [run["per_class_accuracy"] for run in selected]
        )
    return aggregate


def paired_deltas(runs):
    indexed = {(run["split"], run["stage"]): run for run in runs}
    by_split = []
    for split in SPLITS:
        reference = indexed[(split, "scene_shift")]
        candidate = indexed[(split, "scene_shift_foundation_reliability")]
        by_split.append({
            "split": split,
            **{metric: candidate[metric] - reference[metric] for metric in CORE_METRICS},
            "per_class_accuracy": (
                np.asarray(candidate["per_class_accuracy"])
                - np.asarray(reference["per_class_accuracy"])
            ).tolist(),
        })
    aggregate = {
        metric: mean_std([row[metric] for row in by_split]) for metric in CORE_METRICS
    }
    aggregate["per_class_accuracy"] = mean_std(
        [row["per_class_accuracy"] for row in by_split]
    )
    aggregate["win_count"] = {
        metric: int(sum(row[metric] > 0 for row in by_split)) for metric in CORE_METRICS
    }
    aggregate["all_splits_improved"] = {
        metric: bool(all(row[metric] > 0 for row in by_split)) for metric in CORE_METRICS
    }
    return {"by_split": by_split, "aggregate": aggregate}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE / "runs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)

    # This is the only current-stage script that opens target ground truth.
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    target_gt = hdf5storage.loadmat(
        str(ROOT / "datasets/Houston/Houston18_7gt.mat")
    )["map"]
    centers = np.argwhere(target_gt > 0)
    target_x = patches(target, centers)
    labels = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1

    runs = []
    for split in SPLITS:
        for stage in STAGES:
            checkpoint_path = args.root / f"split_{split}" / f"stage_{stage}" / "best.pth"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing matched checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model = SemanticSceneUDA().to(device)
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            predictions = []
            with torch.no_grad():
                for start in range(0, len(target_x), 32):
                    logits = model(torch.from_numpy(target_x[start:start + 32]).to(device))[3]
                    predictions.append(logits.argmax(dim=1).cpu().numpy())
            result = compute_metrics(labels, np.concatenate(predictions))
            runs.append({
                "split": split,
                "stage": stage,
                **result,
                "best_epoch": checkpoint["best"]["epoch"],
                "source_val_accuracy": checkpoint["best"]["val_acc"],
                "checkpoint": str(checkpoint_path),
            })

    deltas = paired_deltas(runs)
    output = {
        "protocol": {
            "target_gt": "post-hoc evaluation only",
            "checkpoint_selection": "source validation only",
            "splits": list(SPLITS),
            "stages": list(STAGES),
            "delta_reference": "scene_shift",
        },
        "runs": runs,
        "aggregate": aggregate_runs(runs),
        "delta_vs_scene_shift": deltas,
    }
    json_path = args.root / "foundation_reliability_summary.json"
    csv_path = args.root / "foundation_reliability_summary.csv"
    json_path.write_text(json.dumps(output, indent=2))

    delta_by_split = {row["split"]: row for row in deltas["by_split"]}
    with csv_path.open("w", newline="") as stream:
        fieldnames = [
            "split", "stage", "oa", "aa", "kappa",
            *[f"class_{cls}_accuracy" for cls in range(1, 8)],
            "delta_oa_vs_scene_shift", "delta_aa_vs_scene_shift",
            "delta_kappa_vs_scene_shift",
            *[f"delta_class_{cls}_vs_scene_shift" for cls in range(1, 8)],
            "best_epoch", "source_val_accuracy", "checkpoint",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            row = {key: run[key] for key in (
                "split", "stage", "oa", "aa", "kappa", "best_epoch",
                "source_val_accuracy", "checkpoint",
            )}
            for cls, value in enumerate(run["per_class_accuracy"], start=1):
                row[f"class_{cls}_accuracy"] = value
            if run["stage"] == "scene_shift_foundation_reliability":
                delta = delta_by_split[run["split"]]
                for metric in CORE_METRICS:
                    row[f"delta_{metric}_vs_scene_shift"] = delta[metric]
                for cls, value in enumerate(delta["per_class_accuracy"], start=1):
                    row[f"delta_class_{cls}_vs_scene_shift"] = value
            writer.writerow(row)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
