"""Single-checkpoint post-hoc evaluator for clean A/B/C/D checkpoints."""
import argparse
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


def center_patches(cube, centers, width=7):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    output = np.empty((len(centers), cube.shape[-1], width, width), np.float32)
    for i, (row, col) in enumerate(centers):
        output[i] = padded[row:row + width, col:col + width].transpose(2, 0, 1)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    source, _ = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    target_gt = hdf5storage.loadmat(
        str(ROOT / "datasets/Houston/Houston18_7gt.mat")
    )["map"]
    if checkpoint["use_ilda"]:
        _, target = ILDA(source, target, 2, 0.009)
    centers = np.argwhere(target_gt > 0).astype(np.int64)
    x = center_patches(target, centers)
    y = target_gt[centers[:, 0], centers[:, 1]].astype(np.int64) - 1
    model = DCRNClassifier().to(args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), 32):
            predictions.append(model(torch.from_numpy(x[start:start + 32]).to(args.device)).argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    cm = metrics.confusion_matrix(y, predictions, labels=np.arange(7))
    per_class = np.diag(cm) / np.maximum(cm.sum(1), 1)
    print(json.dumps({
        "group": checkpoint.get("group"),
        "use_ilda": checkpoint["use_ilda"],
        "use_scene_shift": checkpoint["use_scene_shift"],
        "oa": float((y == predictions).mean()),
        "aa": float(per_class.mean()),
        "kappa": float(metrics.cohen_kappa_score(y, predictions, labels=np.arange(7))),
        "per_class_accuracy": per_class.tolist(),
        "target_gt_used_for_training_or_selection": checkpoint["target_gt_used_for_training_or_selection"],
    }, indent=2))


if __name__ == "__main__":
    main()
