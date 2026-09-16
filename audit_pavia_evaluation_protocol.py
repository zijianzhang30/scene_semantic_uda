"""Audit Pavia evaluation and re-evaluate three checkpoints on fixed target pixels."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from sklearn import metrics
from sklearn import preprocessing
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
MLUDA = Path("/home/zhangzj26/TGRS_MLUDA-2024")
SNAPSHOT = HERE / "runs_mluda_official_sceneshift_v1/pavia/source_snapshot"
sys.path.insert(0, str(SNAPSHOT))
import utils as official_utils  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
from net2 import DSANSS  # noqa: E402
sys.path.pop(0)
sys.path.insert(0, str(HERE))
from models.sceneshift_net_dcrn import SceneShiftNetDCRN  # noqa: E402


def scores(labels, predictions, classes=7):
    cm = metrics.confusion_matrix(labels, predictions, labels=np.arange(classes))
    per_class = np.divide(np.diag(cm), cm.sum(1), out=np.zeros(classes), where=cm.sum(1) != 0)
    return {
        "OA": float((labels == predictions).mean() * 100),
        "AA": float(per_class.mean() * 100),
        "Kappa": float(metrics.cohen_kappa_score(labels, predictions) * 100),
        "per_class_accuracy": (per_class * 100).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def patch_tensor(cube, rows, cols, half=5):
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    values = np.stack([
        padded[row:row + 2 * half + 1, col:col + 2 * half + 1].transpose(2, 0, 1)
        for row, col in zip(rows, cols)
    ]).astype("float32")
    return torch.from_numpy(values)


def load_fixed_target():
    prov = np.load(HERE / "organized_runs/mluda_baseline/runs_mluda_official_reproduction_v1/pavia/seed_1622/evaluation_provenance.npz")
    n = len(prov["prediction"])
    # Provenance stores coordinates in the padded (11x11) cube used by the
    # official helper; convert back to image-center coordinates.
    order = prov["target_order"][:n]
    rows = prov["target_rows"][order] - 5
    cols = prov["target_cols"][order] - 5
    labels = prov["labels"].astype("int64")
    return prov, rows, cols, labels


def official_eval(device, target_rows, target_cols):
    source, source_gt = official_utils.load_data_pavia(
        str(MLUDA / "datasets/Pavia/paviaU.mat"), str(MLUDA / "datasets/Pavia/paviaU_gt_7.mat")
    )
    target, _ = official_utils.load_data_pavia(
        str(MLUDA / "datasets/Pavia/pavia.mat"), str(MLUDA / "datasets/Pavia/pavia_gt_7.mat")
    )
    # Above first call gives source and target variables from separate files;
    # apply exact official ILDA to both domains.
    source, target = ILDA(source, target, 2, 0.00009)
    checkpoint = torch.load(
        HERE / "organized_runs/mluda_baseline/runs_mluda_official_reproduction_v1/pavia/seed_1622/final_epoch100.pth",
        map_location=device,
    )
    model = DSANSS(102, 11, 7).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    provenance = np.load(HERE / "organized_runs/mluda_baseline/runs_mluda_official_reproduction_v1/pavia/seed_1622/evaluation_provenance.npz")
    source_ref = torch.from_numpy(provenance["source_reference"]).to(device)
    target_x = patch_tensor(target, target_rows, target_cols).to(device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(target_x), 32):
            output = model(source_ref, target_x[start:start + 32])[8]
            predictions.append(output.argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    return predictions


def own_eval(checkpoint_path, cube, rows, cols, bands, classes, device):
    model = SceneShiftNetDCRN(bands, classes, 11).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    inputs = patch_tensor(cube, rows, cols)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), 32):
            predictions.append(model(inputs[start:start + 32].to(device))[1].argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def processed_pavia_target():
    source, _ = official_utils.load_data_pavia(
        str(MLUDA / "datasets/Pavia/paviaU.mat"), str(MLUDA / "datasets/Pavia/paviaU_gt_7.mat")
    )
    target, _ = official_utils.load_data_pavia(
        str(MLUDA / "datasets/Pavia/pavia.mat"), str(MLUDA / "datasets/Pavia/pavia_gt_7.mat")
    )
    _source, target = ILDA(source, target, 2, 0.00009)
    return target.astype("float32")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    provenance, rows, cols, labels = load_fixed_target()
    target_processed = processed_pavia_target()
    official_prediction = official_eval(device, rows, cols)
    stored_prediction = provenance["prediction"]
    official_match = float((official_prediction == stored_prediction).mean() * 100)

    methods = []
    methods.append((
        "Original MLUDA official checkpoint",
        "organized_runs/mluda_baseline/.../seed_1622/final_epoch100.pth",
        official_prediction,
    ))
    for method in ("baseline", "sceneshift"):
        checkpoint = HERE / f"runs_sceneshiftnet_dcrn/pavia_protocol_matched/{method}/seed_1622/final.pth"
        prediction = own_eval(checkpoint, target_processed, rows, cols, 102, 7, device)
        methods.append((method, str(checkpoint), prediction))

    output = HERE / "runs_sceneshiftnet_dcrn/pavia/evaluation_audit"
    output.mkdir(parents=True, exist_ok=True)
    rows_out = []
    for method, checkpoint, prediction in methods:
        result = scores(labels, prediction)
        rows_out.append({"Method": method, "Checkpoint": checkpoint, "Eval samples": len(labels), **{k: result[k] for k in ("OA", "AA", "Kappa", "per_class_accuracy")}})
    (output / "unified_evaluation.json").write_text(json.dumps({"rows": rows_out, "official_checkpoint_match_percent": official_match}, indent=2))
    with (output / "unified_evaluation.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Method", "Checkpoint", "Eval samples", "OA", "AA", "Kappa", "per-class accuracy"])
        for row in rows_out:
            writer.writerow([row["Method"], row["Checkpoint"], row["Eval samples"], row["OA"], row["AA"], row["Kappa"], row["per_class_accuracy"]])
    print("Fixed target samples:", len(labels), "of eligible 39355")
    print("Official checkpoint prediction match to stored provenance (%):", official_match)
    for row in rows_out:
        print(row)


if __name__ == "__main__":
    main()
