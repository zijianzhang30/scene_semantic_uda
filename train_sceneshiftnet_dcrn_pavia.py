"""Own-model SceneShiftNet-DCRN runner for PaviaU -> PaviaC."""
import argparse
import sys
from pathlib import Path

import scipy.io as sio
from sklearn import preprocessing

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(ROOT))

from sceneshiftnet_dcrn_train import DatasetSpec, parse_bool, run

SPEC = DatasetSpec(
    name="pavia_u_pavia_c", source="Pavia University", target="Pavia Centre",
    bands=102, classes=7, patch_size=11, ilda_pca=2, ilda_radius=0.00009,
    normalization="sklearn.preprocessing.scale independently on each domain and band",
)


def load_cubes(normalization):
    paths = [
        ROOT / "datasets/Pavia/paviaU.mat", ROOT / "datasets/Pavia/paviaU_gt_7.mat",
        ROOT / "datasets/Pavia/pavia.mat", ROOT / "datasets/Pavia/pavia_gt_7.mat",
    ]
    source = sio.loadmat(paths[0])["paviaU"]
    source_gt = sio.loadmat(paths[1])["paviaU_gt_7"]
    target = sio.loadmat(paths[2])["pavia"]
    target_gt = sio.loadmat(paths[3])["pavia_gt_7"]
    if normalization == "loader":
        source = preprocessing.scale(source.reshape(-1, 102)).reshape(source.shape)
        target = preprocessing.scale(target.reshape(-1, 102)).reshape(target.shape)
    return source, source_gt, target, target_gt, paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--method", choices=("baseline", "sceneshift"), default="sceneshift")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-per-class", type=int, default=180)
    parser.add_argument("--use-ilda", type=parse_bool, default=False)
    parser.add_argument("--normalization", choices=("loader", "none"), default="none")
    parser.add_argument("--lambda-margin", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args, SPEC, load_cubes, Path(__file__).parent / "runs_sceneshiftnet_dcrn/pavia", __file__)
