"""Own-model SceneShiftNet-DCRN runner for Shanghai -> Hangzhou."""
import argparse
import sys
from pathlib import Path

import scipy.io as sio
from sklearn import preprocessing

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(ROOT))

from sceneshiftnet_dcrn_train import DatasetSpec, parse_bool, run

SPEC = DatasetSpec(
    name="shanghai_hangzhou", source="Shanghai", target="Hangzhou",
    bands=198, classes=3, patch_size=1, ilda_pca=2, ilda_radius=0.00009,
    normalization="sklearn.preprocessing.scale independently on each domain and band",
)


def load_cubes(normalization):
    path = ROOT / "datasets/Shanghai-Hangzhou/DataCube.mat"
    values = sio.loadmat(path)
    source, target = values["DataCube1"], values["DataCube2"]
    if normalization == "loader":
        source = preprocessing.scale(source.reshape(-1, 198)).reshape(source.shape)
        target = preprocessing.scale(target.reshape(-1, 198)).reshape(target.shape)
    return source, values["gt1"], target, values["gt2"], [path]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1341)
    parser.add_argument("--method", choices=("baseline", "sceneshift"), default="sceneshift")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-per-class", type=int, default=180)
    parser.add_argument("--use-ilda", type=parse_bool, default=False)
    parser.add_argument("--normalization", choices=("loader", "none"), default="none")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(args, SPEC, load_cubes, Path(__file__).parent / "runs_sceneshiftnet_dcrn/shanghai_hangzhou", __file__)
