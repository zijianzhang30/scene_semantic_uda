"""Strict official-MLUDA-protocol preprocessing/sampling/eval own-model run."""
import argparse
from pathlib import Path

from train_sceneshiftnet_dcrn_pavia import SPEC, load_cubes
from sceneshiftnet_dcrn_train import parse_bool, run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("baseline", "sceneshift"), required=True)
    parser.add_argument("--seed", type=int, default=1622)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--source-per-class", type=int, default=180)
    parser.add_argument("--use-ilda", type=parse_bool, default=True)
    parser.add_argument("--normalization", choices=("loader",), default="loader")
    parser.add_argument("--protocol-matched", action="store_true", default=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    run(
        args, SPEC, load_cubes,
        Path(__file__).parent / "runs_sceneshiftnet_dcrn/pavia_protocol_matched",
        __file__,
    )
