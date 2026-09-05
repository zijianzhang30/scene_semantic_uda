"""Cache frozen Stage-1 HyperSIGMA F_spec without reading target ground truth."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch

ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
HERE = Path(__file__).resolve().parent
HYPERSIGMA = ROOT / "third_party/HyperSIGMA/ImageClassification"
# This directory also contains model.py; exclude it while importing the
# HyperSIGMA namespace package named model.
sys.path = [entry for entry in sys.path if Path(entry).resolve() != HERE]
sys.path[:0] = [str(HYPERSIGMA), str(ROOT)]

from model.ss_fusion_cls import SSFusionFramework  # noqa: E402
import utils  # noqa: E402

IMG_SIZE = 33
HALF = IMG_SIZE // 2
DEFAULT_CHECKPOINT = Path(
    "/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48/stage1_best.pth"
)
DEFAULT_OUTPUT = HERE / "cache" / "hypersigma_fspec_full48_all_target.npz"


def all_centers(shape):
    return np.argwhere(np.ones(shape, dtype=bool)).astype(np.int64)


def extract_fspec(model, cube, centers, device, batch_size):
    padded = np.pad(cube, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="constant")
    features = []
    with torch.inference_mode():
        for start in range(0, len(centers), batch_size):
            batch_centers = centers[start:start + batch_size]
            patches = np.empty(
                (len(batch_centers), cube.shape[-1], IMG_SIZE, IMG_SIZE), dtype=np.float32
            )
            for i, (row, col) in enumerate(batch_centers):
                patches[i] = padded[
                    row:row + IMG_SIZE, col:col + IMG_SIZE
                ].transpose(2, 0, 1)
            spectral = model.spec_encoder(torch.from_numpy(patches).to(device))[0]
            features.append(spectral.mean(dim=1).cpu().numpy().astype(np.float32))
            if start == 0 or (start // batch_size) % 100 == 0:
                print(f"cached {min(start + batch_size, len(centers))}/{len(centers)}", flush=True)
    return np.concatenate(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    source, source_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    # Deliberately load target imagery directly. No target label path appears in this script.
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    source_centers = np.argwhere(source_gt > 0).astype(np.int64)
    target_centers = all_centers(target.shape[:2])

    teacher = SSFusionFramework(
        img_size=IMG_SIZE, in_channels=48, patch_size=2, classes=7, model_size="base"
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    teacher.load_state_dict(checkpoint["model"], strict=True)
    teacher.to(device).eval()
    teacher.requires_grad_(False)
    if teacher.training or any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("HyperSIGMA teacher must be frozen and in eval mode")

    print(
        f"teacher={args.checkpoint} source={len(source_centers)} "
        f"target_unlabeled_full_image={len(target_centers)} target_gt_used=False",
        flush=True,
    )
    source_features = extract_fspec(
        teacher, source, source_centers, device, args.batch_size
    )
    target_features = extract_fspec(
        teacher, target, target_centers, device, args.batch_size
    )
    if not np.isfinite(source_features).all() or not np.isfinite(target_features).all():
        raise RuntimeError("HyperSIGMA F_spec contains NaN/Inf")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        source_centers=source_centers,
        source_fspec=source_features,
        target_centers=target_centers,
        target_fspec=target_features,
        teacher_checkpoint=np.asarray(str(args.checkpoint)),
        feature_name=np.asarray("F_spec_mean_token"),
        patch_size=np.asarray(IMG_SIZE, dtype=np.int64),
        target_sample_universe=np.asarray("complete_image_row_major"),
        target_gt_used_for_cache=np.asarray(False),
    )
    print(
        f"saved={args.output} source_fspec={source_features.shape} "
        f"target_fspec={target_features.shape}",
        flush=True,
    )


if __name__ == "__main__":
    main()
