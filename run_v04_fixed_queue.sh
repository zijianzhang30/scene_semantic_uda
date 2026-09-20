#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangzj26/scene_semantic_uda
gpu=$1
shift
base=runs_v04_fixed_10seeds
mkdir -p "$base"
for seed in "$@"; do
  out="$base/seed_$seed"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES="$gpu" /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u train_houston_v04_sourceval.py --seed "$seed" --epochs 100 --out "$out" > "$out/train.log" 2>&1
done
