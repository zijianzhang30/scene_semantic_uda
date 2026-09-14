#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
cd "$ROOT"
OUT=scene_semantic_uda/final_quantitative/mluda_multiseed
mkdir -p "$OUT"
SEEDS=(1370 1417 1418 1421 1535 1546 1599 1610 1631)
for seed in "${SEEDS[@]}"; do
  for method in full_mluda mluda_dual_counterpart; do
    dir="$OUT/seed_${seed}/${method}"
    mkdir -p "$dir"
    "$PY" scene_semantic_uda/train_full_mluda_scene_shift.py \
      --method "$method" --split-seed "$seed" --optimization-seed "$seed" \
      --epochs 100 --device cuda:6 --scene-shift-alpha 0.8 \
      --dual-gamma 0.5 --output "$dir" \
      > "$OUT/seed_${seed}_${method}.log" 2>&1
  done
done
touch "$OUT/mluda_10seed_missing.done"
