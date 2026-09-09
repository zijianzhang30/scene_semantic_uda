#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT="$ROOT/runs_scene_shift_strength_sweep_1174"
mkdir -p "$OUT/logs"

launch() {
  local gpu=$1 alpha=$2 group=$3
  local tag=${alpha/./_}
  local dir="$OUT/alpha_${tag}"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/train.py" \
    --group "$group" --split-seed 1174 --optimization-seed 1174 \
    --epochs 100 --batch-size 32 --lr 0.002 --device cuda:0 \
    --scene-shift-strength "$alpha" --output "$dir" \
    > "$OUT/logs/alpha_${tag}.log" 2>&1 &
}

# alpha=0 is exactly the no-shift CE path; alpha>0 uses the unchanged
# global Scene Shift formula with only its strength replaced.
launch 2 0.0 A
launch 3 0.2 B
launch 5 0.4 B
launch 7 0.6 B
launch 2 0.8 B
launch 3 1.0 B
wait
