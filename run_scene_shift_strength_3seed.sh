#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT="$ROOT/runs_scene_shift_strength_3seed"
mkdir -p "$OUT/logs"

launch() {
  local gpu=$1 seed=$2 alpha=$3 group=$4
  local tag=${alpha/./_}
  local dir="$OUT/alpha_${tag}/split_${seed}"
  mkdir -p "$dir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/train.py" \
    --group "$group" --split-seed "$seed" --optimization-seed 1174 \
    --epochs 100 --batch-size 32 --lr 0.002 --device cuda:0 \
    --scene-shift-strength "$alpha" --output "$dir" \
    > "$OUT/logs/alpha_${tag}_split_${seed}.log" 2>&1 &
}

seeds=(1174 1703 2141)
alphas=(0.0 0.4 0.8 1.0)
gpus=(2 3 5 7 0 1)
i=0
for seed in "${seeds[@]}"; do
  for alpha in "${alphas[@]}"; do
    if [[ "$alpha" == "0.0" ]]; then group=A; else group=B; fi
    launch "${gpus[$((i % ${#gpus[@]}))]}" "$seed" "$alpha" "$group"
    i=$((i+1))
  done
done
wait
