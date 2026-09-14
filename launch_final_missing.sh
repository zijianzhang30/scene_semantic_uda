#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/zhangzj26
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
cd "$ROOT"

run_ssftt() {
  local dataset="$1" method="$2" seed="$3" gpu="$4"
  mkdir -p "scene_semantic_uda/final_quantitative/backbone_ssftt/${dataset}/seed_${seed}"
  "$PY" scene_semantic_uda/train_ssftt_scene_shift.py \
    --dataset "$dataset" --method "$method" --seed "$seed" \
    --epochs 100 --batch-size 32 --lr 0.002 --device "cuda:${gpu}" \
    --scene_shift_alpha 0.8 --scene_shift_weight 0.5 --scene_shift_clamp auto \
    --output "scene_semantic_uda/final_quantitative/backbone_ssftt/${dataset}/seed_${seed}/${method}" \
    > "scene_semantic_uda/final_quantitative/backbone_ssftt/${dataset}/seed_${seed}_${method}.log" 2>&1
}

run_dcrn() {
  local dataset="$1" method="$2" seed="$3" gpu="$4"
  mkdir -p "scene_semantic_uda/final_quantitative/backbone_cnn/${dataset}/seed_${seed}"
  "$PY" scene_semantic_uda/run_cross_scene_clean.py \
    --dataset "$dataset" --method "$method" --seed "$seed" \
    --epochs 100 --batch-size 32 --lr 0.002 --device "cuda:${gpu}" \
    --output "scene_semantic_uda/final_quantitative/backbone_cnn/${dataset}/seed_${seed}/${method}" \
    > "scene_semantic_uda/final_quantitative/backbone_cnn/${dataset}/seed_${seed}_${method}.log" 2>&1
}

mkdir -p scene_semantic_uda/final_quantitative/backbone_ssftt scene_semantic_uda/final_quantitative/backbone_cnn

# Keep two GPUs occupied without launching duplicate jobs. Each lane runs one
# dataset/seed pair at a time, preserving the already audited recipes.
(
  for dataset in houston pavia shanghai_hangzhou; do
    for seed in 1703 2141; do
      for method in ce scene_shift; do
        run_ssftt "$dataset" "$method" "$seed" 0
      done
    done
  done
) &
P0=$!

(
  for dataset in pavia shanghai_hangzhou; do
    for seed in 1703 2141; do
      for method in ce scene_shift; do
        run_dcrn "$dataset" "$method" "$seed" 2
      done
    done
  done
) &
P1=$!

wait "$P0" "$P1"
touch scene_semantic_uda/final_quantitative/backbone_missing.done
