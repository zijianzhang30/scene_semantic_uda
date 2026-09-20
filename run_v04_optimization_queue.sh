#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangzj26/scene_semantic_uda
gpu=$1
shift
for seed in "$@"; do
  out="runs_v04_optimization_stability/seed_$seed"
  mkdir -p "$out"
  # The original GPU1/GPU7 queues finish their current seed then stop here.
  if test -f "runs_v04_optimization_stability/stop_queue_gpu$gpu"; then break; fi
  if test -f "$out/completed.json"; then continue; fi
  CUDA_VISIBLE_DEVICES="$gpu" /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u train_v04_optimization_stability.py --seed "$seed" --out "$out" > "$out/train.log" 2>&1
  /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python summarize_v04_optimization.py
done
