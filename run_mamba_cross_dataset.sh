#!/usr/bin/env bash
set -euo pipefail

# Launch helper for the six requested runs per dataset.  The script is
# intentionally not started automatically: the current container reports no
# usable CUDA device, while mamba_ssm's selective scan requires CUDA.
ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT=${1:-$ROOT/final_quantitative/mamba_cross_dataset}
GPU_LIST=${GPU_LIST:-"0,1,2,3,4,5"}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
mkdir -p "$OUT/logs"

i=0
for dataset in pavia shanghai_hangzhou; do
  for method in ce scene_shift; do
    for seed in 1174 1703 2141; do
      gpu=${GPUS[$((i % ${#GPUS[@]}))]}
      dir="$OUT/$dataset/$method/seed_$seed"
      mkdir -p "$dir"
      args=(--dataset "$dataset" --method "$method" --split-seed "$seed"
            --optimization-seed "$seed" --epochs 500 --batch-size 8 --lr 0.01
            --device cuda:0 --output "$dir")
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/train_mamba_cross_dataset.py" "${args[@]}" \
        > "$OUT/logs/${dataset}_${method}_${seed}.log" 2>&1 &
      i=$((i + 1))
    done
  done
done
wait
