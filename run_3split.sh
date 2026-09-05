#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT=$ROOT/runs_clean_10seed
GPU=${1:-3}
REQUESTED_SPLIT=${2:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SPLITS="${REQUESTED_SPLIT:-1174 1370 1417 1418 1421 1535 1546 1599 1610 1631}"
for split in $SPLITS; do
  for group in A B; do
    dir="$OUT/split_${split}/group_${group}"
    mkdir -p "$dir" "$OUT/logs"
    if [[ -f "$dir/summary.json" ]]; then
      echo "skip existing split=$split group=$group"
      continue
    fi
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/train.py" \
      --group "$group" --split-seed "$split" --optimization-seed 1174 \
      --epochs 100 --batch-size 32 --lr 0.002 --device cuda:0 \
      --output "$dir" > "$OUT/logs/split_${split}_group_${group}.log" 2>&1
  done
done
