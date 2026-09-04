#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT=$ROOT/runs; mkdir -p "$OUT/logs"
GPU=${1:-3}; SPLIT=${2:-1174}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for stage in baseline scene_shift semantic reliable_alignment full; do
 d="$OUT/split_${SPLIT}/stage_${stage}"; mkdir -p "$d"
 if [[ -f "$d/summary.json" ]]; then echo "skip existing $SPLIT $stage"; continue; fi
 CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/train.py" --stage "$stage" --split-seed "$SPLIT" --optimization-seed 1174 --epochs 100 --batch-size 32 --device cuda:0 --output "$d" > "$OUT/logs/split_${SPLIT}_${stage}.log" 2>&1
done
