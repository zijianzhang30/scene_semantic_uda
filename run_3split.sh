#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT=$ROOT/runs; mkdir -p "$OUT/logs"
CACHE=$ROOT/cache/hypersigma_fspec_full48_all_target.npz
GPU=${1:-3}; REQUESTED_SPLIT=${2:-}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
[[ -f "$CACHE" ]] || { echo "missing cache: $CACHE; run prepare_foundation_cache.py" >&2; exit 1; }
SPLITS="${REQUESTED_SPLIT:-1174 1703 2141}"
for split in $SPLITS; do
 for stage in baseline scene_shift scene_shift_foundation_reliability; do
  d="$OUT/split_${split}/stage_${stage}"; mkdir -p "$d"
  if [[ -f "$d/summary.json" ]]; then echo "skip existing $split $stage"; continue; fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -u "$ROOT/train.py" --stage "$stage" --split-seed "$split" --optimization-seed 1174 --epochs 100 --batch-size 32 --lambda-target 0.1 --tau-h 0.1 --foundation-cache "$CACHE" --device cuda:0 --output "$d" > "$OUT/logs/split_${split}_${stage}.log" 2>&1
 done
done
