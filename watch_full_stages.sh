#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/scene_semantic_uda
PY=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
OUT=$ROOT/runs
while :; do
  all=1
  for split in 1174 1703 2141; do
    rel="$OUT/split_${split}/stage_reliable_alignment/summary.json"
    full="$OUT/split_${split}/stage_full/summary.json"
    if [[ -f "$full" ]]; then continue; fi
    all=0
    if [[ -f "$rel" ]] && ! tmux -L scene has-session -t "scenefull${split}" 2>/dev/null; then
      gpu=0; [[ "$split" == 1703 ]] && gpu=1; [[ "$split" == 2141 ]] && gpu=2
      mkdir -p "$OUT/logs" "$OUT/split_${split}/stage_full"
      tmux -L scene new-session -d -s "scenefull${split}" "cd $ROOT && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=$gpu $PY -u $ROOT/train.py --stage full --split-seed $split --optimization-seed 1174 --epochs 100 --batch-size 32 --device cuda:0 --output $OUT/split_${split}/stage_full > $OUT/logs/split_${split}_full.log 2>&1"
      echo "started full split=$split gpu=$gpu"
    fi
  done
  [[ "$all" == 1 ]] && exit 0
  sleep 20
done
