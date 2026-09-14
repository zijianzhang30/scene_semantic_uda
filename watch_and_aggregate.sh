#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangzj26
while ps -eo args= | grep -E 'train_ssftt_scene_shift.py|run_cross_scene_clean.py|train_full_mluda_scene_shift.py' | grep -v 'watch_and_aggregate.sh' | grep -q .; do
  sleep 60
done
/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python scene_semantic_uda/aggregate_final_quantitative.py > scene_semantic_uda/final_quantitative/aggregate.log 2>&1
touch scene_semantic_uda/final_quantitative/aggregate.done
