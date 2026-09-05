#!/usr/bin/env bash
set -euo pipefail

epochs=${1:?epochs required}
output_root=${2:?output root required}
root=/home/zhangzj26/scene_semantic_uda
python=/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python
mkdir -p "$output_root/logs"

launch() {
    local gpu=$1 method=$2 split=$3
    local output="$output_root/$method/split_$split"
    mkdir -p "$output"
    local shift=()
    if [[ "$method" == "shift" ]]; then
        shift=(--use-scene-shift)
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$python" -u "$root/train_mamba.py" \
        --split-seed "$split" --optimization-seed 1174 --epochs "$epochs" \
        --batch-size 8 --optimizer sgd --official-recipe --lr-scheduler --lr 0.01 \
        --device cuda:0 --output "$output" "${shift[@]}" \
        > "$output_root/logs/${method}_${split}.log" 2>&1 &
}

launch 0 ce 1174
launch 1 ce 1703
launch 2 ce 2141
launch 3 shift 1174
launch 4 shift 1703
launch 5 shift 2141
wait
