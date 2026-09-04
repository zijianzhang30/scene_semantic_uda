#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/zhangzj26/scene_semantic_uda
mkdir -p /tmp/scene_smoke
"/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python" "$ROOT/train.py" --stage full --split-seed 1174 --optimization-seed 1174 --epochs 1 --batch-size 32 --device cpu --output /tmp/scene_smoke
