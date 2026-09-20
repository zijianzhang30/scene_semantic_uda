#!/usr/bin/env bash
set -e
for seed in 1174 1370 1417 1418 1421 1535 1546 1599 1610 1631; do
  out="runs_v04_10seeds/seed_${seed}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=7 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u train_houston_v04.py --mode warmup --epochs 10 --seed "$seed" --out "$out" > "$out/warmup.log" 2>&1
  CUDA_VISIBLE_DEVICES=7 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u train_houston_v04.py --mode linear_bridge --epochs 100 --seed "$seed" --out "$out" > "$out/linear.log" 2>&1
done
