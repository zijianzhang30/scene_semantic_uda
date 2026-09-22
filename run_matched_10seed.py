"""Three static, disjoint workers; never overwrite or silently retry failures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'runs_matched_soft_nointra_mluda_10seed'
SEEDS = [1174,1370,1417,1418,1421,1535,1546,1599,1610,1631]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--worker', type=int, choices=range(3), required=True)
    p.add_argument('--gpu', required=True)
    args = p.parse_args()
    jobs = [(method, seed) for seed in SEEDS for method in ['mluda','soft_nointra'] if (method, seed) != ('soft_nointra',1599)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu)
    for method, seed in jobs[args.worker::3]:
        dest = OUT / method / f'seed_{seed}'
        result_dir = dest / ('official' if method == 'mluda' else 'flow_a')
        if (result_dir / 'results.json').exists():
            continue
        if (dest / 'train.log').exists():
            raise FileExistsError(f'Inspect incomplete run before retry: {dest}')
        dest.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, '-u', str(OUT / 'code_snapshot' / ('train_houston_mluda_matched.py' if method=='mluda' else 'train_houston_scene_shift_reliability_routing.py')), '--seed',str(seed),'--out',str(dest)]
        if method != 'mluda':
            cmd += ['--epochs','100','--routing-assignment','soft','--lambda-intra','0']
        print('START',method,seed,flush=True)
        with (dest / 'train.log').open('x') as log:
            completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        (dest / 'exit.json').write_text(json.dumps({'exit_code':completed.returncode,'command':cmd}))
        if completed.returncode or not (result_dir / 'results.json').exists():
            raise RuntimeError(f'Training failed: {dest}')
        print('DONE',method,seed,flush=True)
    print('WORKER_COMPLETE',args.worker,flush=True)

if __name__ == '__main__':
    main()
