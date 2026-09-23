"""Three independent, GPU-pinned seed queues with the same causal audit gates."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from audit_causal_pairs import compare
from run_causal_experiments import ORDER, ROOT, summarize

GPU_BY_SEED = {1341: '1', 1174: '4', 1370: '5'}


def alive(pid, start):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == start
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt-pid', required=True, type=int)
    args = parser.parse_args()
    out = ROOT / 'runs_causal_ag_20260923'
    lock = (out / 'parallel_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gate = compare(ROOT / 'runs_causal_audit_v2_20260923')
    assert gate['passed'], 'Five-epoch audit failed'
    plan = json.loads((out / 'plan.json').read_text())
    provenance = plan['training_provenance']['files']

    def verify_code():
        for path, expected in provenance.items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f'Code changed: {path}'

    verify_code()
    cmdline = Path(f'/proc/{args.adopt_pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    assert 'train_houston_flow_transport.py --ablation A --seed 1341 --epochs 100' in cmdline
    start = Path(f'/proc/{args.adopt_pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    states = {seed: {'index': 0, 'gpu': gpu, 'process': None, 'handle': None, 'pid': None}
              for seed, gpu in GPU_BY_SEED.items()}
    states[1341].update(pid=args.adopt_pid, adopted_start=start)
    plan['parallel_gpu_by_seed'] = GPU_BY_SEED
    plan['adopted_training_pid'] = args.adopt_pid
    plan['scheduler'] = str(Path(__file__).resolve())
    (out / 'parallel_plan.json').write_text(json.dumps(plan, indent=2))
    failed = None

    def update():
        public = {seed: {'gpu': s['gpu'], 'pid': s['pid'],
                         'arm': ORDER[s['index']] if s['index'] < len(ORDER) else 'complete'}
                  for seed, s in states.items()}
        message = ('STOPPED NEW LAUNCHES: ' + failed) if failed else 'RUNNING in parallel: ' + '; '.join(
            f"seed {seed}: {s['arm']} on GPU {s['gpu']}" for seed, s in public.items())
        (out / 'parallel_status.json').write_text(json.dumps({'status': message, 'workers': public}, indent=2))
        summarize(out, message)
        print(message, flush=True)

    update()
    while True:
        changed = False
        for seed, state in states.items():
            index = state['index']
            if index == len(ORDER):
                continue
            arm = ORDER[index]
            if state['pid'] is not None:
                if state['process'] is None:
                    finished = not alive(state['pid'], state['adopted_start'])
                    returncode = None
                else:
                    returncode = state['process'].poll()
                    finished = returncode is not None
                if not finished:
                    continue
                if state['handle'] is not None:
                    state['handle'].close()
                state['pid'] = None
                state['process'] = None
                state['handle'] = None
                changed = True
                try:
                    assert returncode in (None, 0), f'exit code {returncode}'
                    result = json.loads((out / arm / f'seed_{seed}/flow_a/results.json').read_text())
                    assert result['epoch'] == 100 and result['selection'] == 'fixed_epoch100'
                    if arm == 'C':
                        pair_gate = compare(out, seed=seed, epochs=100)
                        (out / f'paired_hash_audit_seed_{seed}.json').write_text(json.dumps(pair_gate, indent=2))
                        assert pair_gate['passed'], '100-epoch pair hash audit failed'
                    state['index'] += 1
                    print(f'FINISHED {arm} seed {seed} on GPU {state["gpu"]}', flush=True)
                except Exception as exc:
                    failed = f'{arm} seed {seed}: {type(exc).__name__}: {exc}'
                    state['index'] = len(ORDER)
                continue
            if failed:
                continue
            try:
                verify_code()
                destination = out / arm / f'seed_{seed}'
                destination.mkdir(parents=True, exist_ok=False)
                state['handle'] = (destination / 'train.log').open('w')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=state['gpu'], PYTHONHASHSEED=str(seed))
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'train_houston_flow_transport.py'),
                    '--ablation', arm, '--seed', str(seed), '--epochs', '100', '--out', str(destination)],
                    env=env, cwd=ROOT, stdout=state['handle'], stderr=subprocess.STDOUT)
                state['process'] = process
                state['pid'] = process.pid
                changed = True
                print(f'STARTED {arm} seed {seed} GPU {state["gpu"]} PID {process.pid}', flush=True)
            except Exception as exc:
                failed = f'Launch {arm} seed {seed}: {type(exc).__name__}: {exc}'
                changed = True
        if changed:
            update()
        if all(s['index'] == len(ORDER) for s in states.values()) and not failed:
            summarize(out, 'COMPLETE: all 21 fixed-epoch100 runs and three full-trajectory paired hash audits passed (three GPUs).')
            (out / 'parallel_status.json').write_text(json.dumps({'status': 'COMPLETE', 'passed': True}, indent=2))
            break
        if failed and all(s['pid'] is None for s in states.values()):
            update()
            raise RuntimeError(failed)
        time.sleep(5)


if __name__ == '__main__':
    main()
