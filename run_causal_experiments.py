"""Run frozen A-G only after the five-epoch real-training audit passes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from audit_causal_pairs import compare

ROOT = Path(__file__).resolve().parent
SEEDS = [1341, 1174, 1370]
ORDER = ['A', 'F', 'B', 'C', 'D', 'E', 'G']


def summarize(destination, status):
    rows = []
    for seed in SEEDS:
        for arm in 'ABCDEFG':
            path = destination / arm / f'seed_{seed}' / 'flow_a/results.json'
            if path.exists():
                try:
                    r = json.loads(path.read_text())
                except json.JSONDecodeError:
                    continue  # Another worker may still be writing its completed result.
                assert r['epoch'] == 100 and r['selection'] == 'fixed_epoch100'
                rows.append({'arm': arm, 'seed': seed, 'OA': r['oa'] * 100,
                             'AA': r['aa'] * 100, 'Kappa': r['kappa'] * 100,
                             **{f'class_{i+1}': v * 100 for i, v in enumerate(r['per_class_accuracy'])},
                             'best_epoch': r['diagnostic_best_epoch'],
                             'best_OA': r['diagnostic_best_oa'] * 100,
                             'epoch100_best_drop_pp': r['epoch100_best_drop'] * 100})
    if rows:
        with (destination / 'fixed_epoch100.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    text = ['## Fresh formal experiment status', '', status, '',
            'Fixed seeds: 1341, 1174, 1370. Values are percentages; drops/deltas are percentage points.', '',
            '| Arm | Seed | OA | AA | Kappa | C1 | C2 | C3 | C4 | C5 | C6 | C7 | Best epoch | Best OA | Drop |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        vals = [row['arm'], str(row['seed'])] + [f'{row[k]:.4f}' for k in ['OA', 'AA', 'Kappa'] + [f'class_{i}' for i in range(1, 8)]]
        vals += [str(row['best_epoch']), f"{row['best_OA']:.4f}", f"{row['epoch100_best_drop_pp']:.4f}"]
        text.append('| ' + ' | '.join(vals) + ' |')
    text += ['', '### Per-arm fixed-epoch summary', '',
             '| Arm | Completed seeds | OA mean +/- SD | AA mean +/- SD | Kappa mean +/- SD |',
             '|---|---:|---:|---:|---:|']
    for arm in 'ABCDEFG':
        group = [r for r in rows if r['arm'] == arm]
        if group:
            cells = [f"{statistics.mean(r[k] for r in group):.4f} +/- {statistics.pstdev(r[k] for r in group):.4f}"
                     for k in ['OA', 'AA', 'Kappa']]
            text.append(f'| {arm} | {len(group)} | ' + ' | '.join(cells) + ' |')
    text += ['', '### Paired fixed-epoch OA effects', '',
             '| Contrast | Completed pairs | Individual deltas | Mean +/- population SD |',
             '|---|---:|---|---|']
    effects = {}
    indexed = {(r['arm'], r['seed']): r for r in rows}
    contrasts = {'SceneShift B-A': {'B': 1, 'A': -1},
                 'Detached Flow F-A': {'F': 1, 'A': -1},
                 'Detached Flow C-B': {'C': 1, 'B': -1},
                 'FM with SceneShift D-C': {'D': 1, 'C': -1},
                 'FM without SceneShift G-F': {'G': 1, 'F': -1},
                 'Reliability E-D': {'E': 1, 'D': -1},
                 'Interaction (D-C)-(G-F)': {'D': 1, 'C': -1, 'G': -1, 'F': 1}}
    for label, weights in contrasts.items():
        delta = {seed: sum(weight * indexed[arm, seed]['OA'] for arm, weight in weights.items())
                 for seed in SEEDS if all((arm, seed) in indexed for arm in weights)}
        effects[label] = delta
        if delta:
            values = list(delta.values())
            text.append(f'| {label} | {len(delta)} | ' + ', '.join(f'{s}: {v:+.4f}' for s, v in delta.items()) +
                        f' | {statistics.mean(values):+.4f} +/- {statistics.pstdev(values):.4f} |')
    if len(rows) == 21:
        text += ['', '### Interpretation within this three-seed Houston experiment', '']
        for label, delta in effects.items():
            vals = list(delta.values())
            if label.startswith('Detached'):
                interpretation = 'identical fixed-epoch OA; see mandatory full step-hash reports for the causal check'
            elif all(v > 0 for v in vals):
                interpretation = 'positive in all three seeds'
            elif all(v < 0 for v in vals):
                interpretation = 'negative in all three seeds'
            else:
                interpretation = 'mixed or zero effects across seeds; no stable positive contribution established'
            text.append(f'- {label}: {interpretation}.')
        text += ['', 'The interaction is descriptive. It does not establish that SceneShift is universally necessary.',
                 'These are clean objective ablations, not a replacement for complete MLUDA baseline results.']
    text += ['', 'Machine-readable results: `' + str(destination / 'fixed_epoch100.csv') + '`.', '']
    report = '\n'.join(text)
    (destination / 'summary.md').write_text(report)
    (destination / 'summary.json').write_text(json.dumps({'status': status, 'rows': rows, 'paired_oa_deltas_pp': effects}, indent=2))
    readme = ROOT / 'README_ITERATION.md'
    before = readme.read_text().split('<!-- FORMAL_RESULTS -->')[0].rstrip()
    readme.write_text(before + '\n\n<!-- FORMAL_RESULTS -->\n' + report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    audit = args.audit.resolve(); out = args.out.resolve()
    gate = compare(audit)
    if not gate['passed']:
        raise RuntimeError('Five-epoch paired audit failed: refusing formal training')
    provenance = json.loads((audit / 'A/seed_1341/flow_a/provenance.json').read_text())
    for path, expected in provenance['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f'Code changed after audit: {path}'
    out.mkdir(parents=True, exist_ok=False)
    (out / 'five_epoch_gate.json').write_text(json.dumps(gate, indent=2))
    (out / 'plan.json').write_text(json.dumps({'seeds': SEEDS, 'order': ORDER, 'epochs': 100,
        'gpu': args.gpu, 'audit': str(audit), 'training_provenance': provenance}, indent=2))
    try:
        for seed in SEEDS:
            for arm in ORDER:
                for path, expected in provenance['files'].items():
                    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, f'Code changed: {path}'
                destination = out / arm / f'seed_{seed}'
                destination.mkdir(parents=True)
                status = f'RUNNING: {arm}, seed {seed}, fixed epoch100. No target-best model selection.'
                summarize(out, status)
                print(status, flush=True)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, PYTHONHASHSEED=str(seed))
                with (destination / 'train.log').open('w') as handle:
                    subprocess.run([sys.executable, '-u', str(ROOT / 'train_houston_flow_transport.py'),
                                    '--ablation', arm, '--seed', str(seed), '--epochs', '100',
                                    '--out', str(destination)], env=env, cwd=ROOT,
                                   stdout=handle, stderr=subprocess.STDOUT, check=True)
                if arm == 'C':
                    hundred_gate = compare(out, seed=seed, epochs=100)
                    (out / f'paired_hash_audit_seed_{seed}.json').write_text(json.dumps(hundred_gate, indent=2))
                    if not hundred_gate['passed']:
                        raise RuntimeError(f'100-epoch null-pair audit failed at seed {seed}; stopping')
        summarize(out, 'COMPLETE: all 21 fixed-epoch100 runs and three full-trajectory paired hash audits passed.')
    except Exception as exc:
        summarize(out, f'STOPPED: {type(exc).__name__}: {exc}. No subsequent runs launched.')
        raise


if __name__ == '__main__':
    main()
