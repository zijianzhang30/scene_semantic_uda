"""Read-only aggregation of checkpoints' saved results; missing runs stay pending."""
import json
from pathlib import Path
H = Path(__file__).resolve().parent
rows=[]
for name, dirname in [
    ('Original MLUDA', 'runs_mluda_official_reproduction_v1'),
    ('Original SceneShift', 'runs_mluda_official_sceneshift_v1'),
    ('No extra source-SCL', 'runs_mluda_official_remove_extra_source_scl_v1'),
    ('Pure affine, both losses', 'runs_mluda_official_pure_affine_no_extra_source_scl_v1'),
    ('Pure affine, LMMD only', 'runs_official_shift_component_lmmd_seed1256_v1'),
    ('Pure affine, shifted-SCL only', 'runs_official_shift_component_scl_seed1256_v1')]:
    f=H/dirname/'pavia/seed_1256/result.json'
    rows.append((name,json.loads(f.read_text()) if f.exists() else None))
lines=['# Pavia seed1256: LMMD / shifted-SCL conflict ablation',
       '', 'Official-aligned, epoch100 final; alpha=0.8, gamma=0.5. New runs use pure affine, no extra source-SCL. All original/extra forwards and BN behavior retained. No tuning.',
       '', '| Method | OA | AA | Kappa | ΔOA vs Original |', '|---|---:|---:|---:|---:|']
base=rows[0][1]
for name,r in rows:
    lines.append(f'| {name} | '+ (' | '.join(f'{r[k]:.4f}' for k in ['oa','aa','kappa'])+f' | {r["oa"]-base["oa"]:+.4f} |' if r else 'Pending | Pending | Pending | Pending |'))
lines += ['', '| Method | C1 | C2 | C3 | C4 | C5 | C6 | C7 |', '|---|'+ '---:|'*7]
for name,r in rows:
    if r: lines.append('| '+name+' | '+' | '.join(f'{x:.4f}' for x in r['per_class_accuracy'])+' |')
diag=H/'runs_official_shift_diagnostic/pavia_pure_affine_component_seed1256/result.json'
if diag.exists():
    d=json.loads(diag.read_text())
    lines += ['', '## Frozen gradient diagnostic', '',
              '| Parameter group | LMMD/orig norm | SCL/orig norm | cos(orig,LMMD) | cos(orig,SCL) | cos(LMMD,SCL) | Negative orig/LMMD | Negative orig/SCL |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for group,v in d['aggregate'].items():
        values=[f'{v[k]["mean"]:.6f}' if v[k] else 'undefined' for k in ['lmmd_ratio','scl_ratio','cos_orig_lmmd','cos_orig_scl','cos_lmmd_scl']]
        neg=[f'{v[k]["negative_batches"]}/{v[k]["defined_batches"]}' if v[k] else 'undefined' for k in ['cos_orig_lmmd','cos_orig_scl']]
        lines.append('| '+group+' | '+' | '.join(values+neg)+' |')
    lines += ['', 'All ratios include gamma=0.5; LMMD also includes the original 0.3×lambda coefficient. Batch-wise ratios/cosines are averaged. Checkpoint and state_dict restored/unchanged; no optimizer step.',
              '', 'Class/sample attribution is local input-view gradient sensitivity, not an additive per-class parameter-gradient decomposition. LMMD/SCL are batch-coupled. Zero-norm parameter cosines are undefined, not zero. '
              'For input-view vectors, LMMD uses raw views whereas SCL uses augmented views; their disjoint-view cosine of zero is structural and must NOT be interpreted as evidence of no class conflict. Source-side SCL sensitivity can be tiny because the dominant gradient is on the shifted counterpart.',
              '', '| Source class | LMMD shifted-view norm | SCL shifted-view norm | Source-view cos(orig,LMMD) | Source-view cos(orig,SCL) |',
              '|---|---:|---:|---:|---:|']
    for c in d['per_class']:
        lines.append('| C'+str(c['source_class'])+' | '+' | '.join(f'{c[k]:.6g}' if c[k] is not None else 'undefined' for k in ['shifted_lmmd_view_norm_mean','shifted_scl_view_norm_mean','cos_orig_lmmd_mean','cos_orig_scl_mean'])+' |')
    lines += ['', 'Source C2 has negative mean source-view LMMD cosine; this is a diagnostic clue, not a causal class-level loss decomposition. Raw and augmented views remain separately represented, with augmentation held fixed.',
              '', 'Diagnostic JSON/CSVs: `runs_official_shift_diagnostic/pavia_pure_affine_component_seed1256/`.']
else: lines += ['', 'Frozen gradient diagnostic: pending.']
if all(r for _,r in rows[-2:]):
    pure=rows[3][1]
    lines += ['', '## Controlled comparison', '']
    for name,r in rows[-2:]:
        lines += [f'- {name}: ΔOA vs pure-affine both-losses = {r["oa"]-pure["oa"]:+.4f} pp; vs Original = {r["oa"]-base["oa"]:+.4f} pp. '
                  f'C2={r["per_class_accuracy"][1]:.4f}, C3={r["per_class_accuracy"][2]:.4f}.']
    lines += ['', 'Interpretation must jointly consider single-factor test OA/AA/per-class changes and frozen gradients. '
              'A negative final-checkpoint gradient cosine is local conflict evidence, not proof of the training-time cause. '
              'Do not expand seeds or tune alpha/gamma automatically.']
else:
    lines += ['', 'Interpretation: await both controlled runs before assigning causality; a single seed does not establish cross-seed behavior.']
(H/'sceneshift_lmmd_scl_conflict_seed1256.md').write_text('\n'.join(lines)+'\n')
