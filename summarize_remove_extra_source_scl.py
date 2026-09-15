import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 root=HERE/'runs_mluda_official_remove_extra_source_scl_v1/pavia/seed_1256'; b=json.loads((HERE/'runs_mluda_official_reproduction_v1/pavia/seed_1256/result.json').read_text()); s=json.loads((HERE/'runs_mluda_official_sceneshift_v1/pavia/seed_1256/result.json').read_text()); r=json.loads((root/'result.json').read_text())
 lines=['# Official-aligned Pavia seed1256: remove duplicated extra source-SCL','','Only extra SceneShift source-SCL was removed; LMMD-shift, shifted pseudo-SCL and every other protocol remained unchanged.','', '| Method | OA | AA | Kappa |','|---|---:|---:|---:|']
 for n,x in [('Original MLUDA',b),('Current SceneShift',s),('SceneShift w/o extra source-SCL',r)]: lines.append(f"| {n} | {x['oa']:.4f} | {x['aa']:.4f} | {x['kappa']:.4f} |")
 lines += ['', '| Class | Original | Current SceneShift | No extra source-SCL | Δ vs current |','|---:|---:|---:|---:|---:|']
 for i,(x,y,z) in enumerate(zip(b['per_class_accuracy'],s['per_class_accuracy'],r['per_class_accuracy']),1): lines.append(f'| C{i} | {x:.4f} | {y:.4f} | {z:.4f} | {z-y:.4f} |')
 lines += ['', '## Deltas','', '```json',json.dumps({'vs_original':{k:r[k]-b[k] for k in ['oa','aa','kappa']},'vs_current':{k:r[k]-s[k] for k in ['oa','aa','kappa']}},indent=2),'```']
 (HERE/'sceneshift_remove_extra_source_scl_seed1256.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__': main()
