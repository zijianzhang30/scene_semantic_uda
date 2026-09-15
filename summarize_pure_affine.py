import json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
 b=json.loads((HERE/'runs_mluda_official_reproduction_v1/pavia/seed_1256/result.json').read_text()); s=json.loads((HERE/'runs_mluda_official_sceneshift_v1/pavia/seed_1256/result.json').read_text()); r=json.loads((HERE/'runs_mluda_official_pure_affine_no_extra_source_scl_v1/pavia/seed_1256/result.json').read_text())
 lines=['# Pavia seed1256: pure affine SceneShift（无extra source-SCL）','','仅关闭 intensity scaling 与 smooth noise；保留 affine transport、LMMD-shift、shifted pseudo-SCL、gamma=.5、BN与其余protocol。','', '| Method | OA | AA | Kappa |','|---|---:|---:|---:|']
 for n,x in [('Original MLUDA',b),('Current SceneShift',s),('w/o extra source-SCL',json.loads((HERE/'runs_mluda_official_remove_extra_source_scl_v1/pavia/seed_1256/result.json').read_text())),('Pure affine, w/o extra source-SCL',r)]: lines.append(f"| {n} | {x['oa']:.4f} | {x['aa']:.4f} | {x['kappa']:.4f} |")
 lines += ['', '| Class | Original | Current | No extra SCL | Pure affine | Pure - current |','|---:|---:|---:|---:|---:|---:|']
 w=json.loads((HERE/'runs_mluda_official_remove_extra_source_scl_v1/pavia/seed_1256/result.json').read_text())
 for i,(x,y,z,q) in enumerate(zip(b['per_class_accuracy'],s['per_class_accuracy'],w['per_class_accuracy'],r['per_class_accuracy']),1): lines.append(f'| C{i} | {x:.4f} | {y:.4f} | {z:.4f} | {q:.4f} | {q-y:.4f} |')
 lines += ['', '## Deltas vs w/o extra source-SCL','', '```json',json.dumps({k:r[k]-w[k] for k in ['oa','aa','kappa']},indent=2),'```']
 (HERE/'sceneshift_pure_affine_no_extra_sourcescl_seed1256.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
