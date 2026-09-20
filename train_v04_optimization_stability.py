"""Fixed-data experiment delegating every training step to the original V0.4."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import torch
import train_houston_v04 as original

SEEDS = [1174,1370,1417,1418,1421,1535,1546,1599,1610,1631]
ROOT = Path(__file__).resolve().parent
CACHE = ROOT/'runs_v04_optimization_stability/fixed_data.pth'

def digest(t):
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()

def prepare():
    if CACHE.exists():
        raise FileExistsError('Do not overwrite frozen data: '+str(CACHE))
    original.SEED=1341
    original.seed()
    data=original.load_data()
    original.seed()
    repeated=original.load_data()
    assert all(torch.equal(a,b) for a,b in zip(data,repeated))
    assert len(data[0])==1260 and len(data[2])==53200
    manifest={'data_seed':1341,'counts':torch.bincount(data[1]).tolist(),
              'tensor_sha256':dict(zip(['sx','sy','tx','ty'],map(digest,data))),
              'source_code_sha256':hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
              'optimization_seeds':SEEDS,'target_pool_seed':'optimization_seed+1000 (original persistent generator)',
              'ema':'original epoch reset preserved','optimizer':'fresh at epoch11; momentum not restored'}
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,CACHE)
    CACHE.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest),flush=True)

def run(seed,epochs,out,shared=None):
    data=torch.load(CACHE,map_location='cpu',weights_only=False)
    manifest=json.loads(CACHE.with_suffix('.json').read_text())
    assert [digest(t) for t in data]==list(manifest['tensor_sha256'].values())
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()==manifest['source_code_sha256']
    original.SEED=seed
    # Loading the cached tensors does not consume RNG. The old NumPy data RNG
    # has no downstream training use; Torch state before build() is unchanged.
    original.load_data=lambda: data
    out.mkdir(parents=True,exist_ok=True)
    (out/'protocol.json').write_text(json.dumps({**manifest,'optimization_seed':seed},indent=2))
    print('fixed_data_verified',manifest['tensor_sha256'],'optimization_seed',seed,flush=True)
    history=[]
    best=-1.
    # An observation-only print hook avoids editing/reimplementing train().
    # It records the already-computed epoch metrics and the corresponding model.
    def record(*args,**kwargs):
        nonlocal best
        print(*args,**kwargs)
        if not args or not isinstance(args[0],str) or not args[0].startswith('{"epoch":'):
            return
        row=json.loads(args[0]); history.append(row)
        frame=inspect.currentframe().f_back
        assert frame.f_code is original.train.__code__
        model=frame.f_locals['model']
        payload={'model':model.state_dict(),'epoch':row['epoch'],'metrics':row,
                 'seed':seed,'fixed_data_sha256':manifest['tensor_sha256']}
        if row['oa']>best:
            best=row['oa']
            torch.save({**payload,'selection':'target_oa_debug'},out/'best_target_oa.pth')
        if row['epoch'] in (10,30,60,80,100):
            torch.save(payload,out/f'epoch{row["epoch"]}.pth')
        tmp=out/'history.tmp'
        tmp.write_text(json.dumps(history,indent=2)); tmp.replace(out/'history.json')
    original.print=record
    if shared is None:
        warm_path=original.train('source_only',10,out/'warmup')
    else:
        warm_path=shared
    warm=torch.load(warm_path,map_location=original.DEVICE,weights_only=False)
    # Original train() reseeds, rebuilds model+unused Flow (same RNG consumption),
    # and intentionally does not restore linear-mode optimizer momentum.
    original.train('linear_bridge',epochs,out/'linear_bridge',warm)
    (out/'completed.json').write_text(json.dumps({'seed':seed,'epoch':epochs,'best':max(history,key=lambda r:r['oa'])},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--prepare',action='store_true'); p.add_argument('--seed',type=int); p.add_argument('--epochs',type=int,default=100); p.add_argument('--out',type=Path); p.add_argument('--verification-warmup',type=Path)
    a=p.parse_args()
    if a.prepare: prepare()
    else:
        assert a.seed is not None and a.out is not None
        run(a.seed,a.epochs,a.out,a.verification_warmup)
