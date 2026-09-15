"""Independent official MLUDA Houston13 -> Houston18 reproduction.

Stages the original MLUDA_hu.py entry point and only instruments seed count,
epoch/result audit, and output capture. The original training/evaluation path
is otherwise left intact.
"""
import ast, csv, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path('/home/zhangzj26/TGRS-MLUDA-2024')
if not REPO.exists():
    REPO = Path('/home/zhangzj26/TGRS_MLUDA-2024')
BASE = HERE / 'runs_mluda_official_houston'
SEED = 1174

def sha(p):
    h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()

def main():
    root=BASE/'houston13_houston18'
    if (root/'STARTED.json').exists(): raise RuntimeError(f'existing run: {root}')
    snap=root/'source_snapshot'; snap.mkdir(parents=True,exist_ok=True)
    for n in ('MLUDA_hu.py','config_Houston.py','UtilsCMS.py','utils.py','net2.py','mmd.py','contrastive_loss.py','Weight.py'):
        shutil.copy2(REPO/n,snap/n)
    src=(snap/'MLUDA_hu.py').read_text()
    marker='from config_Houston import *'
    assert src.count(marker)==1
    src=src.replace(marker, marker + f'\nseeds = [{SEED}]\nnDataSet = 1')
    src=src.split('#################classification map')[0] if '#################classification map' in src else src
    ast.parse(src); (root/'executed_entry.py').write_text(src)
    manifest={'dataset':'houston13_houston18','entry':'MLUDA_hu.py','seed':SEED,'source':'Houston13','target':'Houston18','classes':7,'bands':48,'half_width':3,'patch_size':7,'train_per_class':180,'batch_size':32,'epochs':100,'optimizer':'SGD recreated each epoch','lr_initial':0.01,'lr_schedule':'0.01 / (1 + 10*(epoch-1)/epochs)^0.75','momentum':0.9,'weight_decay':5e-4,'preprocessing':'official load_data_houston (raw ori_data), then official ILDA','ilda':{'pca_n':2,'radius':0.009},'target_pool':'official get_all_data: all target GT>0 pixels','target_loader':'shuffle=True, drop_last=True','evaluation':'official MLUDA_hu.py; target GT post-hoc only','source_code_hash':sha(snap/'MLUDA_hu.py'),'executed_code_hash':sha(root/'executed_entry.py'),'data_hash':{n:sha(REPO/'datasets/Houston'/n) for n in ('Houston13.mat','Houston13_7gt.mat','Houston18.mat','Houston18_7gt.mat')}}
    (root/'config.json').write_text(json.dumps(manifest,indent=2)); (root/'git_commit.txt').write_text(subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True))
    if not __import__('torch').cuda.is_available(): raise RuntimeError('CUDA unavailable')
    (root/'STARTED.json').write_text(json.dumps({'pid':os.getpid(),'time':time.time()}))
    log=root/'training_log.txt'
    env=os.environ.copy(); env['PYTHONPATH']=str(snap)+os.pathsep+env.get('PYTHONPATH','')
    with log.open('w') as f:
        p=subprocess.run([sys.executable,'-u',str(root/'executed_entry.py')],cwd=str(REPO),env=env,stdout=f,stderr=subprocess.STDOUT)
    if p.returncode: raise SystemExit(p.returncode)
    (root/'COMPLETE.json').write_text(json.dumps({'time':time.time(),'seed':SEED}))
    print(log.read_text())

if __name__=='__main__': main()
