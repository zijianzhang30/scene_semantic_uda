"""Numerical, recipe, and patch-extraction audit for the clean Mamba path."""
import json, sys
from pathlib import Path
import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/home/zhangzj26/TGRS_MLUDA-2024')
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
import importlib.util
spec = importlib.util.spec_from_file_location('hsi_utils', ROOT/'utils.py')
utils = importlib.util.module_from_spec(spec); spec.loader.exec_module(utils)

def patch(cube, row, col, width=12, mode='symmetric'):
    pad = width // 2 + 1 if mode == 'symmetric' else width // 2
    p = np.pad(cube, ((pad,pad),(pad,pad),(0,0)), mode=mode)
    start_r = row + (pad - width // 2)
    start_c = col + (pad - width // 2)
    return p[start_r:start_r+width, start_c:start_c+width].transpose(2,0,1)

def main():
    s,_=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat'))
    t=hdf5storage.loadmat(str(ROOT/'datasets/Houston/Houston18.mat'))['ori_data']
    sf=s.reshape(-1,48); tf=t.reshape(-1,48)
    sm,ss,tm,ts=sf.mean(0),sf.std(0),tf.mean(0),tf.std(0)
    smt=torch.tensor(sm)[None,:,None,None]; sst=torch.tensor(ss)[None,:,None,None]
    tmt=torch.tensor(tm)[None,:,None,None]; tst=torch.tensor(ts)[None,:,None,None]
    # Deterministic perturbation-free affine part; noise is sampled once with a
    # fixed seed only to quantify the exact current clamp behavior.
    torch.manual_seed(1174)
    x=torch.from_numpy(s.transpose(2,0,1)).unsqueeze(0)
    z=(x-smt)/(sst+1e-5)*(0.7*tst+0.3*sst)+0.7*tmt+0.3*smt
    scale=1+0.04*torch.randn(1,1,1,1); noise=F.avg_pool2d(torch.randn_like(z),5,1,2)
    pre=z*scale+0.015*noise; cl=pre.clamp(0,1)
    # Compare our old zero-padding and official symmetric-padding extraction.
    gt,_=utils.load_data_houston(str(ROOT/'datasets/Houston/Houston13.mat'),str(ROOT/'datasets/Houston/Houston13_7gt.mat'))
    h,w=gt.shape[:2]
    rows=[(0,0),(0,10),(6,6),(7,7),(h-1,w-1)]
    diffs=[]
    for r,c in rows:
        a=patch(s,r,c,12,'constant'); b=patch(s,r,c,12,'symmetric'); diffs.append({'center':[r,c],'max_abs_diff':float(np.max(np.abs(a-b))), 'mean_abs_diff':float(np.mean(np.abs(a-b)))})
    out={'raw_source':{'min':float(s.min()),'max':float(s.max()),'mean':float(s.mean()),'std':float(s.std())},'raw_target':{'min':float(t.min()),'max':float(t.max()),'mean':float(t.mean()),'std':float(t.std())},'scene_shift_preclamp':{'min':float(pre.min()),'max':float(pre.max()),'mean':float(pre.mean()),'std':float(pre.std()),'lt0_percent':float((pre<0).float().mean()*100),'gt1_percent':float((pre>1).float().mean()*100)},'scene_shift_clamp':{'min':float(cl.min()),'max':float(cl.max()),'eq0_percent':float((cl==0).float().mean()*100),'eq1_percent':float((cl==1).float().mean()*100)},'official_recipe':{'optimizer':'SGD','lr_default':0.001,'README_Houston_command_lr':0.2,'momentum':0.9,'weight_decay':0.0005,'scheduler':'LambdaLR polynomial','lr_gamma':0.0003,'lr_decay':0.75,'epochs':500,'batch_size':8,'patch_size':12,'padding':'symmetric'},'current_clean_recipe_before_rerun':{'optimizer':'AdamW','lr':0.002,'weight_decay':0.0001,'scheduler':'none','epochs':500,'batch_size':32,'patch_size':12,'padding':'constant-zero'},'current_mamba_wrapper_after_fix':{'padding':'symmetric','patch_size':12,'backbone_output_dim':4608},'patch_extraction_differences':diffs}
    print(json.dumps(out,indent=2)); (HERE/'mamba_audit.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__': main()
