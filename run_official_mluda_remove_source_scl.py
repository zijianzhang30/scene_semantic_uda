"""Single-factor official-aligned Pavia run: remove extra source SCL."""
import run_official_mluda_sceneshift as ss
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ss.BASE=HERE/'runs_mluda_official_pure_affine_no_extra_source_scl_v1'
ss.EXTRA_SOURCE_SCL=False
ss.PURE_AFFINE=True
_base_instrument=ss.instrument
_base_configure=ss.configure
old=ss.hooks
def hooks(dataset,root):
    h=old(dataset,root)
    return h
ss.hooks=hooks
def instrument(code):
    code=code.replace('seeds = seeds[:3]\nnDataSet = len(seeds)','seeds = [1256]\nnDataSet = 1')
    code=_base_instrument(code)
    # The hook itself is controlled by EXTRA_SOURCE_SCL=False.
    return code
ss.instrument=instrument
def configure(manifest,dataset,root):
    _base_configure(manifest,dataset,root); manifest['extra_source_scl']=False; manifest['added_loss']='gamma*(LMMD_shift + shifted pseudo-label SCL only)'; (root/'config.json').write_text(__import__('json').dumps(manifest,indent=2))
ss.configure=configure
if __name__=='__main__': ss.baseline.run('pavia',False,extension=ss)
