"""Only select the extra loss; preserve all forwards, augmentations and RNG isolation."""
import argparse
import sys
from pathlib import Path
import run_official_mluda_sceneshift as ss

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component', choices=['lmmd', 'scl'], required=True)
    a = p.parse_args()
    ss.BASE = Path(__file__).resolve().parent / ('runs_official_shift_component_' + a.component + '_seed1256_v1')
    ss.EXTRA_SOURCE_SCL = False
    ss.PURE_AFFINE = True
    ss.EXTRA_COMPONENT = a.component
    ss.SUMMARY = Path(__file__).resolve().parent / 'summarize_shift_components.py'
    entry, config, _, datasets = ss.baseline.SETTINGS['pavia']
    ss.baseline.SETTINGS['pavia'] = (entry, config, [1256], datasets)
    original_instrument, original_configure = ss.instrument, ss.configure
    def instrument(code):
        return original_instrument(code.replace('seeds = seeds[:3]\nnDataSet = len(seeds)', 'seeds = [1256]\nnDataSet = 1'))
    def configure(manifest, dataset, root):
        original_configure(manifest, dataset, root)
        manifest.update(seeds=[1256], extra_source_scl=False, pure_affine=True,
                        intensity_scaling=False, smooth_noise=False, extra_component=a.component,
                        added_loss='gamma * ' + a.component + '-shift only',
                        forward_policy='All 3 extra forwards retained, including unused loss computations; original BN/RNG behavior unchanged')
        import shutil
        shutil.copy2(__file__, root / Path(__file__).name)
    ss.instrument, ss.configure = instrument, configure
    ss.baseline.run('pavia', False, extension=ss)

if __name__ == '__main__':
    main()
