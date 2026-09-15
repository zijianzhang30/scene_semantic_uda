import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SNAP = ROOT / 'runs_mluda_official_reproduction_v1' / 'shanghai_hangzhou' / 'source_snapshot'
sys.path.insert(0, str(SNAP))
import utils
from UtilsCMS import ILDA
from config_SH2HZ import pca_n, radius

path = './datasets/Shanghai-Hangzhou/DataCube.mat'
xs, xt, label_s, label_t = utils.cubeData(path)
sm = xs.reshape(-1, xs.shape[-1]).mean(0)
ss = xs.reshape(-1, xs.shape[-1]).std(0)
tm = xt.reshape(-1, xt.shape[-1]).mean(0)
ts = xt.reshape(-1, xt.shape[-1]).std(0)
alpha = 0.8
xs_shift = (xs - sm) / (ss + 1e-5) * (alpha * ts + (1-alpha) * ss) + alpha * tm + (1-alpha) * sm

def mad(a, b):
    return float(np.mean(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))

raw_diff = mad(xs_shift, xs)
ilda_xs, ilda_shift = ILDA(xs, xt, pca_n, radius)
ilda_xs_shift, _ = ILDA(xs_shift, xt, pca_n, radius)
ilda_diff = mad(ilda_xs_shift, ilda_xs)
utils.set_seed(1341)
raw_sample, raw_label = utils.get_sample_data(xs, label_s, 0, 180)
utils.set_seed(1341)
shift_sample, shift_label = utils.get_sample_data(xs_shift, label_s, 0, 180)
utils.set_seed(1341)
ilda_sample, _ = utils.get_sample_data(ilda_xs, label_s, 0, 180)
utils.set_seed(1341)
ilda_shift_sample, _ = utils.get_sample_data(ilda_xs_shift, label_s, 0, 180)
sample_raw_diff = mad(shift_sample, raw_sample)
sample_ilda_diff = mad(ilda_shift_sample, ilda_sample)
print('SHANGHAI_PRE_ILDA_SHIFT_DIAGNOSTIC')
print('source_samples', raw_sample.shape, 'seed', 1341, 'labels_equal', bool(np.array_equal(raw_label, shift_label)))
print('mad_pre_ilda', sample_raw_diff)
print('mad_post_ilda', sample_ilda_diff)
print('R', sample_ilda_diff / sample_raw_diff if sample_raw_diff else float('nan'))
print('full_source_mad_pre_ilda', raw_diff, 'full_source_mad_post_ilda', ilda_diff)
print('ilda_shapes', ilda_xs.shape, ilda_xs_shift.shape)
