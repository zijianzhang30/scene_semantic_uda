import argparse, sys
from pathlib import Path
import hdf5storage
import numpy as np
ROOT=Path('/home/zhangzj26/TGRS_MLUDA-2024'); sys.path.insert(0,str(ROOT))
from sceneshiftnet_dcrn_train import DatasetSpec, parse_bool, run
SPEC=DatasetSpec('houston13_houston18','Houston13','Houston18',48,7,7,2,.009,'raw cube values')
def load_cubes(normalization):
    ps=[ROOT/'datasets/Houston/Houston13.mat',ROOT/'datasets/Houston/Houston13_7gt.mat',ROOT/'datasets/Houston/Houston18.mat',ROOT/'datasets/Houston/Houston18_7gt.mat']
    s=hdf5storage.loadmat(str(ps[0]))['ori_data']; sg=hdf5storage.loadmat(str(ps[1]))['map']; t=hdf5storage.loadmat(str(ps[2]))['ori_data']; tg=hdf5storage.loadmat(str(ps[3]))['map']
    return s,sg,t,tg,ps
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=1341); p.add_argument('--method',choices=['baseline','sceneshift'],default='sceneshift'); p.add_argument('--epochs',type=int,default=100); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--source-per-class',type=int,default=180); p.add_argument('--use-ilda',type=parse_bool,default=False); p.add_argument('--normalization',choices=['none','loader'],default='none'); p.add_argument('--lambda-margin',type=float,default=0.0); p.add_argument('--margin',type=float,default=0.2); p.add_argument('--lambda-align',type=float,default=0.0); p.add_argument('--prepare-only',action='store_true'); a=p.parse_args(); run(a,SPEC,load_cubes,Path(__file__).parent/'runs_sceneshiftnet_dcrn/houston_raw',__file__)
