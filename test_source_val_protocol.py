import ast,copy,json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path.cwd()))
import source_val_protocol as protocol
import train_houston_flow_transport as tr

torch.set_num_threads(2)
historical=Path('sceneshift_diff_artifacts_20260923/train_full_mluda_scene_shift.py').read_text()
current=Path('source_val_protocol.py').read_text()
for name in ['source_split','center_patches','evaluate_source','evaluate_target']:
    def function(text):
        return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(function(historical))==ast.dump(function(current))
cache=np.load('runs_strict_mluda_1341/ilda.npz')
_,gt=tr.utils.load_data_houston(str(tr.LEGACY/'datasets/Houston/Houston13.mat'),str(tr.LEGACY/'datasets/Houston/Houston13_7gt.mat'))
for seed in [1174,1341,1370]:
    np.random.seed(seed)
    x,y=tr.utils.get_sample_data(cache['s'],gt,3,180)
    loader,reference,split=protocol.validation_data(cache['s'],gt,seed,x,y)
    assert len(loader.dataset)==1270 and len(x)==1260
assert protocol.improves_source_validation({'source_val_accuracy':.6},{'source_val_accuracy':.5})
assert not protocol.improves_source_validation({'source_val_accuracy':.5},{'source_val_accuracy':.5})
assert not protocol.improves_source_validation({'source_val_accuracy':.4,'oa':1.},{'source_val_accuracy':.5,'oa':0.})

class Pair(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn=nn.BatchNorm2d(3)
        self.drop=nn.Dropout(.3)
        self.fc=nn.Linear(3,7)
    def forward(self,s,t):
        a=self.drop(self.bn(s)).mean((2,3));b=self.drop(self.bn(t)).mean((2,3))
        return (a,None,None,self.fc(a),None,b,None,None,self.fc(b),None)
torch.manual_seed(21)
m=Pair().cuda().train(); x=torch.randn(8,3,7,7,device='cuda');t=x*2+1
baseline=copy.deepcopy(m)
buffers={n:v.clone() for n,v in m.named_buffers()}
rng=torch.cuda.get_rng_state()
out=tr.forward_with_batch_stats_restore_buffers(m,x,t)
torch.cuda.set_rng_state(rng)
ref=baseline(x,t)
assert torch.equal(out[3],ref[3])
assert all(torch.equal(v,buffers[n]) for n,v in m.named_buffers())
out[3].sum().backward();ref[3].sum().backward()
assert all(torch.equal(a.grad,b.grad) for a,b in zip(m.parameters(),baseline.parameters()))
m.bn.eval()
modes=[v.training for v in m.modules()]
cpu=torch.get_rng_state();cuda=torch.cuda.get_rng_state()
with protocol.isolated_evaluation(m):
    torch.rand(4);torch.rand(4,device='cuda');m(x,t)
assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
assert modes==[v.training for v in m.modules()]
stats=(torch.zeros(1,3,1,1,device='cuda'),torch.ones(1,3,1,1,device='cuda'),torch.ones(1,3,1,1,device='cuda'),torch.ones(1,3,1,1,device='cuda'))
for arm in ['B','B_BN','B_loss']:
    model=Pair().cuda().train()
    step=tr.training_step(model,None,x,t,torch.arange(8,device='cuda')%7,stats,arm,torch.Generator())
    expected=step['ce_s']+.5*step['ce_ss'] if arm=='B_loss' else .5*(step['ce_s']+step['ce_ss'])
    assert torch.equal(step['ce'],expected) and step['details'] is None and step['fm']==0
    step['loss'].backward()
print('PASS: historical function parity; 3 exact splits; source-only selection/tie; BN outputs/gradients/buffers; eval RNG/modes; independent loss controls.')
