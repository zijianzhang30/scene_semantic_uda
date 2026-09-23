"""Real DSANSS backward and buffer checks for the auxiliary BN correction."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import nn
import train_houston_flow_transport as tr
from net2 import DSANSS

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(1341)
base=tr.install_deterministic_pool(DSANSS(48,7,7).cuda()).train()
x=torch.rand(8,48,7,7,device='cuda'); t=torch.rand_like(x)
labels=torch.arange(8,device='cuda')%7
stats=tuple(torch.full((1,48,1,1),v,device='cuda') for v in [.1,.05,.15,.08])
initial=copy.deepcopy(base.state_dict())
for arm in ['B','C','D','E']:
    model=tr.install_deterministic_pool(DSANSS(48,7,7).cuda()).train()
    model.load_state_dict(initial)
    flow=tr.ConditionalFlowMLP(288,7,hidden_dim=288).cuda()
    cpu=torch.get_rng_state(); cuda=torch.cuda.get_rng_state()
    reference=copy.deepcopy(model)
    reference(x,t)
    reference_buffers={n:v.clone() for n,v in reference.named_buffers()}
    torch.set_rng_state(cpu);torch.cuda.set_rng_state(cuda)
    step=tr.training_step(model,flow,x,t,labels,stats,arm,torch.Generator().manual_seed(5))
    assert all(torch.equal(v,reference_buffers[n]) for n,v in model.named_buffers()),arm
    assert torch.isfinite(step['loss']),arm
    assert torch.equal(step['ce'],.5*(step['ce_s']+step['ce_ss'])),arm
    step['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),arm
    print('PASS',arm,'full DSANSS backward, finite gradients, original-only BN buffers',flush=True)
