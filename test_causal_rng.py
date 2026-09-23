"""Exercise exec -> training_step -> adaptation using a hostile module global."""
import copy
import json
import torch
import train_houston_flow_transport as training
from net2 import DSANSS


def main():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(1341)
    device = torch.device("cuda")
    model = training.install_deterministic_pool(DSANSS(48, 7, 7).to(device))
    flow = training.ConditionalFlowMLP(288, 7, hidden_dim=288).to(device)
    source = torch.randn(14, 48, 7, 7, device=device)
    target = torch.randn_like(source)
    labels = torch.arange(14, device=device) % 7
    stats = tuple(torch.full((1, 48, 1, 1), v, device=device) for v in (0., 1., .2, 1.1))
    # Compare exact official forward/backward, including tied maxima.
    for custom, official in [
        (training._DeterministicGlobalPool.apply, lambda x: torch.nn.functional.avg_pool3d(x, (1, 7, 7))),
        (training._DeterministicGlobalMaxPool.apply, lambda x: torch.nn.functional.adaptive_max_pool2d(x, 1)),
    ]:
        for tied in [False, True]:
            x = (torch.zeros(3, 4, 7, 7, device=device) if tied else torch.randn(3, 4, 7, 7, device=device)).requires_grad_()
            g = torch.randn(3, 4, 1, 1, device=device)
            a, b = custom(x), official(x)
            assert torch.equal(a, b)
            ga = torch.autograd.grad(a, x, g)[0]
            torch.use_deterministic_algorithms(False)
            try:
                gb = torch.autograd.grad(b, x, g)[0]
            finally:
                torch.use_deterministic_algorithms(True)
            assert torch.equal(ga, gb), (ga - gb).abs().max().item()
    initial = copy.deepcopy(model.state_dict())
    flow_initial = copy.deepcopy(flow.state_dict())
    cpu, cuda = torch.get_rng_state(), torch.cuda.get_rng_state()
    training.flow_rng = None
    outputs = {}
    for ablation in "ABCDEFG":
        model.load_state_dict(initial)
        flow.load_state_dict(flow_initial)
        model.zero_grad(set_to_none=True)
        flow.zero_grad(set_to_none=True)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(cuda)
        ns = dict(torch=torch, model=model, flow=flow, source=source, target=target,
                  labels=labels, stats=stats, ablation=ablation, training_step=training.training_step)
        exec('flow_rng = torch.Generator(device="cpu").manual_seed(100514)\n'
             'step = training_step(model, flow, source, target, labels, stats, ablation, flow_rng)', ns)
        step = ns['step']
        if training.ABLATIONS[ablation][1]:
            grads = torch.autograd.grad(step['fm'], tuple(model.parameters()), retain_graph=True, allow_unused=True)
            nonzero = any(g is not None and g.count_nonzero().item() for g in grads)
            assert nonzero == training.ABLATIONS[ablation][2], ablation
        step['loss'].backward()
        outputs[ablation] = {
            'grad': training.tensor_sequence_hash(p.grad if p.grad is not None else torch.empty(0) for p in model.parameters()),
            'buffers': training.tensor_sequence_hash(model.buffers()),
            'rng': training.tensor_sequence_hash([torch.get_rng_state(), torch.cuda.get_rng_state()]),
            'logits': training.tensor_sequence_hash([step['source_logits'], step['target_logits']]),
            'ce': step['ce'].item(),
        }
    assert outputs['A'] == outputs['F'], (outputs['A'], outputs['F'])
    assert outputs['B'] == outputs['C'], (outputs['B'], outputs['C'])
    assert len({v['rng'] for v in outputs.values()}) == 1
    assert training.flow_rng is None
    try:
        training.adaptation(flow, step['source_features'], step['target_features'], step['target_logits'], labels, False, False, None)
    except TypeError:
        pass
    else:
        raise AssertionError('None generator accepted')
    print(json.dumps({'exec_scope_test': 'passed', 'pairs': ['A/F', 'B/C'],
                      'FM_gradient_contract': 'C/F detached; D/E/G connected',
                      'main_rng_equal_all_A_G': True}))


if __name__ == '__main__':
    main()
