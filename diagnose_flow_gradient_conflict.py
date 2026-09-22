"""Post-hoc, read-only gradient-conflict diagnostic for saved Flow checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
LEGACY = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(LEGACY))
import utils
import UtilsCMS
import config_Houston as cfg
from net2 import DSANSS
sys.path.insert(0, str(ROOT))
from class_conditional_flow import ConditionalFlowMLP, compute_source_prototypes, sample_ot_pairs
from train_houston_flow_transport import agreement_membership, classwise_sinkhorn_ot_weighted
from class_conditional_flow import agreement_flow_matching_loss

NUM_CLASSES = 7

def gradient_vector(loss, params):
    gs = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, gs)])

def cosine(a, b):
    if a.norm() == 0 or b.norm() == 0:
        return None
    return float(F.cosine_similarity(a[None], b[None]).item())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pair-seed', type=int, default=20260922)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = np.load(ROOT / 'runs_strict_mluda_1341' / 'ilda.npz')
    UtilsCMS.ILDA = lambda s, t, n, r: (cache['s'].copy(), cache['t'].copy())
    utils.set_seed(args.seed)
    train_x, train_y = utils.get_sample_data(cache['s'], utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston13.mat'), str(LEGACY/'datasets/Houston/Houston13_7gt.mat'))[1], cfg.HalfWidth, 180)
    _, target_x, target_y, *_ = utils.get_all_data(cache['t'], utils.load_data_houston(str(LEGACY/'datasets/Houston/Houston18.mat'), str(LEGACY/'datasets/Houston/Houston18_7gt.mat'))[1], cfg.HalfWidth)
    # Deterministic diagnostic batches: source first 32 sampled items and target first 32 shuffled items.
    source_loader = DataLoader(TensorDataset(torch.tensor(train_x), torch.tensor(train_y)), batch_size=32, shuffle=True, drop_last=True)
    target_loader = DataLoader(TensorDataset(torch.tensor(target_x), torch.tensor(target_y)), batch_size=32, shuffle=True, drop_last=True)
    xs, ys = next(iter(source_loader)); xt, _ = next(iter(target_loader))
    model = DSANSS(cfg.nBand, cfg.patch_size, NUM_CLASSES).cuda()
    flow = ConditionalFlowMLP(288, NUM_CLASSES, hidden_dim=288).cuda()
    state = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
    model.load_state_dict(state['model']); flow.load_state_dict(state['flow'])
    model.train(); flow.train()
    xs, ys, xt = xs.cuda(), ys.cuda(), xt.cuda()
    source_features, _, _, source_logits, _, target_features, _, _, target_logits, _ = model(xs, xt)
    cls_loss = F.cross_entropy(source_logits, ys)
    with torch.no_grad():
        prototypes, valid = compute_source_prototypes(source_features.detach(), ys, NUM_CLASSES)
        q = target_logits.detach().softmax(1)
        s, _, _ = agreement_membership(target_features.detach(), q, prototypes, valid)
        reliable = q.argmax(1).eq(s.argmax(1)) & valid[q.argmax(1)]
        ot = classwise_sinkhorn_ot_weighted(source_features.detach(), ys, target_features.detach(), q, torch.ones_like(q[:, 0]), NUM_CLASSES, target_mask=reliable)
    torch.manual_seed(args.pair_seed)
    ps, pt, pc = sample_ot_pairs(ot, source_features, target_features.detach(), max_pairs_per_class=32)
    params = tuple(model.feature_layers.parameters())
    g_cls = gradient_vector(cls_loss, params)
    result = {
        'seed': args.seed, 'checkpoint': str(args.checkpoint),
        'pair_seed': args.pair_seed, 'source_labels': ys.cpu().tolist(),
        'reliable_ratio': float(reliable.float().mean()),
        'reliable_count': int(reliable.sum()), 'pair_count': int(len(pc)),
        'overall': {}, 'per_class': {},
    }
    if len(pc):
        torch.manual_seed(args.pair_seed + 1)
        fm_loss, _ = agreement_flow_matching_loss(flow, ps, pt, pc, detach_source=False)
        g_fm = gradient_vector(fm_loss, params)
        result['overall'] = {'cls_loss': float(cls_loss.detach()), 'fm_loss': float(fm_loss.detach()), 'grad_cls_norm': float(g_cls.norm()), 'grad_fm_norm': float(g_fm.norm()), 'cosine': cosine(g_cls, g_fm)}
        for c in range(NUM_CLASSES):
            mask = pc == c
            if not mask.any():
                result['per_class'][str(c)] = None
                continue
            torch.manual_seed(args.pair_seed + 100 + c)
            c_loss, _ = agreement_flow_matching_loss(flow, ps[mask], pt[mask], pc[mask], detach_source=False)
            g_c = gradient_vector(c_loss, params)
            result['per_class'][str(c)] = {'pairs': int(mask.sum()), 'fm_loss': float(c_loss.detach()), 'grad_fm_norm': float(g_c.norm()), 'cosine_cls_fm': cosine(g_cls, g_c)}
    result['batch_hash'] = hashlib.sha256(xs.detach().cpu().numpy().tobytes() + xt.detach().cpu().numpy().tobytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))

if __name__ == '__main__':
    main()
