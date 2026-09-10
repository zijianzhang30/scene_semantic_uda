"""Shape-generalized SSFTT for band-aligned cross-scene inputs.

The attention/tokenizer/transformer structure is copied from the official
repository. Only the hard-coded 8*28 Conv2d input is generalized to
8*(bands-2), and very small spatial patches are padded to the minimum size
required by the official unpadded 3x3 convolutions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

OFFICIAL = Path('/home/zhangzj26/HSI_SSFTT/cls_SSFTT_IP')
sys.path.insert(0, str(OFFICIAL))
from SSFTTnet import Transformer  # noqa: E402


class SSFTTBandAligned(nn.Module):
    def __init__(self, bands, num_classes, patch_size, num_tokens=4, dim=64,
                 depth=1, heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.bands = bands; self.patch_size = patch_size; self.min_spatial = 5
        self.L = num_tokens; self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3)), nn.BatchNorm3d(8), nn.ReLU())
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(8 * (bands - 2), 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU())
        self.token_wA = nn.Parameter(torch.empty(1, self.L, 64)); nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, 64, dim)); nn.init.xavier_normal_(self.token_wV)
        self.pos_embedding = nn.Parameter(torch.empty(1, num_tokens + 1, dim)); nn.init.normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)); self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.nn1 = nn.Linear(dim, num_classes); nn.init.xavier_uniform_(self.nn1.weight); nn.init.normal_(self.nn1.bias, std=1e-6)

    def forward_features(self, x):
        if x.ndim == 4: x = x.unsqueeze(1)
        if x.shape[1] != 1: raise ValueError(f'expected [B,1,C,H,W], got {tuple(x.shape)}')
        # Official convolutions are unpadded spatially; pad only tiny patches.
        h, w = x.shape[-2:]
        if h < self.min_spatial or w < self.min_spatial:
            ph, pw = max(0, self.min_spatial-h), max(0, self.min_spatial-w)
            x = F.pad(x, (pw//2, pw-pw//2, ph//2, ph-ph//2))
        x = self.conv3d_features(x)
        x = x.reshape(x.size(0), x.size(1)*x.size(2), x.size(3), x.size(4))
        x = self.conv2d_features(x).flatten(2).transpose(1, 2)
        wa = self.token_wA.transpose(1, 2)
        A = torch.einsum('bij,bjk->bik', x, wa).transpose(1, 2).softmax(dim=-1)
        VV = torch.einsum('bij,bjk->bik', x, self.token_wV)
        T = torch.einsum('bij,bjk->bik', A, VV)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.dropout(torch.cat((cls, T), 1) + self.pos_embedding)
        x = self.transformer(x)
        return x[:, 0]

    def forward(self, x):
        return self.nn1(self.forward_features(x))
