"""Clean spectral-spatial gated Mamba backbone for HSI patches.

This module contains no domain adaptation objective.  It maps a 48-band
12x12 patch to one representation by running independent Mamba sequences
over bands and spatial positions, then fusing them with a sample-adaptive
feature-only gate.
"""
import torch
import torch.nn as nn
from mamba_ssm import Mamba


class _MambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim)

    def forward(self, x):
        return x + self.mamba(self.norm(x))


class SpectralSpatialGatedMambaClassifier(nn.Module):
    """Two-path spectral/spatial Mamba classifier (source CE only)."""

    patch_size = 12
    representation_dim = 64

    def __init__(self, bands=48, classes=7, stem_dim=32, hidden_dim=64,
                 depth=2, patch_size=12):
        super().__init__()
        if patch_size != 12:
            raise ValueError("The first implementation is defined for 12x12 patches")
        self.patch_size = patch_size
        self.representation_dim = hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(bands, stem_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )
        # Each spectral token is one band's local 12x12 descriptor.
        self.spectral_embed = nn.Linear(patch_size * patch_size, hidden_dim)
        self.spectral_blocks = nn.ModuleList([_MambaBlock(hidden_dim) for _ in range(depth)])
        # Each spatial token carries the stem's learned spectral embedding.
        self.spatial_embed = nn.Linear(stem_dim, hidden_dim)
        self.spatial_blocks = nn.ModuleList([_MambaBlock(hidden_dim) for _ in range(depth)])
        self.spec_norm = nn.LayerNorm(hidden_dim)
        self.spat_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))

    def forward_features(self, x):
        b, c, h, w = x.shape
        if h != self.patch_size or w != self.patch_size:
            raise ValueError(f"expected {self.patch_size}x{self.patch_size} patches, got {h}x{w}")
        # B,C,HW -> B,C,D: band sequence with each token retaining local shape.
        spec = self.spectral_embed(x.flatten(2))
        for block in self.spectral_blocks:
            spec = block(spec)
        z_spec = self.spec_norm(spec.mean(dim=1))

        stem = self.stem(x)
        # B,HW,stem_dim -> B,HW,D: spatial sequence with spectral embedding.
        spat = self.spatial_embed(stem.flatten(2).transpose(1, 2))
        for block in self.spatial_blocks:
            spat = block(spat)
        z_spat = self.spat_norm(spat.mean(dim=1))

        gate = self.gate(torch.cat([z_spec, z_spat], dim=1))
        return gate * z_spec + (1.0 - gate) * z_spat

    def forward(self, x):
        return self.head(self.forward_features(x))


class SpectralPillarsClassifier(nn.Module):
    """Spectral abstraction first, followed by spatial Mamba reasoning.

    Each spatial location is encoded independently as a sequence of spectral
    band tokens.  A token consists of raw reflectance, its deviation from the
    pillar mean, and a learned band-position embedding.  The resulting pillar
    descriptors form a D-channel pseudo-image, whose H*W locations are then
    processed by a separate spatial Mamba encoder.
    """

    patch_size = 12
    representation_dim = 64

    def __init__(self, bands=48, classes=7, hidden_dim=64,
                 spectral_depth=2, spatial_depth=2, patch_size=12):
        super().__init__()
        if patch_size != 12:
            raise ValueError("The first implementation is defined for 12x12 patches")
        self.patch_size = patch_size
        self.bands = bands
        self.representation_dim = hidden_dim
        # shared across all 144 pillars; input is [reflectance, deviation].
        self.band_value_embed = nn.Linear(2, hidden_dim)
        self.band_position = nn.Parameter(torch.zeros(1, bands, hidden_dim))
        nn.init.trunc_normal_(self.band_position, std=0.02)
        self.spectral_blocks = nn.ModuleList(
            [_MambaBlock(hidden_dim) for _ in range(spectral_depth)]
        )
        self.spectral_norm = nn.LayerNorm(hidden_dim)
        # This is the spatial pseudo-image pathway: B,D,H,W -> B,HW,D.
        self.spatial_blocks = nn.ModuleList(
            [_MambaBlock(hidden_dim) for _ in range(spatial_depth)]
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))

    def forward_features(self, x):
        b, c, h, w = x.shape
        if c != self.bands or h != self.patch_size or w != self.patch_size:
            raise ValueError(f"expected [B,{self.bands},{self.patch_size},{self.patch_size}], got {tuple(x.shape)}")
        # B,C,H,W -> (B*H*W),C,2.  Each row is one spectral pillar.
        pillars = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        mean = pillars.mean(dim=1, keepdim=True)
        tokens = torch.stack((pillars, pillars - mean), dim=-1)
        tokens = self.band_value_embed(tokens) + self.band_position
        for block in self.spectral_blocks:
            tokens = block(tokens)
        descriptors = self.spectral_norm(tokens.mean(dim=1))
        # Explicit pseudo-image reconstruction before spatial reasoning.
        pseudo_image = descriptors.reshape(b, h, w, self.representation_dim).permute(0, 3, 1, 2)
        spatial_tokens = pseudo_image.flatten(2).transpose(1, 2)
        for block in self.spatial_blocks:
            spatial_tokens = block(spatial_tokens)
        return self.spatial_norm(spatial_tokens.mean(dim=1))

    def forward(self, x):
        return self.head(self.forward_features(x))
