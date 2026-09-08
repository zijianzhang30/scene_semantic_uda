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


class DistributionalSpectralPillarsClassifier(nn.Module):
    """Distributional spectral pillars followed by lightweight spatial Mamba.

    The spectral token construction and shared spectral encoder are identical
    to :class:`SpectralPillarsClassifier`.  Instead of mean-pooling the 48
    encoded band tokens directly into a pillar descriptor, the shared pooled
    spectral feature predicts a K-way distribution and K state descriptors.
    Their expectation is the pseudo-image feature passed to the spatial path.

    Pillars are intentionally flattened to ``B*H*W`` before spectral encoding:
    there is no Python loop over the 144 spatial locations.
    """

    patch_size = 12
    representation_dim = 64

    def __init__(self, bands=48, classes=7, hidden_dim=64, states=8,
                 spectral_depth=2, spatial_depth=2, patch_size=12):
        super().__init__()
        if patch_size != 12:
            raise ValueError("The first implementation is defined for 12x12 patches")
        self.patch_size = patch_size
        self.bands = bands
        self.states = states
        self.representation_dim = hidden_dim
        self.band_value_embed = nn.Linear(2, hidden_dim)
        self.band_position = nn.Parameter(torch.zeros(1, bands, hidden_dim))
        nn.init.trunc_normal_(self.band_position, std=0.02)
        self.spectral_blocks = nn.ModuleList(
            [_MambaBlock(hidden_dim) for _ in range(spectral_depth)]
        )
        self.spectral_norm = nn.LayerNorm(hidden_dim)
        # Both heads consume the same pooled spectral representation.  They
        # are inexpensive projections, not K separate spectral encoders.
        self.distribution_head = nn.Linear(hidden_dim, states)
        self.state_feature_head = nn.Linear(hidden_dim, states * hidden_dim)
        self.spatial_blocks = nn.ModuleList(
            [_MambaBlock(hidden_dim) for _ in range(spatial_depth)]
        )
        self.spatial_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))

    def forward_features(self, x, return_distribution=False):
        b, c, h, w = x.shape
        if c != self.bands or h != self.patch_size or w != self.patch_size:
            raise ValueError(
                f"expected [B,{self.bands},{self.patch_size},{self.patch_size}], got {tuple(x.shape)}"
            )
        pillars = x.permute(0, 2, 3, 1).reshape(b * h * w, c)
        pillar_mean = pillars.mean(dim=1, keepdim=True)
        tokens = torch.stack((pillars, pillars - pillar_mean), dim=-1)
        tokens = self.band_value_embed(tokens) + self.band_position
        for block in self.spectral_blocks:
            tokens = block(tokens)
        shared_spectral = self.spectral_norm(tokens.mean(dim=1))
        pi = torch.softmax(self.distribution_head(shared_spectral), dim=-1)
        states = self.state_feature_head(shared_spectral).view(-1, self.states, self.representation_dim)
        descriptors = torch.sum(pi.unsqueeze(-1) * states, dim=1)
        pseudo_image = descriptors.reshape(b, h, w, self.representation_dim).permute(0, 3, 1, 2)
        spatial_tokens = pseudo_image.flatten(2).transpose(1, 2)
        for block in self.spatial_blocks:
            spatial_tokens = block(spatial_tokens)
        representation = self.spatial_norm(spatial_tokens.mean(dim=1))
        if return_distribution:
            return representation, pi.view(b, h, w, self.states)
        return representation

    def distribution_stats(self, x):
        """Detached diagnostic statistics for latent-state collapse audits."""
        _, pi = self.forward_features(x, return_distribution=True)
        flat = pi.flatten(0, 2)
        mean_pi = flat.mean(dim=0)
        entropy = -(flat * flat.clamp_min(1e-8).log()).sum(dim=1).mean()
        return {
            "latent_entropy": float(entropy.detach()),
            "latent_effective_states": float(entropy.detach().exp()),
            "latent_max_probability": float(flat.max(dim=1).values.mean().detach()),
            "latent_min_mean_probability": float(mean_pi.min().detach()),
            "latent_max_mean_probability": float(mean_pi.max().detach()),
        }

    def forward(self, x):
        return self.head(self.forward_features(x))


class _JointMixerBlock(nn.Module):
    """Interleaved spectral and spatial mixing on a 3-D HSI feature grid."""

    def __init__(self, dim):
        super().__init__()
        # GroupNorm avoids source/shift batch-statistics coupling while keeping
        # the tensor in [B,D,C,H,W] layout.
        groups = min(8, dim)
        self.spectral_norm = nn.GroupNorm(groups, dim)
        self.spectral_mix = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(5, 1, 1), padding=(2, 0, 0),
                      groups=dim, bias=False),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.spatial_norm = nn.GroupNorm(groups, dim)
        self.spatial_mix = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                      groups=dim, bias=False),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
        )

    def forward(self, x):
        # Spatial mixing consumes the spectrally updated tensor, so the two
        # operations interact inside every block instead of meeting only at
        # the classifier.
        x = x + self.spectral_mix(self.spectral_norm(x))
        return x + self.spatial_mix(self.spatial_norm(x))


class JointSpectralSpatialMambaClassifier(nn.Module):
    """Lightweight joint HSI mixer with relative spectral cues.

    Raw reflectance and adjacent-band differences are projected separately and
    fused immediately.  Two mixer blocks retain the full spectral axis while
    alternating spectral and spatial operations.  Only after joint mixing is
    the spectral axis pooled for one global spatial Mamba block.
    """

    patch_size = 12
    representation_dim = 64

    def __init__(self, bands=48, classes=7, joint_dim=24, hidden_dim=64,
                 joint_depth=2, patch_size=12):
        super().__init__()
        if patch_size != 12:
            raise ValueError("The first implementation is defined for 12x12 patches")
        self.patch_size = patch_size
        self.bands = bands
        self.representation_dim = hidden_dim
        # Treat bands as the depth of a 3-D HSI grid; projections do not erase
        # or pool the spectral dimension.
        self.raw_projection = nn.Conv3d(1, joint_dim, kernel_size=1, bias=False)
        self.delta_projection = nn.Conv3d(1, joint_dim, kernel_size=1, bias=False)
        self.input_fusion = nn.Sequential(
            nn.Conv3d(2 * joint_dim, joint_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, joint_dim), joint_dim),
            nn.GELU(),
        )
        self.joint_blocks = nn.ModuleList(
            [_JointMixerBlock(joint_dim) for _ in range(joint_depth)]
        )
        self.spectral_readout = nn.Sequential(
            nn.Conv3d(joint_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.global_mamba = _MambaBlock(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))

    @staticmethod
    def adjacent_difference(x):
        # delta_b = X_{b+1} - X_b for b < C-1; the final band is zero padded.
        delta = torch.zeros_like(x)
        delta[:, :-1] = x[:, 1:] - x[:, :-1]
        return delta

    def forward_features(self, x):
        b, c, h, w = x.shape
        if c != self.bands or h != self.patch_size or w != self.patch_size:
            raise ValueError(
                f"expected [B,{self.bands},{self.patch_size},{self.patch_size}], got {tuple(x.shape)}"
            )
        raw = self.raw_projection(x.unsqueeze(1))
        delta = self.delta_projection(self.adjacent_difference(x).unsqueeze(1))
        joint = self.input_fusion(torch.cat((raw, delta), dim=1))
        for block in self.joint_blocks:
            joint = block(joint)
        # Joint feature is retained through both mixers; spectral aggregation
        # happens once, directly before spatial long-range reasoning.
        pseudo_image = self.spectral_readout(joint).mean(dim=2)
        tokens = pseudo_image.flatten(2).transpose(1, 2)
        tokens = self.global_mamba(tokens)
        return self.global_norm(tokens.mean(dim=1))

    def forward(self, x):
        return self.head(self.forward_features(x))


class _SceneRobustJointMixerBlock(nn.Module):
    """Medium-capacity joint mixer with strong pointwise spectral channels."""

    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        groups = min(8, dim)
        self.spectral_norm = nn.GroupNorm(groups, dim)
        self.spectral_mix = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(7, 1, 1), padding=(3, 0, 0),
                      groups=dim, bias=False),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.spatial_norm = nn.GroupNorm(groups, dim)
        self.spatial_mix = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=(0, 1, 1),
                      groups=dim, bias=False),
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        # This channel MLP is deliberately dense (rather than depthwise): it
        # provides the spectral/channel discrimination absent from Joint v1.
        self.channel_norm = nn.GroupNorm(groups, dim)
        self.channel_mlp = nn.Sequential(
            nn.Conv3d(dim, dim * mlp_ratio, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv3d(dim * mlp_ratio, dim, kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = x + self.spectral_mix(self.spectral_norm(x))
        x = x + self.spatial_mix(self.spatial_norm(x))
        return x + self.channel_mlp(self.channel_norm(x))


class SceneRobustJointSpectralSpatialMambaClassifier(nn.Module):
    """Medium joint spectral-spatial Mamba retaining absolute and Δ spectra.

    Unlike the pillar models, no per-pixel spectral descriptor is formed.  The
    full [bands, height, width] grid remains present through four interleaved
    joint blocks.  Dense channel MLPs add spectral discrimination, then two
    compact Mamba blocks reason over the final 12x12 spatial token sequence.
    """

    patch_size = 12
    representation_dim = 128

    def __init__(self, bands=48, classes=7, joint_dim=96, hidden_dim=128,
                 joint_depth=4, global_depth=2, patch_size=12):
        super().__init__()
        if patch_size != 12:
            raise ValueError("The first implementation is defined for 12x12 patches")
        self.patch_size = patch_size
        self.bands = bands
        self.representation_dim = hidden_dim
        self.raw_projection = nn.Conv3d(1, joint_dim, kernel_size=1, bias=False)
        self.delta_projection = nn.Conv3d(1, joint_dim, kernel_size=1, bias=False)
        self.input_fusion = nn.Sequential(
            nn.Conv3d(2 * joint_dim, joint_dim, kernel_size=1, bias=False),
            nn.GroupNorm(min(8, joint_dim), joint_dim),
            nn.GELU(),
        )
        self.joint_blocks = nn.ModuleList(
            [_SceneRobustJointMixerBlock(joint_dim) for _ in range(joint_depth)]
        )
        self.spectral_readout = nn.Sequential(
            nn.Conv3d(joint_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.global_mamba = nn.ModuleList(
            [_MambaBlock(hidden_dim) for _ in range(global_depth)]
        )
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, classes))

    @staticmethod
    def adjacent_difference(x):
        delta = torch.zeros_like(x)
        delta[:, :-1] = x[:, 1:] - x[:, :-1]
        return delta

    def forward_features(self, x):
        b, c, h, w = x.shape
        if c != self.bands or h != self.patch_size or w != self.patch_size:
            raise ValueError(
                f"expected [B,{self.bands},{self.patch_size},{self.patch_size}], got {tuple(x.shape)}"
            )
        raw = self.raw_projection(x.unsqueeze(1))
        delta = self.delta_projection(self.adjacent_difference(x).unsqueeze(1))
        joint = self.input_fusion(torch.cat((raw, delta), dim=1))
        for block in self.joint_blocks:
            joint = block(joint)
        pseudo_image = self.spectral_readout(joint).mean(dim=2)
        tokens = pseudo_image.flatten(2).transpose(1, 2)
        for block in self.global_mamba:
            tokens = block(tokens)
        return self.global_norm(tokens.mean(dim=1))

    def forward(self, x):
        return self.head(self.forward_features(x))
