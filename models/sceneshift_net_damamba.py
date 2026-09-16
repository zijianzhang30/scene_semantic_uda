"""Clean DAMamba feature encoder and shared classifier; no UDA modules."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_model import MambaBackboneClassifier


class DAMambaEncoder(nn.Module):
    feature_dim = 256

    def __init__(self, bands=48):
        super().__init__()
        clean = MambaBackboneClassifier(bands=bands, classes=7, bottleneck_width=self.feature_dim)
        self.backbone = clean.backbone
        self.channel_attention = clean.channel_attention
        self.spatial_attention = clean.spatial_attention
        self.pool = clean.pool
        self.bottleneck = clean.bottleneck

    def forward(self, x):
        if x.shape[-2:] != (7, 7):
            raise ValueError(f"Expected external 7x7 patches, got {tuple(x.shape[-2:])}")
        # Official MambaFeature hard-codes a 12x12 token grid. Preserve the
        # external 7x7 protocol and deterministically adapt only inside E().
        x = F.pad(x, (2, 3, 2, 3), mode="reflect")
        f = self.backbone(x)
        f = self.channel_attention(f) * f
        f = self.spatial_attention(f) * f
        return self.bottleneck(self.pool(f).flatten(1))


class SceneShiftNetDAMamba(nn.Module):
    def __init__(self, bands=48, classes=7):
        super().__init__()
        self.encoder = DAMambaEncoder(bands)
        self.classifier = nn.Linear(self.encoder.feature_dim, classes)

    def forward(self, x):
        feature = self.encoder(x)
        return feature, self.classifier(feature)
