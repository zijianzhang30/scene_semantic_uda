"""Clean DAMamba feature extractor + supervised classifier.

Only the official DAMamba ``MambaFeature`` backbone is imported.  The
prototype, pseudo-label, LMMD, FixMatch and other adaptation code is excluded.
The official feature implementation reshapes tokens to 12x12 and emits 4608
features, so this clean wrapper intentionally uses 12x12 patches.
"""
from pathlib import Path
import sys

import torch
import torch.nn as nn

DAMAMBA_ROOT = Path("/home/zhangzj26/DAMamba")
sys.path.insert(0, str(DAMAMBA_ROOT))
from DAMamba_basenet import MambaFeature  # noqa: E402


class ChannelAttention(nn.Module):
    """Official DAMamba feature refinement, copied without adaptation logic."""

    def __init__(self, in_planes=4608, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Official DAMamba spatial attention (the feature map is 1x1 here)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class MambaBackboneClassifier(nn.Module):
    patch_size = 12
    backbone_output_dim = 4608

    def __init__(self, bands=48, classes=7, bottleneck_width=256):
        super().__init__()
        self.backbone = MambaFeature(bands, patch_size=self.patch_size)
        # These modules are part of TransferNet.predict.  The official
        # get_parameters() omits them, so retain that fixed-refinement behavior.
        self.channel_attention = ChannelAttention(self.backbone_output_dim)
        self.spatial_attention = SpatialAttention()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone_output_dim, bottleneck_width), nn.ReLU()
        )
        self.classifier = nn.Linear(bottleneck_width, classes)

    def forward_features(self, x):
        features = self.backbone(x)
        features = self.channel_attention(features) * features
        features = self.spatial_attention(features) * features
        return self.pool(features).flatten(1)

    def forward(self, x):
        return self.classifier(self.bottleneck(self.forward_features(x)))
