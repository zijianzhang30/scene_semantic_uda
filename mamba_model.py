"""Clean DAMamba feature extractor + supervised classifier.

Only the official DAMamba ``MambaFeature`` backbone is imported.  The
prototype, pseudo-label, LMMD, FixMatch and other adaptation code is excluded.
The official feature implementation reshapes tokens to 12x12 and emits 4608
features, so this clean wrapper intentionally uses 12x12 patches.
"""
from pathlib import Path
import sys

import torch.nn as nn

DAMAMBA_ROOT = Path("/home/zhangzj26/DAMamba")
sys.path.insert(0, str(DAMAMBA_ROOT))
from DAMamba_basenet import MambaFeature  # noqa: E402


class MambaBackboneClassifier(nn.Module):
    patch_size = 12
    backbone_output_dim = 4608

    def __init__(self, bands=48, classes=7, bottleneck_width=256):
        super().__init__()
        self.backbone = MambaFeature(bands, patch_size=self.patch_size)
        self.bottleneck = nn.Sequential(
            nn.Linear(self.backbone_output_dim, bottleneck_width), nn.ReLU()
        )
        self.classifier = nn.Linear(bottleneck_width, classes)

    def forward_features(self, x):
        features = self.backbone(x)
        return features.flatten(1)

    def forward(self, x):
        return self.classifier(self.bottleneck(self.forward_features(x)))
