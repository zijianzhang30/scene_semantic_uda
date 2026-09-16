import torch
from torch import nn

class SceneShiftNet(nn.Module):
    """Small shared spectral-spatial encoder and classifier for three domains."""
    def __init__(self, bands=48, classes=7):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(bands, 64, 1), nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, classes)
    def forward(self, x):
        f = self.encoder(x).flatten(1)
        return f, self.classifier(f)
