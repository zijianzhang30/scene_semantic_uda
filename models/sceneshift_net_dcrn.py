import sys
import torch
from torch import nn

sys.path.insert(0, '/home/zhangzj26/TGRS_MLUDA-2024')
from net2 import DCRN_02

class DCRNEncoder(nn.Module):
    """Reuse official DCRN spectral/spatial blocks, stopping before MBCA."""
    def __init__(self, bands=48, patch_size=7):
        super().__init__()
        self.core = DCRN_02(bands, patch_size, 7)
    def forward(self, x):
        x = x.unsqueeze(1)
        a = self.core.activation1(self.core.bn1(self.core.conv1(x)))
        residual = a
        a = self.core.activation2(self.core.bn2(self.core.conv2(a)))
        a = self.core.conv3(a) + residual
        a = self.core.activation3(self.core.bn3(a))
        a = self.core.activation4(self.core.bn4(self.core.conv4(a))).squeeze(2)
        b = self.core.activation5(self.core.bn5(self.core.conv5(x)))
        residual = self.core.conv8(b)
        b = self.core.activation6(self.core.bn6(self.core.conv6(b)))
        b = self.core.activation7(self.core.bn7(self.core.conv7(b) + residual)).squeeze(2)
        z = torch.cat((a, b), 1)
        z = self.core.ca(z) * z
        z = self.core.sa(z) * z
        return torch.nn.functional.adaptive_avg_pool2d(z, 1).flatten(1)

class SceneShiftNetDCRN(nn.Module):
    def __init__(self, bands=48, classes=7, patch_size=7):
        super().__init__()
        self.encoder = DCRNEncoder(bands, patch_size)
        self.classifier = nn.Linear(288, classes)
    def forward(self, x):
        feature = self.encoder(x)
        return feature, self.classifier(feature)
