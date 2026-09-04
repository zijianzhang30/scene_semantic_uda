"""Independent semantic-scene UDA model using MLUDA's DCRN_02 extractor."""
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

LEGACY_ROOT = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(LEGACY_ROOT))
from net2 import DCRN_02


class SemanticSceneUDA(nn.Module):
    """New heads/loss pathway; only DCRN_02 is reused as an extractor.

    Input: [B, 48, 7, 7]. Output h=[B,288], z_sem=[B,128],
    z_scene=[B,64], logits=[B,7].  Passing x twice makes extraction
    single-sample at training and inference, without source-reference input.
    """
    def __init__(self, bands=48, classes=7):
        super().__init__()
        self.extractor = DCRN_02(bands, 7, classes)
        self.semantic_head = nn.Sequential(nn.Linear(288, 128), nn.LayerNorm(128), nn.GELU())
        self.scene_head = nn.Sequential(nn.Linear(288, 64), nn.LayerNorm(64), nn.GELU())
        self.classifier = nn.Linear(128, classes)

    def forward(self, x):
        h, _ = self.extractor(x, x)
        z_sem = self.semantic_head(h)
        z_scene = self.scene_head(h)
        return h, z_sem, z_scene, self.classifier(z_sem)


def orthogonality_loss(z_sem, z_scene):
    # Whiten scale before cross-covariance, so this penalizes correlation rather
    # than representation magnitude.
    a = F.normalize(z_sem - z_sem.mean(0, keepdim=True), dim=0)
    b = F.normalize(z_scene - z_scene.mean(0, keepdim=True), dim=0)
    return (a.transpose(0, 1) @ b).pow(2).mean()
