"""Student classifier using MLUDA's DCRN_02 extractor."""
from pathlib import Path
import sys
import torch
import torch.nn as nn

LEGACY_ROOT = Path('/home/zhangzj26/TGRS_MLUDA-2024')
sys.path.insert(0, str(LEGACY_ROOT))
from net2 import DCRN_02


class SemanticSceneUDA(nn.Module):
    """Checkpoint-compatible student used by the validated Scene Shift runs.

    Input: [B, 48, 7, 7]. Output h=[B,288], z_sem=[B,128],
    z_scene=[B,64], logits=[B,7]. The dormant scene head remains only to keep
    existing baseline/Scene Shift checkpoints strictly loadable.
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
