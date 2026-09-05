"""DCRN-only classifier used by the clean four-group audit.

DCRN_02 exposes two inputs because the original MLUDA model used a paired
source/target path. Its internal CrossAttention is therefore executed here as
``DCRN_02(x, x)``: both inputs are the same sample, so no source-target batch
interaction or adaptation is introduced. All four audit groups use this exact
backbone call.
"""
from pathlib import Path
import sys

import torch.nn as nn

LEGACY_ROOT = Path("/home/zhangzj26/TGRS_MLUDA-2024")
sys.path.insert(0, str(LEGACY_ROOT))
from net2 import DCRN_02


class DCRNClassifier(nn.Module):
    """DCRN_02 feature extractor plus a single supervised CE classifier."""

    backbone_call = "DCRN_02(x, x)"
    cross_attention_source_target_interaction = False

    def __init__(self, bands=48, patch_size=7, classes=7):
        super().__init__()
        self.backbone = DCRN_02(bands, patch_size, classes)
        self.classifier = nn.Linear(288, classes)

    def forward_features(self, x):
        features, _same_input_features = self.backbone(x, x)
        return features

    def forward(self, x):
        return self.classifier(self.forward_features(x))


# Keep the old import name available to legacy evaluators; the clean audit does
# not use the previous semantic/scene heads or their losses.
SemanticSceneUDA = DCRNClassifier
