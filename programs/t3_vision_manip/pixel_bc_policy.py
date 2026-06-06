"""T3 — Pixel-conditioned BC policy (ResNet18 encoder + MLP head).

Input:  agentview_image (256×256×3 uint8) → resize 224×224 → normalize → ResNet18 → 512-dim
Output: 7-dim OSC delta action (same as T0/T1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF


class PixelBCPolicy(nn.Module):
    def __init__(self, action_dim: int = 7, freeze_encoder: bool = False):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.encoder = backbone  # → 512-dim

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # imgs: (B, 3, 224, 224) float32 normalized
        feats = self.encoder(imgs)
        return self.head(feats)


# ImageNet normalization constants
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def preprocess_image(img_hwc_uint8: "np.ndarray", device="cpu") -> torch.Tensor:
    """Convert HWC uint8 numpy image → (1, 3, 224, 224) float tensor, normalized."""
    import numpy as np
    img = torch.from_numpy(img_hwc_uint8.copy()).float() / 255.0  # HWC float [0,1]
    img = img.permute(2, 0, 1)                                     # CHW
    img = TF.resize(img, [224, 224], antialias=True)
    mean = _MEAN.to(img.device)
    std  = _STD.to(img.device)
    img = (img - mean[:, None, None]) / std[:, None, None]
    return img.unsqueeze(0).to(device)
