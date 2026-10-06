"""Sub-center ArcFace (Deng et al. 2020), as used by the EdgeSpot / Gok et al. teachers."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SubCenterArcFace(nn.Module):
    def __init__(self, dim: int, n_classes: int, k: int = 3, scale: float = 30.0,
                 margin: float = 0.3) -> None:
        super().__init__()
        self.n_classes, self.k, self.scale = n_classes, k, scale
        self.weight = nn.Parameter(torch.empty(n_classes * k, dim))
        nn.init.normal_(self.weight, std=0.01)
        self.set_margin(margin)

    def set_margin(self, margin: float) -> None:
        """Margin warm-up: with tens of thousands of classes, a full margin from step 0
        stalls learning, so training ramps it up."""
        self.margin = margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        # Past theta = pi - m, cos(theta + m) stops being monotonic: fall back to a linear penalty.
        self.th, self.mm = math.cos(math.pi - margin), math.sin(math.pi - margin) * margin

    def cosine(self, embedding: torch.Tensor) -> torch.Tensor:
        """[B, D] -> [B, C]: best sub-center cosine per class, in float32."""
        cos = F.linear(F.normalize(embedding.float()), F.normalize(self.weight.float()))
        return cos.view(-1, self.n_classes, self.k).amax(dim=-1)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cosine(embedding)
        target = cos.gather(1, labels[:, None]).clamp(-1 + 1e-7, 1 - 1e-7)
        sin = (1 - target * target).sqrt()
        phi = torch.where(target > self.th, target * self.cos_m - sin * self.sin_m, target - self.mm)
        logits = cos.scatter(1, labels[:, None], phi) * self.scale
        return F.cross_entropy(logits, labels), cos
