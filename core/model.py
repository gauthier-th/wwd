"""Per-wake-word head and the full audio -> score model that gets exported."""

from __future__ import annotations

import torch
from torch import nn

from core.encoder import Encoder


class Head(nn.Module):
    """Small MLP on the L2-normalized embedding -> wake-word logit."""

    def __init__(self, embedding_dim: int = 256, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding).squeeze(-1)


class WakeWordModel(nn.Module):
    """[N, window_samples] audio -> [N] probability."""

    def __init__(self, encoder: Encoder, head: Head) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(audio)))
