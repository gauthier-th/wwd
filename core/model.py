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


class PrototypeHead(nn.Module):
    """logit = scale * cos(embedding, direction) + bias: one linear layer (exports as Gemm).

    The default quick-mode scorer. On held-out words it beat a trained MLP head by a wide
    margin at a 0.5 false-accepts/h operating point (PLAN.md §9.3): fitting 10 clips' worth of
    augmentations overfits, while the embedding's own geometry generalizes to new speakers.
    """

    def __init__(self, direction: torch.Tensor, scale: float, bias: float) -> None:
        super().__init__()
        self.linear = nn.Linear(direction.numel(), 1)
        with torch.no_grad():
            self.linear.weight.copy_(direction.reshape(1, -1) * scale)
            self.linear.bias.fill_(bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.linear(embedding).squeeze(-1)


class WakeWordModel(nn.Module):
    """[N, window_samples] audio -> [N] probability."""

    def __init__(self, encoder: Encoder, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(audio)))
