"""Keyword embedding encoder: PCEN frontend -> BC-ResNet -> temporal attention -> embedding.

BC-ResNet follows Kim et al. 2021 (broadcasted residual learning); the temporal
self-attention + attentive pooling tail follows the EdgeSpot recipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from core.frontend import Frontend


@dataclass(frozen=True)
class EncoderConfig:
    width: int = 32  # stage-1 channels; BC-ResNet-tau uses 8 * tau
    embedding_dim: int = 256
    sub_bands: int = 5
    dropout: float = 0.1
    attention: bool = True


class SubSpectralNorm(nn.Module):
    """BatchNorm with separate statistics per (channel, frequency sub-band).

    In eval mode it collapses to a per-(channel, frequency) affine, which exports as a plain
    Mul + Add instead of reshapes around a BatchNormalization node.
    """

    def __init__(self, channels: int, sub_bands: int) -> None:
        super().__init__()
        self.channels, self.sub_bands = channels, sub_bands
        self.bn = nn.BatchNorm2d(channels * sub_bands)
        self.frozen: tuple[torch.Tensor, torch.Tensor] | None = None

    @torch.no_grad()
    def freeze(self, n_freq: int) -> None:
        """Precompute the eval-mode affine as [1, C, F, 1] constants for export."""
        bn = self.bn
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale
        per_bin = n_freq // self.sub_bands
        self.frozen = tuple(
            v.reshape(self.channels, self.sub_bands).repeat_interleave(per_bin, dim=1)[None, :, :, None]
            for v in (scale, shift)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.frozen is not None:
            scale, shift = self.frozen
            return x * scale + shift
        n, c, f, t = x.shape
        return self.bn(x.reshape(n, c * self.sub_bands, f // self.sub_bands, t)).reshape(n, c, f, t)


class BCResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride_f: int, dilation_t: int, sub_bands: int,
                 dropout: float) -> None:
        super().__init__()
        self.transition = c_in != c_out or stride_f != 1
        self.proj = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU())
            if c_in != c_out
            else nn.Identity()
        )
        self.freq = nn.Sequential(  # f2: frequency-wise depthwise conv
            nn.Conv2d(c_out, c_out, (3, 1), stride=(stride_f, 1), padding=(1, 0), groups=c_out,
                      bias=False),
            SubSpectralNorm(c_out, sub_bands),
        )
        self.temporal = nn.Sequential(  # f1: temporal depthwise conv on the frequency average
            nn.Conv2d(c_out, c_out, (1, 3), padding=(0, dilation_t), dilation=(1, dilation_t),
                      groups=c_out, bias=False),
            nn.BatchNorm2d(c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 1, bias=False),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x2 = self.freq(self.proj(x))
        out = x2 + self.temporal(x2.mean(dim=2, keepdim=True))
        if not self.transition:
            out = out + identity
        return F.relu(out)


class TemporalAttention(nn.Module):
    """Single-head self-attention over time with a residual connection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, T, D]
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = torch.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)
        return x + self.out(attn @ v)


class AttentivePooling(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [N, T, D] -> [N, D]
        w = torch.softmax(self.score(x), dim=1)
        return (w * x).sum(dim=1)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # Written out (not F.normalize) to export as ReduceSum/Sqrt/Div only.
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)


class Encoder(nn.Module):
    """[N, samples] audio -> [N, embedding_dim] L2-normalized embedding."""

    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config = config or EncoderConfig()
        c = config.width
        widths = [c, c * 3 // 2, c * 2, c * 5 // 2]
        stages = [  # (blocks, stride on frequency, temporal dilation)
            (2, 1, 1),
            (2, 2, 2),
            (4, 2, 4),
            (4, 1, 8),
        ]
        self.frontend = Frontend()
        self.head = nn.Sequential(
            nn.Conv2d(1, 2 * c, 5, stride=(2, 1), padding=2, bias=False),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(),
        )
        blocks, c_in = [], 2 * c
        for (n, stride_f, dilation), c_out in zip(stages, widths):
            for i in range(n):
                blocks.append(BCResBlock(c_in, c_out, stride_f if i == 0 else 1, dilation,
                                         config.sub_bands, config.dropout))
                c_in = c_out
        self.blocks = nn.Sequential(*blocks)
        tail_dim = 4 * c
        # The 5-tall depthwise kernel collapses the remaining 5 frequency bins.
        self.tail = nn.Sequential(
            nn.Conv2d(c_in, c_in, 5, padding=(0, 2), groups=c_in, bias=False),
            nn.Conv2d(c_in, tail_dim, 1, bias=False),
            nn.BatchNorm2d(tail_dim),
            nn.ReLU(),
        )
        self.attention = TemporalAttention(tail_dim) if config.attention else nn.Identity()
        self.pool = AttentivePooling(tail_dim)
        self.project = nn.Linear(tail_dim, config.embedding_dim)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.tail(self.blocks(self.head(self.frontend(audio))))  # [N, D, 1, T]
        x = self.attention(x.squeeze(2).transpose(1, 2))
        return l2_normalize(self.project(self.pool(x)))
