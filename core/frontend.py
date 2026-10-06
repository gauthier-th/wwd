"""Audio frontend: raw 16 kHz audio -> PCEN-compressed mel energies.

Built only from Conv/Mul/Pow/Pad so it lives inside every exported graph without STFT, DFT
or Loop ops (PLAN.md §3): the STFT is a strided Conv1d with fixed windowed-DFT kernels, the
mel filterbank is a 1x1 Conv1d, and the PCEN smoother is a truncated causal FIR filter.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def hz_to_mel(f: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + f / 700.0)


def mel_to_hz(m: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def mel_filterbank(n_fft: int, n_mels: int, sample_rate: int, f_min: float, f_max: float) -> torch.Tensor:
    """HTK-scale triangular filters, shape [n_mels, n_fft // 2 + 1], no area normalization."""
    bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1, dtype=torch.float64)
    edges = mel_to_hz(
        torch.linspace(
            hz_to_mel(torch.tensor(f_min, dtype=torch.float64)),
            hz_to_mel(torch.tensor(f_max, dtype=torch.float64)),
            n_mels + 2,
            dtype=torch.float64,
        )
    )
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (bins - lo) / (mid - lo)
    down = (hi - bins) / (hi - mid)
    return torch.clamp(torch.minimum(up, down), min=0.0).float()


def dft_kernels(n_fft: int) -> torch.Tensor:
    """Hann-windowed real/imaginary DFT rows as Conv1d weights, shape [2 * bins, 1, n_fft]."""
    n = torch.arange(n_fft, dtype=torch.float64)
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
    window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)
    angle = 2 * math.pi * k * n / n_fft
    kernels = torch.cat([window * torch.cos(angle), -window * torch.sin(angle)])
    return kernels[:, None, :].float()


class Frontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop: int = 160,
        n_mels: int = 40,
        f_min: float = 20.0,
        f_max: float = 7600.0,
        pcen_taps: int = 64,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hop, self.n_mels, self.eps = hop, n_mels, eps
        mel = mel_filterbank(n_fft, n_mels, sample_rate, f_min, f_max)
        self.register_buffer("dft", dft_kernels(n_fft), persistent=False)
        # Applied to [re^2; im^2] stacked on channels, so it also sums re^2 + im^2.
        self.register_buffer("mel", torch.cat([mel, mel], dim=1)[:, :, None], persistent=False)
        # Exponent of each FIR tap, reversed so the last tap weights the current frame.
        self.register_buffer(
            "tap_powers", torch.arange(pcen_taps - 1, -1, -1, dtype=torch.float32), persistent=False
        )
        # Trainable PCEN, per mel channel, in log space (as in EdgeSpot).
        self.log_s = nn.Parameter(torch.full((n_mels,), math.log(0.04)))
        self.log_alpha = nn.Parameter(torch.full((n_mels,), math.log(0.98)))
        self.log_delta = nn.Parameter(torch.full((n_mels,), math.log(2.0)))
        self.log_r = nn.Parameter(torch.full((n_mels,), math.log(0.5)))
        self.frozen: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def freeze(self) -> None:
        """Bake the PCEN parameters into constants so export sees no parameter math."""
        self.frozen = {
            "smoother": self.smoother().clone(),
            "alpha": self.log_alpha.exp()[None, :, None].clone(),
            "delta": self.log_delta.exp()[None, :, None].clone(),
            "r": self.log_r.exp()[None, :, None].clone(),
        }
        self.frozen["delta_r"] = self.frozen["delta"] ** self.frozen["r"]

    def n_frames(self, n_samples: int) -> int:
        return (n_samples - self.dft.shape[-1]) // self.hop + 1

    def smoother(self) -> torch.Tensor:
        """Truncated, renormalized IIR smoother s * (1 - s)^k as depthwise Conv1d weights."""
        s = self.log_s.exp().clamp(max=0.5)[:, None]
        h = s * (1.0 - s) ** self.tap_powers
        return (h / h.sum(dim=1, keepdim=True))[:, None, :]

    def mel_energies(self, audio: torch.Tensor) -> torch.Tensor:
        spec = F.conv1d(audio[:, None, :], self.dft, stride=self.hop)
        return F.conv1d(spec * spec, self.mel)  # [N, n_mels, T]

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """[N, samples] -> [N, 1, n_mels, frames]. Always float32: PCEN divides energies as
        small as 1e-9, which bf16/fp16 autocast would destroy."""
        with torch.autocast(audio.device.type, enabled=False):
            return self._forward(audio.float())

    def _forward(self, audio: torch.Tensor) -> torch.Tensor:
        energy = self.mel_energies(audio)
        if self.frozen is not None:
            weights, alpha, delta, r, delta_r = (
                self.frozen[k] for k in ("smoother", "alpha", "delta", "r", "delta_r")
            )
        else:
            weights = self.smoother()
            alpha = self.log_alpha.exp()[None, :, None]
            delta = self.log_delta.exp()[None, :, None]
            r = self.log_r.exp()[None, :, None]
            delta_r = delta**r
        smooth = F.conv1d(F.pad(energy, (weights.shape[-1] - 1, 0), mode="replicate"), weights,
                          groups=self.n_mels)
        pcen = (energy / (self.eps + smooth) ** alpha + delta) ** r - delta_r
        return pcen[:, None]
