"""Batch augmentation: level, reverb, microphone colour, noise, noise floor.

Device-agnostic torch: training runs it on the GPU, the fine-tuning server on the CPU. The
banks are built from plain arrays (training.augment loads MUSAN / OpenSLR RIRs; the server
loads the subset shipped with each base model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AugmentConfig:
    level_dbfs: tuple[float, float] = (-40.0, -15.0)  # target speech RMS
    p_reverb: float = 0.5
    p_noise: float = 0.8
    snr_db: tuple[float, float] = (0.0, 25.0)
    p_babble: float = 0.2  # babble instead of noise/music
    p_eq: float = 0.5
    eq_db: float = 6.0
    floor_dbfs: tuple[float, float] = (-85.0, -60.0)  # always on: no digital silence


class NoiseBank:
    """Noise audio concatenated in one tensor; crops are random slices."""

    def __init__(self, audio: np.ndarray, device: torch.device | str = "cpu") -> None:
        self.data = torch.from_numpy(np.asarray(audio, dtype=np.float16)).to(device)

    def sample(self, n: int, length: int) -> torch.Tensor:
        starts = torch.randint(0, len(self.data) - length, (n,), device=self.data.device)
        idx = starts[:, None] + torch.arange(length, device=self.data.device)[None]
        return self.data[idx].float()


class RirBank:
    """Room impulse responses, trimmed to `length` and aligned so the direct path is at 0."""

    def __init__(self, rirs: list[np.ndarray] | np.ndarray, device: torch.device | str = "cpu",
                 length: int = 8000) -> None:
        bank = np.zeros((len(rirs), length), dtype=np.float32)
        for i, r in enumerate(rirs):
            r = np.asarray(r, dtype=np.float32)
            r = r[int(np.argmax(np.abs(r))):][:length]
            bank[i, : len(r)] = r / (np.abs(r).max() + 1e-9)
        self.data = torch.from_numpy(bank).to(device)

    def sample(self, n: int) -> torch.Tensor:
        return self.data[torch.randint(0, len(self.data), (n,), device=self.data.device)]


def _rms(x: torch.Tensor) -> torch.Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).add(1e-10).sqrt()


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.empty(n, 1, device=device).uniform_(lo, hi)


class Augmenter:
    def __init__(self, config: AugmentConfig = AugmentConfig(), noise: NoiseBank | None = None,  # noqa: B008
                 babble: NoiseBank | None = None, rirs: RirBank | None = None) -> None:
        self.config, self.noise, self.babble, self.rirs = config, noise, babble, rirs

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T] clean speech placed in the window (zeros elsewhere)."""
        cfg, (b, t), dev = self.config, x.shape, x.device
        # Level: speech RMS measured over its active region (the window is mostly padding).
        active = (x.abs() > 1e-4).float().mean(dim=-1, keepdim=True).clamp(min=0.1)
        x = x / (_rms(x) / active.sqrt()) * 10 ** (_uniform(b, *cfg.level_dbfs, dev) / 20)

        if self.rirs is not None and cfg.p_reverb > 0:
            use = torch.rand(b, 1, device=dev) < cfg.p_reverb
            rir = self.rirs.sample(b)
            n = t + rir.shape[-1]
            wet = torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(rir, n), n)[:, :t]
            wet = wet * (_rms(x) / _rms(wet))
            x = torch.where(use, wet, x)

        if cfg.p_eq > 0:  # smooth random spectral bumps: microphone and device colour
            use = torch.rand(b, 1, device=dev) < cfg.p_eq
            spec = torch.fft.rfft(x)
            knots = torch.empty(b, 1, 8, device=dev).uniform_(-cfg.eq_db, cfg.eq_db)
            curve = torch.nn.functional.interpolate(knots, size=spec.shape[-1], mode="linear",
                                                    align_corners=True)[:, 0]
            x = torch.where(use, torch.fft.irfft(spec * 10 ** (curve / 20), t), x)

        speech_rms = _rms(x) / active.sqrt()
        if self.noise is not None and cfg.p_noise > 0:
            use = torch.rand(b, 1, device=dev) < cfg.p_noise
            noise = self.noise.sample(b, t)
            if self.babble is not None:
                babble = torch.rand(b, 1, device=dev) < cfg.p_babble
                noise = torch.where(babble, self.babble.sample(b, t), noise)
            snr = _uniform(b, *cfg.snr_db, dev)
            noise = noise / _rms(noise) * speech_rms / 10 ** (snr / 20)
            x = x + noise * use

        floor = 10 ** (_uniform(b, *cfg.floor_dbfs, dev) / 20)
        x = x + torch.randn_like(x) * floor
        return x.clamp(-1.0, 1.0)
