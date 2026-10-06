"""Load the augmentation banks for base-model training from MUSAN and OpenSLR RIRs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.augment import AugmentConfig, Augmenter, NoiseBank, RirBank
from training.paths import MUSAN, RIRS
from training.store import read_wav_16k


def _concat(files: list[Path], max_seconds: float, rng: np.random.Generator) -> np.ndarray:
    out, total = [], 0.0
    for f in rng.permutation(files):
        a = read_wav_16k(f)
        out.append(a)
        total += len(a) / 16000
        if total >= max_seconds:
            break
    return np.concatenate(out)


def musan_audio(kinds: tuple[str, ...], hours: float, seed: int = 0) -> np.ndarray:
    files = [f for k in kinds for f in sorted((MUSAN / k).rglob("*.wav"))]
    if not files:
        raise FileNotFoundError(f"no MUSAN {kinds} audio under {MUSAN}")
    return _concat(files, hours * 3600, np.random.default_rng(seed))


def openslr_rirs(count: int, seed: int = 0) -> list[np.ndarray]:
    files = sorted((RIRS / "simulated_rirs").rglob("*.wav")) + sorted(
        (RIRS / "real_rirs_isotropic_noises").rglob("*rir*.wav"))
    if not files:
        raise FileNotFoundError(f"no RIRs under {RIRS}")
    picked = np.random.default_rng(seed).choice(len(files), min(count, len(files)), replace=False)
    return [read_wav_16k(files[i])[:8000] for i in picked]


def build_augmenter(device: torch.device | str, noise_hours: float = 3.0) -> Augmenter:
    """Training augmenter; missing datasets are skipped with a warning (for smoke tests)."""
    banks: dict[str, NoiseBank | RirBank | None] = {}
    loaders = {
        "noise": lambda: NoiseBank(musan_audio(("noise", "music"), noise_hours), device),
        "babble": lambda: NoiseBank(musan_audio(("speech",), noise_hours / 3, seed=1), device),
        "rirs": lambda: RirBank(openslr_rirs(4000), device),
    }
    for name, load in loaders.items():
        try:
            banks[name] = load()
        except FileNotFoundError as err:
            print(f"augment: {err}; training without {name}")
            banks[name] = None
    return Augmenter(AugmentConfig(), banks["noise"], banks["babble"], banks["rirs"])
