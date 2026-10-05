"""Minimal 16-bit PCM WAV I/O (stdlib only, so the runtime needs no audio libraries)."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Return mono float32 samples in [-1, 1] and the sample rate."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported")
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        rate, channels = w.getframerate(), w.getnchannels()
    audio = data.reshape(-1, channels).mean(axis=1) if channels > 1 else data
    return (audio / 32768.0).astype(np.float32), rate


def write_wav(path: str | Path, audio: np.ndarray, rate: int = 16000) -> None:
    pcm = np.clip(np.round(np.asarray(audio, dtype=np.float64) * 32768.0), -32768, 32767)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype("<i2").tobytes())
