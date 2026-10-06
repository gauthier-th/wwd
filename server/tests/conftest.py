"""A tiny but real base model (same files as training.export writes), built in seconds so the
whole server flow runs in CI without the multi-GB release."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

from core.calibrate import LongformCache
from core.encoder import Encoder, EncoderConfig
from core.export import export_module, quantize_int8
from core.finetune import WINDOW, Bank
from core.verify import stream_embeddings
from runtime.wav import read_wav
from server.settings import Settings

AUDIO = Path(__file__).resolve().parents[2] / "runtime" / "testdata" / "audio"
SR = 16000
HOP = 1536


def synthetic_speech(rng: np.random.Generator, seconds: float) -> np.ndarray:
    """Noise with bursts of harmonic tones: something for the encoder to look at."""
    t = np.arange(int(seconds * SR)) / SR
    out = rng.normal(0, 0.003, len(t))
    pos = 0.3
    while pos < seconds - 0.6:
        n = int(rng.uniform(0.2, 0.5) * SR)
        a = int(pos * SR)
        f0 = rng.uniform(100, 250)
        tone = sum(np.sin(2 * np.pi * f0 * k * t[a : a + n]) / k for k in (1, 2, 3))
        out[a : a + n] += 0.1 * tone * np.hanning(n)
        pos += rng.uniform(0.4, 1.0)
    return out.astype(np.float32)


def windows_of(stream: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.concatenate([np.zeros(WINDOW, np.float32), stream]))
    return x.unfold(0, WINDOW, HOP)[1 : len(stream) // HOP + 1]


def build_tiny_base(out: Path) -> None:
    out.mkdir(parents=True)
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    config = EncoderConfig(width=8, embedding_dim=32)
    encoder = Encoder(config).eval()
    torch.save({"encoder_config": asdict(config), "encoder": encoder.state_dict()}, out / "encoder.pt")
    meta = {"format_version": "1", "base_model_version": "tiny", "embedding_dim": "32"}
    export_module(encoder, out / "encoder.onnx", output_name="embedding", metadata=meta)
    calib = np.stack([synthetic_speech(rng, 1.5) for _ in range(8)])
    quantize_int8(out / "encoder.onnx", out / "encoder.int8.onnx", calib, output_name="embedding")

    with torch.no_grad():
        clips = torch.from_numpy(np.stack([synthetic_speech(rng, 1.5) for _ in range(300)]))
        bank = encoder(clips).numpy()
        streams = [synthetic_speech(rng, 40.0) for _ in range(2)]
        cache = [encoder(windows_of(s)).numpy() for s in streams]
    Bank(bank.astype(np.float16), np.array([f"w{i % 100}" for i in range(300)]),
         np.array(["en", "fr", "de"] * 100)).save(out / "neg_bank.npz")
    starts = np.cumsum([0] + [len(c) for c in cache])
    LongformCache(np.concatenate(cache), starts, np.array(["w1"]), np.array([[10, 14]])
                  ).save(out / "longform_cache.npz")
    int8 = [stream_embeddings(out / "encoder.int8.onnx", s) for s in streams]
    LongformCache(np.concatenate(int8), starts, np.array(["w1"]), np.array([[10, 14]])
                  ).save(out / "longform_cache.int8.npz")
    np.savez(out / "longform_verify.npz",
             **{f"s{i}": (synthetic_speech(rng, 30.0) * 32767).astype(np.int16) for i in range(2)})
    decay = np.exp(-np.arange(4000) / 400.0)
    np.savez(out / "augment_assets.npz",
             noise=rng.normal(0, 0.1, 5 * SR).astype(np.float16),
             babble=rng.normal(0, 0.1, 5 * SR).astype(np.float16),
             rirs=np.stack([np.pad(decay * rng.normal(size=4000), (0, 4000)) for _ in range(4)])
             .astype(np.float16))
    files = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.iterdir())}
    (out / "base_model.json").write_text(json.dumps({
        "version": "tiny", "created_at": "2026-01-01T00:00:00Z",
        "encoder_config": asdict(config), "longform_hours": 0.02, "files": files}))


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    build_tiny_base(root / "base" / "tiny")
    return root


@pytest.fixture
def settings(models_dir, tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "server", models_dir=models_dir, verify_workers=2,
                    finetune={"augment_per_clip": 8, "partial_per_clip": 4, "hard_negatives": 50,
                              "random_negatives": 200})


@pytest.fixture(scope="session")
def speech() -> np.ndarray:
    audio, rate = read_wav(AUDIO / "speech.wav")
    assert rate == SR
    return audio


@pytest.fixture(scope="session")
def noise() -> np.ndarray:
    return read_wav(AUDIO / "noise.wav")[0]
