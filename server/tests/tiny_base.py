"""A tiny but real base model (the files training.export writes), built in seconds, so the
whole server flow runs in CI and in the web app's end-to-end tests without the multi-GB release.

    python -m server.tests.tiny_base <models-dir> [--fixtures <dir>]

builds <models-dir>/base/tiny (unless present) and, with --fixtures, the WAVs the end-to-end
tests upload: a take of one word said 6 times, other speech, ambience.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from core.calibrate import LongformCache
from core.encoder import Encoder, EncoderConfig
from core.export import export_module, quantize_int8
from core.finetune import WINDOW, Bank
from core.verify import stream_embeddings
from runtime.vad import SileroVad
from runtime.wav import read_wav, write_wav

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



def take(speech: np.ndarray, n: int = 6) -> np.ndarray:
    """One word of speech.wav said n times, 0.8 s apart, at varying levels."""
    from server import audio
    from server.basemodel import VAD_FILE

    word = next(c for c in audio.split_positives(speech, SileroVad(VAD_FILE))
                if c.accepted and c.speech_s < 0.9)
    gap = np.zeros(int(0.8 * SR), np.float32)
    rng = np.random.default_rng(0)
    return np.concatenate([gap] + [x for _ in range(n)
                                   for x in (word.audio * rng.uniform(0.2, 0.9), gap)])


def write_fixtures(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    speech = read_wav(AUDIO / "speech.wav")[0]
    write_wav(out / "take.wav", take(speech))
    write_wav(out / "other_speech.wav", speech)
    write_wav(out / "ambience.wav", read_wav(AUDIO / "noise.wav")[0])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("models_dir", type=Path)
    p.add_argument("--fixtures", type=Path)
    args = p.parse_args(argv)
    base = args.models_dir / "base" / "tiny"
    if not (base / "base_model.json").exists():
        build_tiny_base(base)
    if args.fixtures:
        write_fixtures(args.fixtures)


if __name__ == "__main__":
    main()
