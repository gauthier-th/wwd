"""Assemble a base-model release from a training checkpoint (PLAN.md §7 step 8).

    uv run --extra training python -m training.export models/runs/<run>/best.pt --version v0.1

Writes models/base/<version>/:
  encoder.pt              encoder config + weights (torch, for fine-tuning)
  encoder.onnx            [N, 24000] -> embedding, checked by core.export
  encoder.int8.onnx       static QDQ int8, frontend kept in float
  neg_bank.npz            augmented embeddings of the 200k bank clips (core.finetune.Bank)
  longform_cache.npz      per-tick embeddings of the calibration streams (core.calibrate)
  longform_verify.npz     raw audio of the held-out verification streams
  augment_assets.npz      noise / babble / RIR subsets so the CPU server can augment
  base_model.json         version, sources, file hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.augment import Augmenter
from core.calibrate import LongformCache
from core.encoder import Encoder
from core.export import export_module, quantize_int8
from core.finetune import WINDOW, Bank
from runtime.spec import FORMAT_VERSION
from training.augment import build_augmenter, musan_audio, openslr_rirs
from training.build_bank import BANK, LONGFORM
from training.evaluate import load_encoder
from training.paths import MODELS

HOP = 1536


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def embed_windows(encoder: Encoder, windows: torch.Tensor, augmenter: Augmenter | None,
                  batch: int = 1024) -> np.ndarray:
    out = []
    for i in range(0, len(windows), batch):
        x = windows[i : i + batch].cuda(non_blocking=True)
        if augmenter is not None:
            x = augmenter(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(encoder(x).float().cpu())
    return torch.cat(out).numpy()


def stream_windows(audio: np.ndarray) -> torch.Tensor:
    """Windows of every tick of a stream, exactly as runtime/SPEC.md §3 builds them."""
    pcm = torch.from_numpy(audio.astype(np.float32) / 32768.0)
    n_ticks = len(pcm) // HOP
    padded = torch.cat([torch.zeros(WINDOW), pcm])
    return padded.unfold(0, WINDOW, HOP)[1 : n_ticks + 1]


def embed_bank(encoder: Encoder, augmenter: Augmenter) -> Bank:
    manifest = pd.read_parquet(BANK / "manifest.parquet")
    audio = np.load(BANK / "audio.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    chunks = []
    for i in range(0, len(audio), 8192):
        clips = torch.from_numpy(audio[i : i + 8192].astype(np.float32) / 32768.0)
        windows = torch.zeros(len(clips), WINDOW)
        starts = rng.integers(0, WINDOW - clips.shape[1] + 1, len(clips))
        for j, s in enumerate(starts):
            windows[j, s : s + clips.shape[1]] = clips[j]
        chunks.append(embed_windows(encoder, windows, augmenter))
    emb = np.concatenate(chunks)
    return Bank(emb / np.linalg.norm(emb, axis=1, keepdims=True),
                manifest.word.to_numpy(), manifest.lang.to_numpy())


def embed_longform(encoder: Encoder) -> tuple[LongformCache, dict[str, np.ndarray]]:
    streams = pd.read_parquet(LONGFORM / "streams.parquet")
    occ = pd.read_parquet(LONGFORM / "occurrences.parquet")
    embs, starts, occ_words, occ_ticks, verify = [], [0], [], [], {}
    for s in streams.itertuples():
        audio = np.load(LONGFORM / "streams" / f"{s.id}.npy")
        if s.split == "verify":
            verify[s.id] = audio
            continue
        windows = stream_windows(audio)
        base = starts[-1]
        embs.append(embed_windows(encoder, windows, None))
        starts.append(base + len(windows))
        for o in occ[occ.id == s.id].itertuples():
            first = o.start // HOP  # first tick whose window ends after the word starts
            stop = min(len(windows), -(-(o.end + WINDOW) // HOP) - 1)
            occ_words.append(o.word)
            occ_ticks.append((base + first, base + max(first, stop)))
    emb = np.concatenate(embs)
    cache = LongformCache(emb / np.linalg.norm(emb, axis=1, keepdims=True), np.array(starts),
                          np.array(occ_words, dtype=str), np.array(occ_ticks, np.int64).reshape(-1, 2))
    return cache, verify


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--version", required=True)
    args = p.parse_args(argv)
    out = MODELS / "base" / args.version
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder = load_encoder(args.checkpoint)
    torch.save({"encoder_config": ckpt["encoder_config"], "encoder": encoder.state_dict()},
               out / "encoder.pt")

    meta = {"format_version": FORMAT_VERSION, "base_model_version": args.version,
            "embedding_dim": str(encoder.config.embedding_dim)}
    cpu_encoder = load_encoder(args.checkpoint, device="cpu")
    export_module(cpu_encoder, out / "encoder.onnx", output_name="embedding", metadata=meta)
    bank_audio = np.load(BANK / "audio.npy", mmap_mode="r")
    calib = np.zeros((256, WINDOW), np.float32)
    pick = np.random.default_rng(0).choice(len(bank_audio), 256, replace=False)
    calib[:, 4000:20000] = bank_audio[np.sort(pick)].astype(np.float32) / 32768.0
    quantize_int8(out / "encoder.onnx", out / "encoder.int8.onnx", calib, output_name="embedding")
    print(f"onnx exported ({time.time() - started:.0f}s)")

    augmenter = build_augmenter("cuda")
    embed_bank(encoder, augmenter).save(out / "neg_bank.npz")
    print(f"bank embedded ({time.time() - started:.0f}s)")
    cache, verify = embed_longform(encoder)
    cache.save(out / "longform_cache.npz")
    np.savez(out / "longform_verify.npz", **verify)
    print(f"long-form: {cache.hours:.1f} h calibration, {len(verify)} verification streams "
          f"({time.time() - started:.0f}s)")

    np.savez(out / "augment_assets.npz",
             noise=musan_audio(("noise", "music"), 0.5, seed=7).astype(np.float16),
             babble=musan_audio(("speech",), 1 / 6, seed=8).astype(np.float16),
             rirs=np.stack([np.pad(r, (0, 8000 - len(r))) for r in openslr_rirs(500, seed=9)])
             .astype(np.float16))

    files = sorted(f.name for f in out.iterdir() if f.is_file() and f.name != "base_model.json")
    manifest = {
        "version": args.version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint": str(args.checkpoint),
        "step": ckpt["step"],
        "train_args": ckpt["args"],
        "encoder_config": asdict(encoder.config),
        "longform_hours": round(cache.hours, 2),
        "bank_clips": len(bank_audio),
        "files": {f: sha256(out / f) for f in files},
    }
    (out / "base_model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"base model {args.version} -> {out} ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
