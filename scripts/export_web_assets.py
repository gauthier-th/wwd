"""Static assets for the browser-only app: everything quick mode needs from a base model,
precomputed as flat binary files that the web app fetches (web/src/local/assets.ts).

    uv run --extra training python -m scripts.export_web_assets models/base/v1
    uv run --extra training python -m scripts.export_web_assets models/base/v1 --variant data/v1-web

Writes <out>/<version>/ (default web/public/base/, served as static files):
  manifest.json         sizes, quantization, template offsets, languages, base-model summary
  encoder.onnx          the base encoder, for embedding recordings in the browser
  model_template.onnx   encoder + a marker head: the browser writes the trained head's weights
                        at `template.weight_offset` / `bias_offset` and appends the metadata
  bank.*                negative bank: int8 embeddings + per-row scale, word and language ids
  cache.*               calibration ticks: int8 embeddings + per-row scale, stream starts, and
                        where MSWC words were placed (so the user's own word isn't counted)
  verify.*              held-out verification ticks (embeddings through encoder.onnx on
                        onnxruntime, exactly what the shipped file computes) + VAD per frame
  noise.i16, babble.i16, rirs.i16   augmentation audio (RIRs aligned and peak-normalized)

--variant DIR also writes the same reduced, quantized data back as a base-model directory the
Python tools load (training.eval_e2e), to measure what the reduction costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from core.calibrate import LongformCache
from core.export import compose_head
from core.finetune import Bank, normalize_word
from core.model import PrototypeHead
from core.verify import parallel, stream_embeddings, stream_vad
from server.basemodel import VAD_FILE
from server.languages import LANGUAGES

REPO = Path(__file__).resolve().parents[1]
VAD_LICENSE = REPO / "runtime" / "assets" / "LICENSE-silero-vad"


def quantize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row symmetric int8: x ~ q * scale. Rows are unit vectors, so the dot-product error
    stays ~1e-3, far below what moves a calibrated threshold."""
    x = x.astype(np.float32)
    scale = np.abs(x).max(axis=1) / 127 + 1e-12
    return np.round(x / scale[:, None]).astype(np.int8), scale.astype(np.float32)


def dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * scale[:, None]


def subset_bank(bank: Bank, size: int, rng: np.random.Generator) -> Bank:
    if size >= len(bank.words):
        return bank
    keep = np.sort(rng.choice(len(bank.words), size, replace=False))
    return Bank(bank.embeddings[keep], bank.words[keep], bank.langs[keep])


def subset_cache(cache: LongformCache, hours: float) -> LongformCache:
    """Every k-th stream, so the mix of word salad, read speech and noise is kept."""
    streams = len(cache.stream_starts) - 1
    keep_n = min(streams, max(1, round(streams * hours / cache.hours)))
    keep = np.unique(np.linspace(0, streams - 1, keep_n).round().astype(int))
    parts, starts, words, ticks = [], [0], [], []
    for s in keep:
        a, b = cache.stream_starts[s], cache.stream_starts[s + 1]
        shift = starts[-1] - a
        parts.append(cache.embeddings[a:b])
        starts.append(starts[-1] + b - a)
        inside = (cache.occ_ticks[:, 0] >= a) & (cache.occ_ticks[:, 1] <= b)
        words.append(cache.occ_words[inside])
        ticks.append(cache.occ_ticks[inside] + shift)
    return LongformCache(np.concatenate(parts), np.array(starts), np.concatenate(words),
                         np.concatenate(ticks).reshape(-1, 2), cache.hop_samples)


def aligned_rirs(rirs: np.ndarray, length: int = 8000) -> np.ndarray:
    """As core.augment.RirBank: direct path at 0, peak-normalized."""
    out = np.zeros((len(rirs), length), np.float32)
    for i, r in enumerate(rirs.astype(np.float32)):
        r = r[int(np.argmax(np.abs(r))):][:length]
        out[i, : len(r)] = r / (np.abs(r).max() + 1e-9)
    return out


def to_i16(x: np.ndarray) -> np.ndarray:
    return np.round(np.clip(x.astype(np.float32), -1, 1) * 32767).astype(np.int16)


def word_table(words: np.ndarray) -> tuple[list[str], np.ndarray]:
    """Normalized (NFC, casefolded) unique words and each row's index into them."""
    norm = np.array([normalize_word(str(w)) for w in words])
    table, idx = np.unique(norm, return_inverse=True)
    return [str(w) for w in table], idx.astype(np.uint32)


def template(encoder: Path, out: Path, dim: int) -> dict:
    """encoder + a head with marker weights; returns where the browser writes the real ones."""
    marker = (np.arange(dim, dtype=np.float32) + 1) * np.float32(1.0e-3) + np.float32(0.123456)
    head = PrototypeHead(torch.from_numpy(marker), 1.0, 0.0)
    with torch.no_grad():
        head.linear.bias.fill_(-7.654321)
    compose_head(encoder, head, out, metadata={})
    data = out.read_bytes()
    w, b = marker.tobytes(), np.float32(-7.654321).tobytes()
    if data.count(w) != 1 or data.count(b) != 1:
        raise RuntimeError("template markers are not unique in the composed graph")
    return {"weight_offset": data.index(w), "bias_offset": data.index(b), "dim": dim}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("base", type=Path)
    p.add_argument("--out", type=Path, default=REPO / "web" / "public" / "base")
    p.add_argument("--bank-size", type=int, default=200_000)
    p.add_argument("--cache-hours", type=float, default=100.0, help="default: all of it")
    p.add_argument("--noise-min", type=float, default=5.0)
    p.add_argument("--babble-min", type=float, default=2.0)
    p.add_argument("--rirs", type=int, default=200)
    p.add_argument("--finetune", default="{}",
                   help="JSON overrides of core.finetune.FinetuneConfig for the browser (tests)")
    p.add_argument("--variant", type=Path, default=None,
                   help="also write the reduced data as a Python base-model directory")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = p.parse_args(argv)
    started = time.time()
    rng = np.random.default_rng(0)
    manifest_in = json.loads((args.base / "base_model.json").read_text())
    version = manifest_in["version"]
    out = args.out / version
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    def write(name: str, array: np.ndarray) -> dict:
        array = np.ascontiguousarray(array)
        (out / name).write_bytes(array.tobytes())
        return {"file": name, "dtype": str(array.dtype), "shape": list(array.shape)}

    def write_json(name: str, value: object) -> str:
        (out / name).write_text(json.dumps(value, ensure_ascii=False) + "\n")
        return name

    bank = subset_bank(Bank.load(args.base / "neg_bank.npz"), args.bank_size, rng)
    dim = bank.embeddings.shape[1]
    shutil.copyfile(args.base / "encoder.onnx", out / "encoder.onnx")
    tpl = template(args.base / "encoder.onnx", out / "model_template.onnx", dim)

    bank_q, bank_s = quantize(bank.embeddings)
    bank_words, bank_word_idx = word_table(bank.words)
    langs = sorted(LANGUAGES)
    bank_lang = np.array([langs.index(str(lang)) for lang in bank.langs], np.uint8)

    cache = subset_cache(LongformCache.load(args.base / "longform_cache.npz"), args.cache_hours)
    cache_q, cache_s = quantize(cache.embeddings)
    occ_words, occ_idx = word_table(cache.occ_words)

    encoder_bytes = (args.base / "encoder.onnx").read_bytes()
    z = np.load(args.base / "longform_verify.npz")
    streams = [z[k] for k in sorted(z.files)]
    v_emb = parallel(lambda a: stream_embeddings(encoder_bytes, a), streams, args.workers)
    v_vad = parallel(lambda a: stream_vad(VAD_FILE.read_bytes(), a), streams, args.workers)
    v_starts = np.cumsum([0] + [len(e) for e in v_emb])
    v_q, v_s = quantize(np.concatenate(v_emb))
    vad_frames = np.concatenate([v[: 3 * len(e)] for v, e in zip(v_vad, v_emb, strict=True)])
    print(f"verification: {len(v_starts) - 1} streams, {v_starts[-1]:,} ticks ({time.time() - started:.0f}s)")

    aug = np.load(args.base / "augment_assets.npz")
    noise = aug["noise"][: int(args.noise_min * 60 * 16000)]
    babble = aug["babble"][: int(args.babble_min * 60 * 16000)]
    rirs = aligned_rirs(aug["rirs"][: args.rirs])

    manifest = {
        "format": 1,
        "version": version,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_created_at": manifest_in["created_at"],
        "embedding_dim": dim,
        "finetune": json.loads(args.finetune),
        "template": tpl,
        "languages": langs,
        "bank": {"embeddings": write("bank.i8", bank_q), "scales": write("bank.scale.f32", bank_s),
                 "word": write("bank.word.u32", bank_word_idx), "lang": write("bank.lang.u8", bank_lang),
                 "words": write_json("bank.words.json", bank_words)},
        "cache": {"embeddings": write("cache.i8", cache_q), "scales": write("cache.scale.f32", cache_s),
                  "stream_starts": write("cache.streams.u32", cache.stream_starts.astype(np.uint32)),
                  "occ_word": write("cache.occ_word.u32", occ_idx),
                  "occ_ticks": write("cache.occ_ticks.u32", cache.occ_ticks.astype(np.uint32)),
                  "occ_words": write_json("cache.occ_words.json", occ_words), "hours": cache.hours,
                  "hop_samples": cache.hop_samples},
        "verify": {"embeddings": write("verify.i8", v_q), "scales": write("verify.scale.f32", v_s),
                   "stream_starts": write("verify.streams.u32", v_starts.astype(np.uint32)),
                   "vad": write("verify.vad.u8", np.round(vad_frames * 255).astype(np.uint8))},
        "augment": {"noise": write("noise.i16", to_i16(noise)), "babble": write("babble.i16", to_i16(babble)),
                    "rirs": write("rirs.i16", to_i16(rirs))},
        "base": {k: manifest_in.get(k) for k in ("version", "created_at", "encoder_config", "teacher",
                                                  "evaluation")},
        "languages_detail": {code: LANGUAGES[code] for code in langs},
    }
    shutil.copyfile(VAD_FILE, out / "vad.onnx")
    shutil.copyfile(VAD_LICENSE, out / "LICENSE-silero-vad")
    manifest["files"] = {f.name: {"bytes": f.stat().st_size,
                                  "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
                         for f in sorted(out.iterdir())}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False) + "\n")
    index_path = args.out / "index.json"  # the versions the app can load; the newest is its default
    index = json.loads(index_path.read_text()) if index_path.exists() else {"versions": []}
    index["versions"] = sorted({*index["versions"], version})
    index["default"] = version
    index_path.write_text(json.dumps(index) + "\n")
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"web assets {version} -> {out}: {total / 1e6:.1f} MB in {len(manifest['files'])} files "
          f"(bank {len(bank_q):,}, cache {cache.hours:.1f} h, verify {v_starts[-1]:,} ticks) "
          f"({time.time() - started:.0f}s)")

    if args.variant:
        v = args.variant
        v.mkdir(parents=True, exist_ok=True)
        for name in ("encoder.pt", "encoder.onnx", "encoder.int8.onnx", "longform_verify.npz",
                     "base_model.json"):
            shutil.copyfile(args.base / name, v / name)
        Bank(dequantize(bank_q, bank_s).astype(np.float16), bank.words, bank.langs).save(v / "neg_bank.npz")
        LongformCache(dequantize(cache_q, cache_s).astype(np.float16), cache.stream_starts,
                      cache.occ_words, cache.occ_ticks, cache.hop_samples).save(v / "longform_cache.npz")
        np.savez(v / "augment_assets.npz", noise=noise, babble=babble, rirs=aug["rirs"][: args.rirs])
        print(f"python variant -> {v}")


if __name__ == "__main__":
    main()
