"""Regenerate runtime/testdata: golden audio, stub models and expected values.

    uv run --extra training python -m scripts.make_golden
    uv run --extra training python -m scripts.make_golden --words-only   # just the "words" part

Expected values come from independent oracles, never from the runtimes under test:
  - VAD: the official `silero-vad` package wrapper;
  - fp32 scores and embeddings: the PyTorch modules;
  - int8 scores: onnxruntime (Python) on the int8 file;
  - word files' scores: onnxruntime (Python) on each whole file (the runtimes split them);
  - gate and events: the plain-loop reference at the bottom of this file.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from core.encoder import Encoder
from core.export import compose_head, export_module, quantize_int8, read_metadata, set_metadata
from core.model import Head, PrototypeHead, WakeWordModel
from runtime.spec import FORMAT_VERSION, VAD_CONTEXT, VAD_FRAME, DetectorConfig
from runtime.wav import read_wav, write_wav

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runtime" / "testdata"
VAD_PATH = ROOT / "runtime" / "assets" / "silero_vad_16k_op15.onnx"
SR = 16000
SEED = 1234
MSWC = "https://huggingface.co/datasets/MLCommons/ml_spoken_words/resolve/main/data/wav/{}/train/audio/0.tar.gz"
# Every golden score sits this far from the threshold, so the per-tick drift allowed between
# runtimes (TOLERANCE) can never flip a trigger. int8 kernels differ across platforms (wasm,
# x86 VNNI, ARM dot-product): a rare one-step requantization difference moves a score ~1e-2.
TOLERANCE = {"vad": 1e-5, "fp32": 1e-4, "int8": 2e-2, "int8_mean": 1e-3, "embedding": 1e-4}
MARGIN = {"fp32": 0.006, "int8": 0.025}
MARGIN_VAD = 1e-3
# Several wake words (runtime/SPEC.md §5): two files composed like shipped ones, on the same
# encoder.onnx, so a detector listening for both shares it. Their settings differ, so each
# word's gate and trigger are its own.
WORDS = {
    "word_a.onnx": ("golden-a", {}),
    "word_b.onnx": ("golden-b", {"smoothing_k": 1, "smoothing_n": 2, "refractory_ms": 1000,
                                 "vad_threshold": 0.6, "vad_hangover_ms": 1000}),
}


# --------------------------------------------------------------------------- audio


def fetch_mswc_clips(lang: str, n_words: int) -> list[tuple[str, np.ndarray]]:
    """First clips of shard 0 (one per word), via an HTTP range request on the tarball."""
    req = urllib.request.Request(MSWC.format(lang), headers={"Range": "bytes=0-2499999"})
    data = urllib.request.urlopen(req, timeout=60).read()
    clips: dict[str, tuple[str, bytes]] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r|gz") as tf:
            for member in tf:
                if not (member.isfile() and member.name.endswith(".wav")):
                    continue
                name = member.name.rsplit("/", 1)[-1]
                word = name.split("_common_voice")[0]
                if word not in clips:
                    clips[word] = (name, tf.extractfile(member).read())
                if len(clips) == n_words:
                    break
    except (tarfile.ReadError, EOFError, OSError):
        pass  # the range ends mid-archive; keep the complete members
    out = []
    for word, (name, raw) in clips.items():
        tmp = OUT / "_tmp.wav"
        tmp.write_bytes(raw)
        audio, rate = read_wav(tmp)
        tmp.unlink()
        assert rate == SR, (name, rate)
        out.append((f"{lang}/{name}", audio))
    return out


def build_speech(rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
    clips = fetch_mswc_clips("fr", 5) + fetch_mswc_clips("de", 5) + fetch_mswc_clips("es", 5)
    parts = [np.zeros(SR, dtype=np.float32)]
    for i, (_, clip) in enumerate(clips):
        clip = clip / max(np.abs(clip).max(), 1e-3) * rng.uniform(0.2, 0.7)
        parts.append(clip.astype(np.float32))
        gap = 3.0 if i in (4, 9) else rng.uniform(0.15, 1.2)  # two long pauses close the gate
        parts.append(np.zeros(int(gap * SR), dtype=np.float32))
    audio = np.concatenate(parts)
    audio += rng.normal(0, 10 ** (-60 / 20), audio.size).astype(np.float32)  # -60 dBFS floor
    return audio, [name for name, _ in clips]


def build_noise(rng: np.random.Generator) -> np.ndarray:
    t = np.arange(8 * SR) / SR
    white = rng.normal(0, 1, t.size)
    pink = np.cumsum(white)
    pink = (pink - np.convolve(pink, np.ones(400) / 400, mode="same")) * 0.02  # crude 1/f-ish
    audio = 0.05 * pink
    audio[2 * SR : 4 * SR] += 0.2 * np.sin(2 * np.pi * 440 * t[2 * SR : 4 * SR])
    audio[5 * SR : 6 * SR] = rng.normal(0, 0.1, SR)  # loud white-noise burst
    return np.clip(audio, -0.95, 0.95).astype(np.float32)


# --------------------------------------------------------------------------- oracles


def oracle_vad(audio: np.ndarray) -> np.ndarray:
    from silero_vad import load_silero_vad

    vad = load_silero_vad(onnx=True)
    vad.reset_states()
    frames = len(audio) // VAD_FRAME
    assert VAD_CONTEXT == 64
    return np.array([
        float(vad(torch.from_numpy(audio[i * VAD_FRAME : (i + 1) * VAD_FRAME].copy()), SR))
        for i in range(frames)
    ])


def tick_windows(audio: np.ndarray, cfg: DetectorConfig) -> np.ndarray:
    n_ticks = (len(audio) // VAD_FRAME) // cfg.frames_per_tick
    padded = np.concatenate([np.zeros(cfg.window_samples, dtype=np.float32), audio])
    ends = [(j + 1) * cfg.hop_samples for j in range(n_ticks)]
    return np.stack([padded[e : e + cfg.window_samples] for e in ends])


def reference_gate(vad: np.ndarray, cfg: DetectorConfig) -> tuple[np.ndarray, np.ndarray]:
    n_ticks = len(vad) // cfg.frames_per_tick
    vad_max = np.array([
        vad[max(0, (j + 1) * cfg.frames_per_tick - cfg.gate_frames) : (j + 1) * cfg.frames_per_tick].max()
        for j in range(n_ticks)
    ])
    return vad_max, vad_max >= cfg.vad_threshold


def reference_events(scores: np.ndarray, cfg: DetectorConfig) -> list[int]:
    events, next_allowed = [], 0
    for j in range(len(scores)):
        recent = scores[max(0, j - cfg.smoothing_n + 1) : j + 1]
        if j >= next_allowed and (recent >= cfg.threshold).sum() >= cfg.smoothing_k:
            events.append(j)
            next_allowed = j + cfg.refractory_ticks + 1
    return events


def train_stub(model: WakeWordModel, windows: np.ndarray, targets: np.ndarray, steps: int = 600) -> None:
    """Short training on speech-vs-not, so BatchNorm statistics and dynamic range look like a
    trained model's. Noise/gain augmentation + weight decay keep it smooth: an overfit stub
    turns float rounding differences between runtimes into large score differences."""
    device = "cpu"  # deterministic, so regenerating the golden files is reproducible
    torch.manual_seed(SEED)
    model.to(device).train()
    x = torch.from_numpy(windows).to(device)
    y = torch.from_numpy(targets.astype(np.float32)).to(device) * 0.9 + 0.05  # label smoothing
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (32,), generator=gen).to(device)
        gain = 10 ** (torch.empty(32, 1, device=device).uniform_(-12, 6) / 20)
        noise_db = torch.empty(32, 1, device=device).uniform_(-80, -40)
        batch = x[idx] * gain + torch.randn_like(x[idx]) * 10 ** (noise_db / 20)
        logits = model.head(model.encoder(batch))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.cpu().eval()


def sensitivity(model: WakeWordModel, windows: np.ndarray) -> tuple[float, float]:
    """Max score change under (a) float32-rounding-scale relative noise, which is what differs
    between runtimes, and (b) inaudible -90 dBFS additive noise, for information."""
    gen = torch.Generator().manual_seed(SEED)
    x = torch.from_numpy(windows)
    with torch.no_grad():
        ref = model(x)
        rounding = model(x * (1 + 1e-6 * torch.randn(x.shape, generator=gen)))
        noise = model(x + torch.randn(x.shape, generator=gen) * 10 ** (-90 / 20))
    return float((rounding - ref).abs().max()), float((noise - ref).abs().max())


def pick_threshold(fp32: np.ndarray, int8: np.ndarray | None = None) -> float:
    """Threshold leaving 5-40% of ticks above it, as far as possible from every score
    (distance measured in units of each kind's required margin)."""
    candidates = np.linspace(0.05, 0.95, 1801)
    above = np.array([(fp32 >= t).mean() for t in candidates])
    clearance = np.abs(fp32[None] - candidates[:, None]).min(1) / MARGIN["fp32"]
    if int8 is not None:
        clearance = np.minimum(clearance, np.abs(int8[None] - candidates[:, None]).min(1) / MARGIN["int8"])
    clearance[(above < 0.05) | (above > 0.40)] = -1
    best = int(np.argmax(clearance))
    assert clearance[best] >= 1, f"no clean threshold (clearance {clearance[best]:.2f})"
    return round(float(candidates[best]), 4)


def word_heads(embeddings: np.ndarray) -> list[torch.nn.Module]:
    """The head the app trains (a prototype: one Gemm) and an MLP, scaled so that their scores
    spread over (0, 1) on these embeddings."""
    anchor = embeddings[int(np.argmax(embeddings @ embeddings.mean(0)))]
    cos = embeddings @ anchor
    prototype = PrototypeHead(torch.from_numpy(anchor), 20.0, -20.0 * float(np.quantile(cos, 0.8)))
    torch.manual_seed(SEED)
    mlp = Head(embeddings.shape[1])
    with torch.no_grad():
        logits = mlp(torch.from_numpy(embeddings))
        out = mlp.net[2]
        out.weight *= 2.5 / logits.std()
        out.bias.copy_((out.bias - logits.mean()) * 2.5 / logits.std() - 1.0)
    return [prototype, mlp]


def make_words(audio: dict[str, np.ndarray], windows: dict[str, np.ndarray],
               vad: dict[str, np.ndarray]) -> dict:
    models = OUT / "models"
    encoder = ort.InferenceSession(str(models / "encoder.onnx"), providers=["CPUExecutionProvider"])
    heads = word_heads(encoder.run(None, {"audio": windows["speech.wav"]})[0])
    out = {}
    for (name, (wake_word, settings)), head in zip(WORDS.items(), heads, strict=True):
        cfg = DetectorConfig().with_overrides(**settings)
        meta = {"format_version": FORMAT_VERSION, "wake_word": wake_word, "language": "xx",
                "base_model_version": "stub", "created_at": "1970-01-01T00:00:00Z",
                "vad_file": VAD_PATH.name}
        compose_head(models / "encoder.onnx", head, models / name, metadata={**meta, **cfg.to_metadata()})
        sess = ort.InferenceSession(str(models / name), providers=["CPUExecutionProvider"])
        scores = {n: sess.run(None, {"audio": w})[0].astype(np.float64) for n, w in windows.items()}
        cfg = cfg.with_overrides(threshold=pick_threshold(np.concatenate(list(scores.values()))))
        proto = onnx.load(str(models / name))
        set_metadata(proto, cfg.to_metadata())
        onnx.save(proto, str(models / name))
        entry: dict = {"config": cfg.to_metadata(), "files": {}}
        for n in audio:
            vad_max, gate_open = reference_gate(vad[n], cfg)
            assert np.abs(vad_max - cfg.vad_threshold).min() >= MARGIN_VAD, f"{name}/{n}: VAD too close"
            entry["files"][n] = {
                "vad_max": [round(float(v), 7) for v in vad_max],
                "gate_open": [bool(g) for g in gate_open],
                "scores": [round(float(v), 7) for v in scores[n]],
                "events": {"gate": reference_events(np.where(gate_open, scores[n], 0.0), cfg),
                           "no_gate": reference_events(scores[n], cfg)},
            }
            print(f"{name} {n}: gate open {gate_open.mean():.0%}, events {entry['files'][n]['events']}")
        out[name] = entry
    return out


# --------------------------------------------------------------------------- main


def words_only() -> None:
    """Rebuild the word files and their values on the existing audio, encoder and VAD values."""
    golden = json.loads((OUT / "golden.json").read_text())
    audio = {n: read_wav(OUT / "audio" / n)[0] for n in golden["files"]}
    windows = {n: tick_windows(a, DetectorConfig()) for n, a in audio.items()}
    vad = {n: np.array(golden["files"][n]["vad"]) for n in audio}
    golden["words"] = make_words(audio, windows, vad)
    (OUT / "golden.json").write_text(json.dumps(golden, separators=(",", ":")) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--words-only", action="store_true", help="only rebuild the word files (WORDS)")
    if p.parse_args().words_only:
        return words_only()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)

    speech, sources = build_speech(rng)
    write_wav(OUT / "audio" / "speech.wav", speech)
    write_wav(OUT / "audio" / "noise.wav", build_noise(rng))
    audio = {name: read_wav(OUT / "audio" / name)[0] for name in ("speech.wav", "noise.wav")}

    cfg = DetectorConfig()
    windows = {name: tick_windows(a, cfg) for name, a in audio.items()}
    all_windows = np.concatenate(list(windows.values()))

    # Stub model: briefly trained to fire on windows whose last 0.5 s holds speech.
    vad = {name: oracle_vad(a) for name, a in audio.items()}
    targets = []
    tail = int(0.5 * SR / VAD_FRAME)
    for name, w in windows.items():
        ends = [(j + 1) * cfg.frames_per_tick for j in range(len(w))]
        targets += [vad[name][max(0, e - tail) : e].mean() > 0.5 for e in ends]
    model = WakeWordModel(Encoder(), Head())
    train_stub(model, all_windows, np.array(targets))
    rounding, noise = sensitivity(model, all_windows)
    print(f"stub sensitivity: 1e-6 relative {rounding:.2e}, -90 dBFS noise {noise:.4f}")
    assert rounding < 5e-5, "stub model is ill-conditioned; golden parity would be flaky"
    with torch.no_grad():
        fp32 = {n: model(torch.from_numpy(w)).numpy().astype(np.float64) for n, w in windows.items()}

    models = OUT / "models"
    meta = {
        "format_version": FORMAT_VERSION,
        "wake_word": "golden-stub",
        "language": "xx",
        "base_model_version": "stub",
        "created_at": "1970-01-01T00:00:00Z",
        "vad_file": VAD_PATH.name,
        **cfg.to_metadata(),
    }
    export_module(model, models / "model.onnx", output_name="score", metadata=meta)
    export_module(model.encoder, models / "encoder.onnx", output_name="embedding",
                  metadata={k: meta[k] for k in ("format_version", "base_model_version")})
    calib = all_windows[rng.choice(len(all_windows), 64, replace=False)]
    quantize_int8(models / "model.onnx", models / "model.int8.onnx", calib, output_name="score")

    opts = ort.SessionOptions()
    opts.add_session_config_entry("session.x64quantprecision", "1")  # exact on any x86 (SPEC §4)
    sess = ort.InferenceSession(str(models / "model.int8.onnx"), opts,
                                providers=["CPUExecutionProvider"])
    int8 = {n: sess.run(None, {"audio": w})[0].astype(np.float64) for n, w in windows.items()}

    q_err = max(np.abs(fp32[n] - int8[n]).max() for n in windows)
    print(f"int8 vs fp32 max |score diff|: {q_err:.4f}")
    threshold = pick_threshold(np.concatenate(list(fp32.values())), np.concatenate(list(int8.values())))
    cfg = cfg.with_overrides(threshold=threshold)
    for f in ("model.onnx", "model.int8.onnx"):
        proto = onnx.load(str(models / f))
        set_metadata(proto, {**read_metadata(models / f), **cfg.to_metadata()})
        onnx.save(proto, str(models / f))

    golden: dict = {
        "generated_by": "scripts/make_golden.py",
        "config": cfg.to_metadata(),
        "tolerance": TOLERANCE,
        "files": {},
    }
    for name in audio:
        vad_max, gate_open = reference_gate(vad[name], cfg)
        n = len(fp32[name])
        assert len(vad_max) == n
        assert np.abs(vad_max - cfg.vad_threshold).min() >= MARGIN_VAD, f"{name}: VAD too close"
        for kind, scores in (("fp32", fp32[name]), ("int8", int8[name])):
            assert np.abs(scores - threshold).min() >= MARGIN[kind], f"{name}/{kind} too close"
        events = {
            kind: {
                "gate": reference_events(np.where(gate_open, scores, 0.0), cfg),
                "no_gate": reference_events(scores, cfg),
            }
            for kind, scores in (("fp32", fp32[name]), ("int8", int8[name]))
        }
        golden["files"][name] = {
            "vad": [round(float(v), 7) for v in vad[name]],
            "vad_max": [round(float(v), 7) for v in vad_max],
            "gate_open": [bool(g) for g in gate_open],
            "scores": {k: [round(float(v), 7) for v in s[name]] for k, s in (("fp32", fp32), ("int8", int8))},
            "events": events,
        }
        print(f"{name}: {n} ticks, gate open {gate_open.mean():.0%}, events {events}")

    # Encoder embeddings for a few windows of the speech file.
    pick = np.linspace(0, len(windows["speech.wav"]) - 1, 6).astype(int)
    with torch.no_grad():
        emb = model.encoder(torch.from_numpy(windows["speech.wav"][pick])).numpy()
    golden["encoder"] = {
        "file": "speech.wav",
        "ticks": [int(i) for i in pick],
        "embeddings": [[round(float(v), 7) for v in row] for row in emb],
    }
    golden["words"] = make_words(audio, windows, vad)
    (OUT / "golden.json").write_text(json.dumps(golden, separators=(",", ":")) + "\n")
    (OUT / "SOURCES.txt").write_text(
        "speech.wav is built from these MSWC clips (Multilingual Spoken Words Corpus, CC-BY 4.0,\n"
        "https://mlcommons.org/datasets/multilingual-spoken-words/), rescaled, with gaps and a\n"
        "-60 dBFS noise floor added. noise.wav is synthetic.\n\n" + "\n".join(sources) + "\n"
    )
    print(f"threshold {threshold}; wrote {OUT}")


if __name__ == "__main__":
    main()
