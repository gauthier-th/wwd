"""Verification of shipped files (PLAN.md §9.3 step 7): run them through the Python runtime.

Scores come from `runtime.Detector` itself, run on the model files exactly as they ship, over
whole long-form streams. The gate-on result reuses those scores: the model is stateless, so a
gate-open tick scores the same with the gate on or off; `replay` then feeds the runtime's own
`Gate` and `Trigger`, in the order `Detector.process` does.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from runtime.detector import Detector, Gate, Trigger
from runtime.spec import VAD_FRAME, DetectorConfig
from runtime.vad import SileroVad, make_session


def as_float(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32, copy=False)


def stream_scores(model: str | Path | bytes, audio: np.ndarray) -> np.ndarray:
    """Per-tick scores of one stream, gate off (runtime/SPEC.md §3 windows).

    Batch 1: with many sessions in parallel, larger fp32 batches spill the caches and become
    memory-bound (measured 7x slower in aggregate at batch 64 on 16 threads)."""
    det = Detector(model, None, gate=False, record_ticks=True, batch_size=1)
    det.process(as_float(audio))
    return np.array([t.score for t in det.ticks], np.float32)


def stream_embeddings(encoder: str | Path | bytes, audio: np.ndarray, hop: int = 1536,
                      window: int = 24000) -> np.ndarray:
    """Per-tick outputs of an encoder file on one stream, windowed as runtime/SPEC.md §3 does
    (one tick per `hop` samples, the window ending at (tick + 1) * hop, zeros before the start).
    Batch 1, for the same reason as `stream_scores`."""
    session, audio = make_session(encoder), as_float(audio)
    padded = np.concatenate([np.zeros(window, np.float32), audio])
    name = session.get_inputs()[0].name
    return np.concatenate([session.run(None, {name: padded[None, (t + 1) * hop : (t + 1) * hop + window]})[0]
                           for t in range(len(audio) // hop)])


def stream_vad(vad: str | Path | bytes, audio: np.ndarray) -> np.ndarray:
    """Per-frame VAD probabilities of one stream."""
    v, audio = SileroVad(vad), as_float(audio)
    return np.array([v(audio[i : i + VAD_FRAME])
                     for i in range(0, len(audio) - VAD_FRAME + 1, VAD_FRAME)], np.float32)


def replay(scores: np.ndarray, vad: np.ndarray | None, cfg: DetectorConfig,
           gate: bool) -> list[int]:
    """Ticks at which the runtime fires on a stream, given its gate-off scores."""
    g, trigger, events = Gate(cfg), Trigger(cfg), []
    per_tick = cfg.frames_per_tick
    for tick, score in enumerate(scores):
        if gate:
            for p in vad[tick * per_tick : (tick + 1) * per_tick]:
                g.push(p)
        if trigger.update(tick, float(score) if not gate or g.open else 0.0):
            events.append(tick)
    return events


def parallel(fn, items: Sequence, workers: int) -> list:
    """onnxruntime releases the GIL, so threads with one single-threaded session each scale
    like processes (the runtime's sessions are single-threaded, as on a device)."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(fn, items))


def false_accepts(model: str | Path | bytes, streams: Sequence[np.ndarray],
                  vad: Sequence[np.ndarray], cfg: DetectorConfig, workers: int) -> dict:
    """False accepts per hour over streams that never contain the wake word."""
    scores = parallel(lambda a: stream_scores(model, a), streams, workers)
    hours = sum(len(s) for s in scores) * cfg.hop_samples / cfg.sample_rate / 3600
    out = {"hours": hours, "scores": scores}
    for gate in (False, True):
        n = sum(len(replay(s, v, cfg, gate)) for s, v in zip(scores, vad, strict=True))
        out["gate" if gate else "no_gate"] = {"false_accepts": n, "per_hour": n / hours}
    return out
