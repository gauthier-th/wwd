"""Field-test logs (PLAN.md M4): what a detector did over hours of real audio.

One directory per run, written by `runtime.listen --log` on a Pi and by the web app's device
test on a phone (uploaded through the server), read by `runtime.fieldreport`:

    meta.json      model metadata and settings, host, start time
    events.jsonl   one line per trigger: wall time, tick, score, clip file
    stats.jsonl    one line per minute: CPU, latency per tick, gate-open share, overruns
    ticks.f32      every tick's (score, vad_max) as little-endian float32 pairs; the score is
                   0 while the speech gate is closed. Enough to replay the trigger at any
                   threshold afterwards: one 24 h run gives the whole false-accept curve.
    clips/         audio around each trigger (16 kHz WAV), to check what set it off
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .wav import write_wav


class FieldLog:
    def __init__(self, root: str | Path, meta: dict) -> None:
        self.root = Path(root)
        (self.root / "clips").mkdir(parents=True, exist_ok=True)
        self.meta = {"started_at": time.time(), **meta}
        self._write_meta()
        self.events = 0

    @classmethod
    def reopen(cls, root: str | Path) -> FieldLog:
        """Append to an existing log (the server receives a phone's log a minute at a time)."""
        log = cls.__new__(cls)
        log.root = Path(root)
        log.meta = json.loads((log.root / "meta.json").read_text())
        log.events = len(_jsonl(log.root / "events.jsonl"))
        return log

    def _write_meta(self) -> None:
        tmp = self.root / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False) + "\n")
        tmp.replace(self.root / "meta.json")

    def ticks(self, scores: np.ndarray, vad_max: np.ndarray) -> None:
        pairs = np.stack([np.asarray(scores, "<f4"), np.asarray(vad_max, "<f4")], axis=1)
        with open(self.root / "ticks.f32", "ab") as f:
            f.write(pairs.astype("<f4").tobytes())

    def event(self, record: dict, clip: np.ndarray | bytes | None = None,
              sample_rate: int = 16000) -> None:
        """`clip`: samples, or a WAV file's bytes."""
        self.events += 1
        if clip is not None:
            name = f"clips/event-{self.events:04d}.wav"
            if isinstance(clip, bytes):
                (self.root / name).write_bytes(clip)
            else:
                write_wav(self.root / name, clip, sample_rate)
            record = {**record, "clip": name}
        self._append("events.jsonl", record)

    def stats(self, record: dict) -> None:
        self._append("stats.jsonl", record)

    def close(self, **extra) -> None:
        self.meta.update({"ended_at": time.time(), **extra})
        self._write_meta()

    def _append(self, name: str, record: dict) -> None:
        with open(self.root / name, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


@dataclass
class FieldRun:
    root: Path
    meta: dict
    events: list[dict]
    stats: list[dict]
    scores: np.ndarray  # [T] float32
    vad_max: np.ndarray  # [T] float32 (NaN without a VAD)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(root: str | Path) -> FieldRun:
    root = Path(root)
    ticks = np.zeros((0, 2), "<f4")
    if (root / "ticks.f32").exists():
        raw = np.fromfile(root / "ticks.f32", dtype="<f4")
        ticks = raw[: len(raw) // 2 * 2].reshape(-1, 2)  # a crash can leave half a pair
    return FieldRun(root, json.loads((root / "meta.json").read_text()), _jsonl(root / "events.jsonl"),
                    _jsonl(root / "stats.jsonl"), ticks[:, 0].copy(), ticks[:, 1].copy())


class AudioRing:
    """The last `seconds` of a stream, addressable by absolute sample position."""

    def __init__(self, seconds: float, sample_rate: int = 16000) -> None:
        self.size = int(seconds * sample_rate)
        self.buf = np.zeros(self.size, np.float32)
        self.end = 0  # absolute position one past the newest sample

    def push(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, np.float32)[-self.size :]
        n = len(samples)
        self.buf = np.roll(self.buf, -n)
        self.buf[self.size - n :] = samples
        self.end += n

    def slice(self, start: int, stop: int) -> np.ndarray:
        """Samples [start, stop) that are still in the ring (zeros before the stream began)."""
        start, stop = max(start, self.end - self.size, 0), min(stop, self.end)
        if stop <= start:
            return np.zeros(0, np.float32)
        return self.buf[start - (self.end - self.size) : stop - (self.end - self.size)].copy()


def host_info() -> dict:
    """Which machine a measurement comes from."""
    import os
    import platform

    import onnxruntime

    model = ""
    for path, key in (("/proc/device-tree/model", None), ("/proc/cpuinfo", "model name"),
                      ("/proc/cpuinfo", "Model")):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if key is None:
            model = text.strip("\x00\n ")
        else:
            model = next((ln.split(":", 1)[1].strip() for ln in text.splitlines()
                          if ln.startswith(key)), "")
        if model:
            break
    return {"machine": platform.machine(), "system": f"{platform.system()} {platform.release()}",
            "cpu": model or platform.processor(), "cores": os.cpu_count(),
            "python": platform.python_version(), "onnxruntime": onnxruntime.__version__,
            "numpy": np.__version__}
