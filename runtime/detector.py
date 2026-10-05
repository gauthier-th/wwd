"""Streaming wake-word detector (runtime/SPEC.md §3) on plain onnxruntime + numpy."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from runtime.spec import VAD_FRAME, DetectorConfig
from runtime.vad import SileroVad, make_session


@dataclass(frozen=True)
class Event:
    tick: int
    sample: int  # stream position (samples) at the end of the triggering window
    time_s: float
    score: float


@dataclass(frozen=True)
class Tick:
    index: int
    vad_max: float  # max VAD probability over the gate span; NaN when no VAD is used
    gate_open: bool
    score: float  # 0.0 when the gate was closed


class Gate:
    """Speech gate: open while any VAD frame in the last `gate_frames` reached the threshold."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._recent: deque[float] = deque(maxlen=self.config.gate_frames)

    def push(self, vad_prob: float) -> None:
        self._recent.append(float(vad_prob))

    @property
    def vad_max(self) -> float:
        return max(self._recent, default=0.0)

    @property
    def open(self) -> bool:
        return self.vad_max >= self.config.vad_threshold


class Trigger:
    """k-of-n smoothing + refractory period, fed one tick score at a time."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._recent: deque[bool] = deque(maxlen=self.config.smoothing_n)
        self._next_allowed = 0

    def update(self, tick: int, score: float) -> bool:
        self._recent.append(float(score) >= self.config.threshold)
        if tick < self._next_allowed or sum(self._recent) < self.config.smoothing_k:
            return False
        self._next_allowed = tick + self.config.refractory_ticks + 1
        return True


def read_metadata(session) -> dict[str, str]:
    return dict(session.get_modelmeta().custom_metadata_map)


class Detector:
    """Feed 16 kHz float32 audio of any chunk size to `process`; get trigger events back.

    Offline callers can pass long chunks: the wake model then runs batched, with results
    identical to small-chunk streaming.
    """

    def __init__(
        self,
        model: str | Path | bytes,
        vad: str | Path | bytes | None = None,
        *,
        gate: bool = True,
        config: DetectorConfig | None = None,
        overrides: dict[str, float | int | None] | None = None,
        batch_size: int = 64,
        record_ticks: bool = False,
    ) -> None:
        self._session = make_session(model)
        self.metadata = read_metadata(self._session)
        base = config or DetectorConfig.from_metadata(self.metadata)
        self.config = base.with_overrides(**(overrides or {}))
        self._vad = SileroVad(vad) if vad is not None else None
        self.gate = gate and self._vad is not None
        self.batch_size = batch_size
        self.record_ticks = record_ticks
        self._input = self._session.get_inputs()[0].name
        self._gate = Gate(self.config)
        self._trigger = Trigger(self.config)
        self.reset()

    def reset(self) -> None:
        cfg = self.config
        self._history = np.zeros(cfg.window_samples, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames = 0
        self._ticks = 0
        self._gate.reset()
        self._trigger.reset()
        if self._vad is not None:
            self._vad.reset()
        self.ticks: list[Tick] = []

    def process(self, samples: np.ndarray) -> list[Event]:
        cfg = self.config
        audio = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32).ravel()])
        n_frames = len(audio) // VAD_FRAME
        self._pending = audio[n_frames * VAD_FRAME :]
        if n_frames == 0:
            return []
        buf = np.concatenate([self._history, audio[: n_frames * VAD_FRAME]])

        due: list[tuple[int, float, bool, int]] = []  # (tick, vad_max, gate_open, window end)
        for i in range(n_frames):
            end = cfg.window_samples + (i + 1) * VAD_FRAME
            if self._vad is not None:
                self._gate.push(self._vad(buf[end - VAD_FRAME : end]))
            self._frames += 1
            if self._frames % cfg.frames_per_tick:
                continue
            vad_max = self._gate.vad_max if self._vad is not None else float("nan")
            gate_open = not self.gate or self._gate.open
            due.append((self._ticks, vad_max, gate_open, end))
            self._ticks += 1

        scores = self._score([end for _, _, open_, end in due if open_], buf)
        events: list[Event] = []
        it = iter(scores)
        for tick, vad_max, gate_open, end in due:
            score = float(next(it)) if gate_open else 0.0
            if self.record_ticks:
                self.ticks.append(Tick(tick, vad_max, gate_open, score))
            if self._trigger.update(tick, score):
                sample = (tick + 1) * cfg.hop_samples
                events.append(Event(tick, sample, sample / cfg.sample_rate, score))

        self._history = buf[-cfg.window_samples :]
        return events

    def _score(self, ends: list[int], buf: np.ndarray) -> np.ndarray:
        if not ends:
            return np.zeros(0, dtype=np.float32)
        w = self.config.window_samples
        out = []
        for i in range(0, len(ends), self.batch_size):
            batch = np.stack([buf[e - w : e] for e in ends[i : i + self.batch_size]])
            out.append(self._session.run(None, {self._input: batch})[0].reshape(-1))
        return np.concatenate(out)
