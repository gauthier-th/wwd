"""Streaming wake-word detector (runtime/SPEC.md §3, §5) on plain onnxruntime + numpy."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from runtime.onnxfile import read_metadata, split_model
from runtime.spec import VAD_FRAME, DetectorConfig
from runtime.vad import SileroVad, make_session

ModelSource = str | Path | bytes
Overrides = dict[str, float | int | None]


@dataclass(frozen=True)
class Event:
    tick: int
    sample: int  # stream position (samples) at the end of the triggering window
    time_s: float
    score: float
    word: int = 0  # which wake word fired: its index in the models given to a MultiDetector


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


@dataclass
class Word:
    """One wake word of a MultiDetector: its file's metadata and settings, and its ticks."""

    index: int
    metadata: dict[str, str]
    config: DetectorConfig
    ticks: list[Tick] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.metadata.get("wake_word") or f"word {self.index + 1}"


class _Pass:
    """A session the audio windows go through: one whole file, or an encoder that several files
    share, followed by each of their heads."""

    def __init__(self, words: list[int], session, heads: list | None = None) -> None:
        self.words, self.session, self.heads = words, session, heads
        self._input = session.get_inputs()[0].name

    def run(self, windows: np.ndarray) -> np.ndarray:
        """[N, window_samples] audio -> [N, len(words)] scores."""
        out = self.session.run(None, {self._input: windows})[0]
        if self.heads is None:
            return out.reshape(-1, 1)
        return np.stack([h.run(None, {h.get_inputs()[0].name: out})[0].reshape(-1)
                         for h in self.heads], axis=1)


def _passes(models: list[bytes], share_encoder: bool) -> list[_Pass]:
    """Files whose encoders are byte-identical (made from the same base model and precision)
    share one encoder pass; any other file runs whole, as it would alone."""
    splits = [split_model(m) if share_encoder and len(models) > 1 else None for m in models]
    groups: dict[bytes | int, list[int]] = {}
    for i, split in enumerate(splits):
        groups.setdefault(split.encoder if split else i, []).append(i)
    passes = []
    for words in groups.values():
        if len(words) == 1:
            passes.append(_Pass(words, make_session(models[words[0]])))
        else:
            passes.append(_Pass(words, make_session(splits[words[0]].encoder),
                                [make_session(splits[i].head) for i in words]))
    return passes


class MultiDetector:
    """Listen for several wake words at once. Feed 16 kHz float32 audio of any chunk size to
    `process`; get trigger events back, in tick order, then word order.

    Each word keeps its file's settings, its own gate and trigger, and gets exactly the ticks
    and events a Detector on its file would. The words share the audio, the VAD and the ticks,
    and files made from the same base model run their common encoder once per window: a second
    wake word then costs a head of a few hundred multiply-adds. `groups` lists the words of each
    pass (one group: everything shared).

    Offline callers can pass long chunks: the models then run batched, with results identical
    to small-chunk streaming.
    """

    def __init__(
        self,
        models: Sequence[ModelSource],
        vad: ModelSource | None = None,
        *,
        gate: bool = True,
        overrides: Overrides | Sequence[Overrides | None] | None = None,
        batch_size: int = 64,
        record_ticks: bool = False,
        share_encoder: bool = True,
    ) -> None:
        """`overrides`: one dict for every word, or one per word (None keeps a file's settings)."""
        if not models:
            raise ValueError("give at least one wake-word model")
        data = [m if isinstance(m, bytes) else Path(m).read_bytes() for m in models]
        if overrides is None or isinstance(overrides, dict):
            overrides = [overrides] * len(data)
        if len(overrides) != len(data):
            raise ValueError(f"{len(overrides)} overrides for {len(data)} models")
        self.words = [
            Word(i, meta, DetectorConfig.from_metadata(meta).with_overrides(**(o or {})))
            for i, (meta, o) in enumerate(zip(map(read_metadata, data), overrides, strict=True))
        ]
        stream = {(w.config.window_samples, w.config.hop_samples) for w in self.words}
        if len(stream) > 1:
            raise ValueError(f"the wake words need the same window_samples and hop_samples: {stream}")
        self._config = self.words[0].config  # the stream settings, common to every word
        self._gates = [Gate(w.config) for w in self.words]
        self._triggers = [Trigger(w.config) for w in self.words]
        self._vad = SileroVad(vad) if vad is not None else None
        self.gate = gate and self._vad is not None
        self.batch_size = batch_size
        self.record_ticks = record_ticks
        self._passes = _passes(data, share_encoder)
        self.groups = [p.words for p in self._passes]
        self.reset()

    def reset(self) -> None:
        self._history = np.zeros(self._config.window_samples, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames = 0
        self._ticks = 0
        for word, gate, trigger in zip(self.words, self._gates, self._triggers, strict=True):
            word.ticks = []
            gate.reset()
            trigger.reset()
        if self._vad is not None:
            self._vad.reset()

    def process(self, samples: np.ndarray) -> list[Event]:
        cfg = self._config
        audio = np.concatenate([self._pending, np.asarray(samples, dtype=np.float32).ravel()])
        n_frames = len(audio) // VAD_FRAME
        self._pending = audio[n_frames * VAD_FRAME :]
        if n_frames == 0:
            return []
        buf = np.concatenate([self._history, audio[: n_frames * VAD_FRAME]])

        ends, vad_max, gate_open = [], [], []  # per due tick; per word for the last two
        for i in range(n_frames):
            end = cfg.window_samples + (i + 1) * VAD_FRAME
            if self._vad is not None:
                prob = self._vad(buf[end - VAD_FRAME : end])
                for gate in self._gates:
                    gate.push(prob)
            self._frames += 1
            if self._frames % cfg.frames_per_tick:
                continue
            ends.append(end)
            vad_max.append([g.vad_max if self._vad is not None else math.nan for g in self._gates])
            gate_open.append([not self.gate or g.open for g in self._gates])

        open_ = np.array(gate_open, dtype=bool).reshape(len(ends), len(self.words))
        scores = self._score(ends, open_, buf)
        events: list[Event] = []
        for j in range(len(ends)):
            tick = self._ticks + j
            for word in self.words:
                score = float(scores[j, word.index])
                if self.record_ticks:
                    word.ticks.append(Tick(tick, vad_max[j][word.index], gate_open[j][word.index],
                                           score))
                if self._triggers[word.index].update(tick, score):
                    sample = (tick + 1) * cfg.hop_samples
                    events.append(Event(tick, sample, sample / cfg.sample_rate, score, word.index))
        self._ticks += len(ends)
        self._history = buf[-cfg.window_samples :]
        return events

    def _score(self, ends: list[int], gate_open: np.ndarray, buf: np.ndarray) -> np.ndarray:
        """[ticks, words] scores; 0 where a word's gate is closed. A pass runs on the ticks where
        any of its words' gates is open."""
        scores = np.zeros(gate_open.shape, dtype=np.float32)
        w = self._config.window_samples
        for p in self._passes:
            rows = np.flatnonzero(gate_open[:, p.words].any(axis=1))
            for i in range(0, len(rows), self.batch_size):
                batch = rows[i : i + self.batch_size]
                windows = np.stack([buf[ends[j] - w : ends[j]] for j in batch])
                scores[np.ix_(batch, p.words)] = p.run(windows)
        scores[~gate_open] = 0.0
        return scores


class Detector(MultiDetector):
    """One wake word. Feed 16 kHz float32 audio of any chunk size to `process`; get trigger
    events back.

    Offline callers can pass long chunks: the wake model then runs batched, with results
    identical to small-chunk streaming.
    """

    def __init__(
        self,
        model: ModelSource,
        vad: ModelSource | None = None,
        *,
        gate: bool = True,
        config: DetectorConfig | None = None,
        overrides: Overrides | None = None,
        batch_size: int = 64,
        record_ticks: bool = False,
    ) -> None:
        """`config` replaces the settings in the file's metadata; `overrides` then apply."""
        if config is not None:
            given = {k: v for k, v in (overrides or {}).items() if v is not None}
            overrides = {**asdict(config), **given}
        super().__init__([model], vad, gate=gate, overrides=overrides, batch_size=batch_size,
                         record_ticks=record_ticks)

    @property
    def config(self) -> DetectorConfig:
        return self.words[0].config

    @property
    def metadata(self) -> dict[str, str]:
        return self.words[0].metadata

    @property
    def ticks(self) -> list[Tick]:
        return self.words[0].ticks

    @ticks.setter
    def ticks(self, ticks: list[Tick]) -> None:
        self.words[0].ticks = ticks
