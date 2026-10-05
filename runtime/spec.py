"""Detector contract shared by every runtime. Mirrors runtime/SPEC.md and runtime/js/src/spec.ts."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

FORMAT_VERSION = "1"
SAMPLE_RATE = 16000
VAD_FRAME = 512  # new samples per VAD call (32 ms)
VAD_CONTEXT = 64  # samples of the previous call prepended to each VAD call


@dataclass(frozen=True)
class DetectorConfig:
    sample_rate: int = SAMPLE_RATE
    window_samples: int = 24000
    hop_samples: int = 1536
    threshold: float = 0.5
    smoothing_k: int = 2
    smoothing_n: int = 3
    refractory_ms: int = 1500
    vad_threshold: float = 0.5
    vad_hangover_ms: int = 1500

    def __post_init__(self) -> None:
        if self.sample_rate != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE}")
        if self.hop_samples % VAD_FRAME:
            raise ValueError(f"hop_samples must be a multiple of {VAD_FRAME}")
        if not 1 <= self.smoothing_k <= self.smoothing_n:
            raise ValueError("need 1 <= smoothing_k <= smoothing_n")

    @property
    def frames_per_tick(self) -> int:
        return self.hop_samples // VAD_FRAME

    @property
    def gate_frames(self) -> int:
        return math.ceil(self.vad_hangover_ms * self.sample_rate / 1000 / VAD_FRAME)

    @property
    def refractory_ticks(self) -> int:
        return math.ceil(self.refractory_ms * self.sample_rate / 1000 / self.hop_samples)

    @classmethod
    def from_metadata(cls, meta: dict[str, str]) -> DetectorConfig:
        """Build from ONNX metadata_props / model.json; unknown keys are ignored."""
        kwargs = {}
        for f in fields(cls):
            if f.name in meta:
                kwargs[f.name] = float(meta[f.name]) if f.type == "float" else int(meta[f.name])
        return cls(**kwargs)

    def to_metadata(self) -> dict[str, str]:
        return {f.name: str(getattr(self, f.name)) for f in fields(self)}

    def with_overrides(self, **overrides: float | None) -> DetectorConfig:
        return replace(self, **{k: v for k, v in overrides.items() if v is not None})
