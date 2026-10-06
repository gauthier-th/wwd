"""Threshold calibration against long-form audio (PLAN.md §9.3 step 5).

False accepts are counted exactly as runtime/SPEC.md §3 fires triggers (k-of-n smoothing and
refractory period per stream), with the gate off: the conservative number.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

from core.finetune import normalize_word
from runtime.spec import DetectorConfig


@dataclass
class LongformCache:
    """Encoder embeddings of every tick of the long-form set.

    Streams are contiguous runs of ticks; `stream_starts[i]:stream_starts[i+1]` is stream i.
    Word occurrences (MSWC word salad) are recorded so a wake word that happens to be spoken
    in the long-form audio isn't counted as a false accept.
    """

    embeddings: np.ndarray  # [T, D] float16
    stream_starts: np.ndarray  # [S + 1] int64
    occ_words: np.ndarray  # [O] str
    occ_ticks: np.ndarray  # [O, 2] int64, half-open tick ranges whose window holds the word
    hop_samples: int = 1536

    @property
    def hours(self) -> float:
        return len(self.embeddings) * self.hop_samples / 16000 / 3600

    @classmethod
    def load(cls, path: str | Path) -> LongformCache:
        z = np.load(path, allow_pickle=False)
        return cls(z["embeddings"], z["stream_starts"], z["occ_words"], z["occ_ticks"],
                   int(z["hop_samples"]))

    def save(self, path: str | Path) -> None:
        np.savez(path, embeddings=self.embeddings.astype(np.float16),
                 stream_starts=self.stream_starts, occ_words=self.occ_words.astype(str),
                 occ_ticks=self.occ_ticks, hop_samples=self.hop_samples)

    def mask_word(self, word: str) -> np.ndarray:
        """Ticks whose window contains `word` (excluded from false-accept counting)."""
        mask = np.zeros(len(self.embeddings), bool)
        target = normalize_word(word)
        for w, (a, b) in zip(self.occ_words, self.occ_ticks, strict=True):
            if normalize_word(w) == target:
                mask[a:b] = True
        return mask


def count_triggers(scores: np.ndarray, stream_starts: np.ndarray, threshold: float,
                   cfg: DetectorConfig) -> int:
    """Number of triggers the runtime would fire, each stream starting fresh."""
    hits = (scores.astype(np.float64) >= threshold).astype(np.int32)
    kernel = np.ones(cfg.smoothing_n, np.int32)
    total = 0
    for a, b in pairwise(stream_starts):
        window = np.convolve(hits[a:b], kernel)[: b - a]  # hits among the last n ticks
        next_allowed = 0
        for t in np.flatnonzero(window >= cfg.smoothing_k):
            if t >= next_allowed:
                total += 1
                next_allowed = t + cfg.refractory_ticks + 1
    return total


def calibrate(scores: np.ndarray, cache: LongformCache, target_fa_per_hour: float,
              cfg: DetectorConfig, ignore: np.ndarray | None = None) -> tuple[float, float]:
    """Lowest threshold whose false-accept rate is at most the target -> (threshold, FA/h)."""
    if ignore is not None:
        scores = np.where(ignore, 0.0, scores)
    def fa(thr: float) -> float:
        return count_triggers(scores, cache.stream_starts, thr, cfg) / cache.hours

    lo, hi = 0.0, 1.0
    if fa(hi - 1e-6) > target_fa_per_hour:
        return 1.0, fa(1.0)
    for _ in range(30):
        mid = (lo + hi) / 2
        if fa(mid) <= target_fa_per_hour:
            hi = mid
        else:
            lo = mid
    return hi, fa(hi)


def stream_recall(shifted_scores: np.ndarray, threshold: float, cfg: DetectorConfig) -> float:
    """Recall of clips scored at consecutive tick positions ([P, shifts]): a clip is detected
    when some run of `smoothing_n` consecutive ticks holds `smoothing_k` hits."""
    hits = (shifted_scores.astype(np.float64) >= threshold).astype(np.int32)
    n = cfg.smoothing_n
    runs = np.stack([hits[:, i : i + n].sum(1) for i in range(hits.shape[1] - n + 1)], axis=1)
    return float((runs.max(1) >= cfg.smoothing_k).mean())
