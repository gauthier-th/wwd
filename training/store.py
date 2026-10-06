"""Random access to MSWC clips in the packed stores written by training.download."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
import soxr

from training.opus import AVAILABLE as OPUS_AVAILABLE
from training.opus import OpusDecoder
from training.paths import MSWC, SPLITS

TARGET_RATE = 16000


def load_index(lang: str) -> pd.DataFrame:
    """Every packed clip of a language joined with its split CSV row.

    Columns: lang, word, clip, speaker, gender, mswc_split, blob, offset, length.
    Only shards that finished downloading are included.
    """
    parts = []
    for split in SPLITS:
        for index in sorted((MSWC / lang / split).glob("*.parquet")):
            t = pq.read_table(index).to_pandas()
            t["blob"] = str(index.with_suffix(".bin"))
            t["mswc_split"] = split
            parts.append(t)
    if not parts:
        return pd.DataFrame()
    packed = pd.concat(parts, ignore_index=True)
    meta = []
    for split in SPLITS:
        csv = pd.read_csv(MSWC / lang / "splits" / f"{split}.csv",
                          usecols=["LINK", "WORD", "VALID", "SPEAKER", "GENDER"])
        csv["clip"] = csv.LINK.str.rsplit("/", n=1).str[-1].str.removesuffix(".opus")
        meta.append(csv.drop(columns="LINK"))
    meta = pd.concat(meta, ignore_index=True).rename(
        columns={"WORD": "word", "VALID": "valid", "SPEAKER": "speaker", "GENDER": "gender"})
    df = packed.merge(meta, on=["word", "clip"], how="inner")
    df = df[df.valid].drop(columns="valid")
    df.insert(0, "lang", lang)
    return df.reset_index(drop=True)


class ClipReader:
    """Decodes clips to 16 kHz float32. Opens blob files lazily, once per process, so it is
    safe to use from DataLoader workers.

    Uses libopus directly at 16 kHz when available (training.opus, ~5x faster), else
    soundfile + soxr. Every training and evaluation path reads through this class, so a given
    machine always decodes the same way.
    """

    def __init__(self) -> None:
        self._fds: dict[str, int] = {}
        self._pid = os.getpid()
        self._opus = OpusDecoder() if OPUS_AVAILABLE else None

    def _fd(self, blob: str) -> int:
        if os.getpid() != self._pid:  # forked into a worker: don't share descriptors
            self._fds, self._pid = {}, os.getpid()
        fd = self._fds.get(blob)
        if fd is None:
            fd = self._fds[blob] = os.open(blob, os.O_RDONLY)
        return fd

    def raw(self, blob: str, offset: int, length: int) -> bytes:
        return os.pread(self._fd(blob), length, offset)

    def read(self, blob: str, offset: int, length: int, speed: float = 1.0) -> np.ndarray:
        """Decode one clip. `speed` > 1 plays faster (fewer samples), as speed perturbation."""
        raw = self.raw(blob, offset, length)
        if self._opus is not None:
            audio, rate = self._opus(raw), TARGET_RATE
        else:
            audio, rate = sf.read(io.BytesIO(raw), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        if rate == TARGET_RATE and speed == 1.0:
            return audio
        return soxr.resample(audio, rate, TARGET_RATE / speed).astype(np.float32, copy=False)

    def __getstate__(self) -> dict:
        return {}  # descriptors don't survive pickling into workers

    def __setstate__(self, state: dict) -> None:
        self.__init__()


def read_wav_16k(path: str | Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    return audio if rate == TARGET_RATE else soxr.resample(audio, rate, TARGET_RATE).astype(np.float32)
