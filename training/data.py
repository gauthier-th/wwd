"""PyTorch datasets over the clip manifests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, get_worker_info

from training.store import ClipReader

WINDOW = 24000


class ClipDataset(Dataset):
    """MSWC clips decoded to 16 kHz and placed in a 1.5 s window.

    train=True: random speed perturbation and random position in the window.
    train=False: no perturbation, centered, plus a fixed -70 dBFS noise floor seeded by index,
    so evaluation is deterministic and never sees digital silence.
    """

    def __init__(self, frame: pd.DataFrame, labels: np.ndarray | None = None, *, train: bool,
                 speed: tuple[float, float] = (0.9, 1.1)) -> None:
        # Plain numpy arrays, not a DataFrame: workers would otherwise copy-on-touch refcounts.
        codes, blobs = pd.factorize(frame.blob)
        self.blobs = list(blobs)
        self.blob_id = codes.astype(np.int32)
        self.offset = frame.offset.to_numpy(np.int64)
        self.length = frame.length.to_numpy(np.int32)
        self.labels = np.full(len(frame), -1, np.int64) if labels is None else labels.astype(np.int64)
        self.train, self.speed = train, speed
        self.reader = ClipReader()

    def __len__(self) -> int:
        return len(self.offset)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        speed = rng.uniform(*self.speed) if self.train else 1.0
        clip = self.reader.read(self.blobs[self.blob_id[i]], self.offset[i], self.length[i], speed)
        clip = clip[:WINDOW]
        out = np.zeros(WINDOW, np.float32)
        start = rng.integers(0, WINDOW - len(clip) + 1) if self.train else (WINDOW - len(clip)) // 2
        out[start : start + len(clip)] = clip
        if not self.train:
            out += rng.normal(0, 10 ** (-70 / 20), WINDOW).astype(np.float32)
        return torch.from_numpy(out), int(self.labels[i])


def seed_worker(_: int) -> None:
    info = get_worker_info()
    np.random.seed(info.seed % (1 << 32))
