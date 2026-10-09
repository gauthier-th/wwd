"""Embed every training clip with a trained teacher, for distillation (PLAN.md §7 step 4).

    uv run --extra training python -m training.cache_teacher models/runs/<teacher>

Clips go in clean and centered (ClipDataset train=False), in train.parquet order. The
student later sees the same clips augmented, so it learns to map noisy audio onto the
teacher's clean embedding. Writes <run>/train_emb.f16 ([clips, D] float16, raw) and
train_emb.json (shape, progress, manifest fingerprint). Resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from training.data import ClipDataset
from training.paths import DATA

MANIFESTS = DATA / "manifests"
FLUSH_EVERY = 64  # batches between progress saves


def fingerprint(train: pd.DataFrame) -> str:
    rows = pd.util.hash_pandas_object(train[["blob", "offset", "class_id"]], index=False)
    return hashlib.md5(rows.to_numpy().tobytes()).hexdigest()


def load_cache(run: Path) -> np.ndarray:
    """The finished cache as a read-only memmap, checked against the current manifest."""
    meta = json.loads((run / "train_emb.json").read_text())
    if meta["done"] < meta["rows"]:
        raise RuntimeError(f"{run}: teacher cache incomplete ({meta['done']:,}/{meta['rows']:,})")
    if meta["fingerprint"] != fingerprint(pd.read_parquet(MANIFESTS / "train.parquet")):
        raise RuntimeError(f"{run}: train.parquet changed since the teacher cache was built")
    return np.memmap(run / "train_emb.f16", np.float16, "r", shape=(meta["rows"], meta["dim"]))


def main(argv: list[str] | None = None) -> None:
    from training.teacher import load_teacher

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("run", type=Path)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="stop after this many rows (testing)")
    args = p.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train = pd.read_parquet(MANIFESTS / "train.parquet")
    teacher = load_teacher(args.run / "best.pt")
    dim = teacher.config.embedding_dim
    meta_path, emb_path = args.run / "train_emb.json", args.run / "train_emb.f16"
    meta = {"rows": len(train), "dim": dim, "done": 0, "fingerprint": fingerprint(train),
            "checkpoint_step": torch.load(args.run / "best.pt", map_location="cpu",
                                          weights_only=False)["step"]}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        if {k: old[k] for k in ("rows", "dim", "fingerprint", "checkpoint_step")} == \
                {k: meta[k] for k in ("rows", "dim", "fingerprint", "checkpoint_step")}:
            meta["done"] = old["done"]
    mode = "r+" if meta["done"] and emb_path.exists() else "w+"
    emb = np.memmap(emb_path, np.float16, mode, shape=(len(train), dim))
    if meta["done"] >= len(train):
        print("teacher cache already complete")
        return
    print(f"embedding {len(train) - meta['done']:,} of {len(train):,} clips")

    stop = min(len(train), args.limit or len(train))
    dataset = Subset(ClipDataset(train, train=False), range(meta["done"], stop))
    loader = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers,
                        pin_memory=True, prefetch_factor=4)
    t0, start = time.time(), meta["done"]
    pos = start
    for i, (x, _) in enumerate(loader, 1):
        with torch.no_grad():
            out = teacher(x.cuda(non_blocking=True)).float().cpu().numpy()
        emb[pos : pos + len(out)] = out.astype(np.float16)
        pos += len(out)
        if i % FLUSH_EVERY == 0 or pos == stop:
            emb.flush()
            meta["done"] = pos
            meta_path.write_text(json.dumps(meta) + "\n")
            rate = (pos - start) / (time.time() - t0)
            print(json.dumps({"done": pos, "clips_per_s": round(rate),
                              "eta_h": round((len(train) - pos) / rate / 3600, 2)}), flush=True)
    print(f"teacher cache complete: {emb_path}" if pos == len(train) else f"stopped at {pos:,}")


if __name__ == "__main__":
    main()
