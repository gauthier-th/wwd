"""Build the training / evaluation manifests from the downloaded MSWC stores.

    uv run --extra training python -m training.prepare [--langs cy,eu] [--cap 300]

Splits are our own (MSWC's train/dev/test share almost all speakers), all by stable hash:
  - eval speakers: 10% of each language's speakers; none of their clips are trained on;
  - held-out words: 10% of each language's eligible words; never trained on, used for
    few-shot and end-to-end evaluation;
  - training classes: the other eligible words, clips from the other speakers, capped.

Writes data/manifests/{classes,train,val,heldout}.parquet.
"""

from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from training.paths import DATA, LANGUAGES
from training.store import load_index

MANIFESTS = DATA / "manifests"
MIN_CLIPS = 20
MIN_SPEAKERS = 5


def bucket(key: str, buckets: int = 10) -> int:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % buckets


def prepare_language(lang: str, cap: int, rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    df = load_index(lang)
    if df.empty:
        return {}
    eval_speaker = df.speaker.map(lambda s: bucket(f"spk:{s}") == 0)
    stats = (df[~eval_speaker].groupby("word")
             .agg(n=("clip", "size"), speakers=("speaker", "nunique")))
    eligible = stats[(stats.n >= MIN_CLIPS) & (stats.speakers >= MIN_SPEAKERS)].index
    heldout_words = {w for w in eligible if bucket(f"word:{lang}:{w}") == 0}
    train_words = sorted(set(eligible) - heldout_words)

    pool = df[~eval_speaker & df.word.isin(train_words)]
    # Cap clips per word, keeping as many distinct speakers as possible: shuffle, then take
    # each speaker's clips round-robin.
    pool = pool.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    pool = pool.assign(turn=pool.groupby(["word", "speaker"]).cumcount())
    train = (pool.sort_values(["word", "turn"]).groupby("word", group_keys=False).head(cap)
             .drop(columns="turn"))

    val = (df[eval_speaker & df.word.isin(train_words)]
           .sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
           .groupby("word", group_keys=False).head(3))
    heldout = df[df.word.isin(heldout_words)].assign(eval_speaker=eval_speaker)
    return {"train": train, "val": val, "heldout": heldout}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--langs", default=",".join(LANGUAGES))
    p.add_argument("--cap", type=int, default=300, help="max training clips per word")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    parts: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "heldout": []}
    summary = {}
    for lang in [lang for lang in LANGUAGES if lang in args.langs.split(",")]:
        out = prepare_language(lang, args.cap, rng)
        if not out:
            print(f"{lang}: no downloaded shards, skipped")
            continue
        for k, v in out.items():
            parts[k].append(v)
        summary[lang] = {
            "train_words": int(out["train"].word.nunique()),
            "train_clips": len(out["train"]),
            "val_clips": len(out["val"]),
            "heldout_words": int(out["heldout"].word.nunique()),
            "heldout_clips": len(out["heldout"]),
        }
        print(f"{lang}: {summary[lang]}")

    MANIFESTS.mkdir(parents=True, exist_ok=True)
    train = pd.concat(parts["train"], ignore_index=True)
    classes = (train[["lang", "word"]].drop_duplicates().sort_values(["lang", "word"])
               .reset_index(drop=True))
    classes["class_id"] = np.arange(len(classes), dtype=np.int32)
    keys = ["lang", "word"]
    train = train.merge(classes, on=keys)
    val = pd.concat(parts["val"], ignore_index=True).merge(classes, on=keys)
    classes.to_parquet(MANIFESTS / "classes.parquet")
    train.to_parquet(MANIFESTS / "train.parquet")
    val.to_parquet(MANIFESTS / "val.parquet")
    pd.concat(parts["heldout"], ignore_index=True).to_parquet(MANIFESTS / "heldout.parquet")
    summary["total"] = {"classes": len(classes), "train_clips": len(train), "val_clips": len(val)}
    (MANIFESTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"total: {summary['total']}")


if __name__ == "__main__":
    main()
