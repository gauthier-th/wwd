"""Few-shot evaluation on held-out words (never seen in training).

    uv run --extra training python -m training.evaluate models/runs/<run>/best.pt

Protocol, per language: pick a held-out target word, average K enrollment clips into a
prototype, score other clips of that word (positives) and clips of other held-out words of
the same language (negatives) by cosine similarity. Report recall at the threshold that
gives 1% false alarms on the negatives, and the equal error rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from core.encoder import Encoder
from training.data import ClipDataset
from training.paths import DATA

HELDOUT = DATA / "manifests" / "heldout.parquet"


def eval_subset(heldout: pd.DataFrame, clips_per_word: int, words_per_lang: int,
                seed: int = 0) -> pd.DataFrame:
    """A fixed, reproducible slice of the held-out set."""
    enough = heldout.groupby(["lang", "word"]).clip.transform("size") >= clips_per_word
    df = heldout[enough]
    words = (df[["lang", "word"]].drop_duplicates().sample(frac=1.0, random_state=seed)
             .groupby("lang", group_keys=False).head(words_per_lang))
    df = df.merge(words, on=["lang", "word"])
    return (df.sample(frac=1.0, random_state=seed).groupby(["lang", "word"], group_keys=False)
            .head(clips_per_word).reset_index(drop=True))


@torch.no_grad()
def embed(encoder: Encoder, frame: pd.DataFrame, *, batch: int = 512, workers: int = 12,
          device: str = "cuda") -> np.ndarray:
    loader = DataLoader(ClipDataset(frame, train=False), batch_size=batch, num_workers=workers,
                        pin_memory=True, persistent_workers=False)
    was_training = encoder.training
    encoder.eval()
    out = []
    for x, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            out.append(encoder(x.to(device, non_blocking=True)).float().cpu())
    encoder.train(was_training)
    return torch.cat(out).numpy()


def fewshot(emb: np.ndarray, frame: pd.DataFrame, *, shots: tuple[int, ...] = (1, 5, 10),
            episodes: int = 200, far: float = 0.01, max_neg: int = 2000,
            seed: int = 0) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    results: dict[str, dict[str, float]] = {}
    for lang, lf in frame.groupby("lang"):
        words = lf.word.to_numpy()
        idx = lf.index.to_numpy()
        vocab = np.unique(words)
        if len(vocab) < 2:
            continue
        res = {}
        for k in shots:
            recalls, eers = [], []
            for _ in range(episodes):
                target = vocab[rng.integers(len(vocab))]
                pos = rng.permutation(idx[words == target])
                if len(pos) < k + 5:
                    continue
                neg = idx[words != target]
                neg = rng.choice(neg, min(max_neg, len(neg)), replace=False)
                proto = emb[pos[:k]].mean(0)
                proto /= np.linalg.norm(proto)
                s_pos, s_neg = emb[pos[k:]] @ proto, emb[neg] @ proto
                thr = np.quantile(s_neg, 1 - far)
                recalls.append(float((s_pos > thr).mean()))
                eers.append(_eer(s_pos, s_neg))
            res[f"recall@{far:.0%}FAR_{k}shot"] = float(np.mean(recalls))
            res[f"eer_{k}shot"] = float(np.mean(eers))
        results[lang] = res
    keys = next(iter(results.values())).keys()
    results["mean"] = {key: float(np.mean([r[key] for r in results.values()])) for key in keys}
    return results


def _eer(pos: np.ndarray, neg: np.ndarray) -> float:
    thr = np.sort(np.concatenate([pos, neg]))
    frr = np.searchsorted(np.sort(pos), thr, side="left") / len(pos)
    far = 1 - np.searchsorted(np.sort(neg), thr, side="left") / len(neg)
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def load_encoder(checkpoint: str | Path, device: str = "cuda") -> Encoder:
    from core.encoder import EncoderConfig

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder = Encoder(EncoderConfig(**ckpt["encoder_config"]))
    encoder.load_state_dict(ckpt["encoder"])
    return encoder.to(device).eval()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--clips-per-word", type=int, default=40)
    p.add_argument("--words-per-lang", type=int, default=300)
    p.add_argument("--episodes", type=int, default=500)
    args = p.parse_args(argv)
    frame = eval_subset(pd.read_parquet(HELDOUT), args.clips_per_word, args.words_per_lang)
    emb = embed(load_encoder(args.checkpoint), frame)
    res = fewshot(emb, frame, episodes=args.episodes)
    print(json.dumps(res, indent=2))
    out = Path(args.checkpoint).with_suffix(".fewshot.json")
    out.write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
