"""Train the keyword-embedding encoder directly with Sub-center ArcFace (M1 baseline, no teacher).

    uv run --extra training python -m training.train_student --name v0.1 [--steps 80000]

Writes models/runs/<name>/: last.pt (resumable), best.pt (best few-shot recall), log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from core.encoder import Encoder, EncoderConfig
from training import evaluate
from training.augment import build_augmenter
from training.data import ClipDataset, seed_worker
from training.losses import SubCenterArcFace
from training.paths import DATA, MODELS

MANIFESTS = DATA / "manifests"


def language_weights(langs: pd.Series, temperature: float) -> np.ndarray:
    """Per-clip sampling weights giving each language a share proportional to n_lang**t
    (t=1: as in the data, t=0: equal), uniform within a language."""
    counts = langs.value_counts()
    share = counts**temperature / (counts**temperature).sum()
    return (langs.map(share) / langs.map(counts)).to_numpy()


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    return peak * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--subcenters", type=int, default=3)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--margin-warmup", type=int, default=0,
                   help="ramp the ArcFace margin from 0 to --margin over this many steps")
    p.add_argument("--scale", type=float, default=30.0)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--noise-hours", type=float, default=3.0)
    p.add_argument("--lang-temperature", type=float, default=0.5,
                   help="language sampling share ~ clips**t (1 = natural, 0 = balanced)")
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--langs", default=None, help="restrict to these languages (debugging)")
    args = p.parse_args(argv)

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = MODELS / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(MANIFESTS / "train.parquet")
    classes = pd.read_parquet(MANIFESTS / "classes.parquet")
    heldout = pd.read_parquet(MANIFESTS / "heldout.parquet")
    if args.langs:
        keep = args.langs.split(",")
        train, heldout = train[train.lang.isin(keep)], heldout[heldout.lang.isin(keep)]
    eval_frame = evaluate.eval_subset(heldout, clips_per_word=20, words_per_lang=100)
    print(f"{len(train):,} training clips, {len(classes):,} classes, "
          f"{len(eval_frame):,} few-shot eval clips")

    config = EncoderConfig(width=args.width, embedding_dim=args.embedding_dim)
    encoder = Encoder(config).to(device)
    arcface = SubCenterArcFace(args.embedding_dim, len(classes), k=args.subcenters,
                               scale=args.scale, margin=args.margin).to(device)
    params = list(encoder.parameters()) + list(arcface.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    augment = build_augmenter(device, args.noise_hours)

    step, best = 0, -1.0
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        encoder.load_state_dict(ckpt["encoder"])
        arcface.load_state_dict(ckpt["arcface"])
        opt.load_state_dict(ckpt["optimizer"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed at step {step}")
    print(f"encoder {sum(p.numel() for p in encoder.parameters()) / 1e3:.0f}k params, "
          f"arcface head {arcface.weight.numel() / 1e6:.1f}M")

    weights = language_weights(train.lang, args.lang_temperature)
    share = pd.Series(weights).groupby(train.lang.to_numpy()).sum().round(3).to_dict()
    print(f"language sampling share: {share}")
    sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(train),
                                    replacement=True)
    loader = DataLoader(
        ClipDataset(train, train.class_id.to_numpy(), train=True),
        batch_size=args.batch, sampler=sampler, drop_last=True, num_workers=args.workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=4, worker_init_fn=seed_worker,
    )

    def save(path: Path) -> None:
        torch.save({
            "step": step, "best": best, "args": vars(args),
            "encoder_config": asdict(config),
            "encoder": encoder.state_dict(), "arcface": arcface.state_dict(),
            "optimizer": opt.state_dict(),
        }, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)

    def record(rec: dict) -> None:
        print(json.dumps(rec), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    encoder.train()
    t0, seen, run_loss, run_acc = time.time(), 0, 0.0, 0.0
    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            if args.margin_warmup:
                arcface.set_margin(args.margin * min(1.0, step / args.margin_warmup))
            x = augment(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                emb = encoder(x)
            loss, cos = arcface(emb.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            step += 1
            seen += len(y)
            run_loss += loss.item()
            run_acc += (cos.argmax(1) == y).float().mean().item()

            if step % args.log_every == 0:
                rate = seen / (time.time() - t0)
                rec = {"step": step, "loss": run_loss / args.log_every,
                       "train_acc": run_acc / args.log_every, "lr": opt.param_groups[0]["lr"],
                       "margin": round(arcface.margin, 4), "clips_per_s": round(rate)}
                record(rec)
                t0, seen, run_loss, run_acc = time.time(), 0, 0.0, 0.0

            if step % args.eval_every == 0 or step == args.steps:
                emb_eval = evaluate.embed(encoder, eval_frame, workers=args.workers)
                res = evaluate.fewshot(emb_eval, eval_frame, episodes=200)
                score = res["mean"]["recall@1%FAR_10shot"]
                rec = {"step": step, "eval": res["mean"],
                       "per_lang_10shot": {k: round(v["recall@1%FAR_10shot"], 4)
                                           for k, v in res.items() if k != "mean"}}
                record(rec)
                if score > best:
                    best = score
                    save(out / "best.pt")
                save(out / "last.pt")
                encoder.train()
                t0, seen = time.time(), 0
    print(f"done: best 10-shot recall@1%FAR {best:.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4)
    main()
