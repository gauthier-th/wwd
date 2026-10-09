"""Train the XLS-R teacher's head with Sub-center ArcFace (M5), or probe XLS-R's layers.

    uv run --extra training python -m training.train_teacher --probe
    uv run --extra training python -m training.train_teacher --name teacher [--steps 28000]

Same clips, sampling, augmentation and loss as the student (training.train_student); XLS-R
stays frozen and only the head trains. --probe scores each XLS-R layer alone (mean-pooled,
centered) with the few-shot protocol, which shows where word identity lives.

Writes models/runs/<name>/: last.pt (resumable), best.pt, log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from training import evaluate
from training.augment import build_augmenter
from training.data import ClipDataset
from training.losses import SubCenterArcFace
from training.paths import MODELS
from training.teacher import Teacher, TeacherConfig, XlsrFeatures
from training.train_student import (
    fewshot_record,
    load_manifests,
    lr_at,
    run_logger,
    save_atomic,
    train_loader,
)


class PooledStates(nn.Module):
    """Every XLS-R state, mean-pooled over time: [N, layers + 1, 1024]."""

    def __init__(self, features: XlsrFeatures) -> None:
        super().__init__()
        self.features = features

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.features(audio).float().mean(2)


def probe(eval_frame: pd.DataFrame, workers: int) -> dict:
    pooled = evaluate.embed(PooledStates(XlsrFeatures()).cuda(), eval_frame, workers=workers)
    rows = {}
    for layer in range(pooled.shape[1]):
        emb = pooled[:, layer] - pooled[:, layer].mean(0)
        res = evaluate.fewshot(emb, eval_frame, episodes=200)["mean"]
        rows[layer] = {"1shot": round(res["recall@1%FAR_1shot"], 4),
                       "10shot": round(res["recall@1%FAR_10shot"], 4)}
        print(f"layer {layer:2d}: {rows[layer]}", flush=True)
    return rows


@torch.no_grad()
def init_centers(teacher: Teacher, arcface: SubCenterArcFace, train: pd.DataFrame,
                 per_class: int, workers: int) -> None:
    """Start every class center at the mean embedding of a few of its clips.

    With random centers, early training bends the head toward random targets: in the first
    attempt few-shot recall fell from 31% (untrained head on XLS-R) to 5.5% after 4k steps,
    with training accuracy still ~0. Centers that agree with the features from step 0 skip
    that cold start; v0.1's student, trained from scratch, could afford it over 80k steps.
    """
    frame = train.groupby("class_id", group_keys=False).head(per_class)
    loader = DataLoader(ClipDataset(frame, frame.class_id.to_numpy(), train=False),
                        batch_size=512, num_workers=workers, pin_memory=True)
    teacher.eval()
    sums = torch.zeros(arcface.n_classes, arcface.weight.shape[1], device="cuda")
    for x, y in loader:
        sums.index_add_(0, y.cuda(), teacher(x.cuda(non_blocking=True)).float())
    centers = torch.nn.functional.normalize(sums, dim=1).repeat_interleave(arcface.k, 0)
    arcface.weight.copy_(centers + 0.01 * torch.randn_like(centers))  # split the sub-centers
    print(f"class centers initialized from {len(frame):,} clips", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--name", default="teacher")
    p.add_argument("--probe", action="store_true", help="score each XLS-R layer and exit")
    p.add_argument("--steps", type=int, default=28_000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--layers", type=int, default=24, help="XLS-R layers to run")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--context-layers", type=int, default=2)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--subcenters", type=int, default=3)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--scale", type=float, default=30.0)
    p.add_argument("--init-clips", type=int, default=2,
                   help="clips per class whose mean embedding starts its center (0: random)")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--noise-hours", type=float, default=3.0)
    p.add_argument("--lang-temperature", type=float, default=0.5)
    p.add_argument("--eval-every", type=int, default=4000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--langs", default=None, help="restrict to these languages (debugging)")
    args = p.parse_args(argv)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    out = MODELS / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    train, classes, eval_frame = load_manifests(args.langs)

    if args.probe:
        (out / "probe.json").write_text(json.dumps(probe(eval_frame, args.workers), indent=2) + "\n")
        return

    config = TeacherConfig(layers=args.layers, dim=args.dim, context_layers=args.context_layers,
                           embedding_dim=args.embedding_dim)
    teacher = Teacher(config).to(device)
    arcface = SubCenterArcFace(args.embedding_dim, len(classes), k=args.subcenters,
                               scale=args.scale, margin=args.margin).to(device)
    params = list(teacher.head.parameters()) + list(arcface.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, fused=True)
    augment = build_augmenter(device, args.noise_hours)

    step, best = 0, -1.0
    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        teacher.head.load_state_dict(ckpt["head"])
        arcface.load_state_dict(ckpt["arcface"])
        opt.load_state_dict(ckpt["optimizer"])
        step, best = ckpt["step"], ckpt["best"]
        print(f"resumed at step {step}")
    elif args.init_clips:
        init_centers(teacher, arcface, train, args.init_clips, args.workers)
    print(f"teacher head {sum(p.numel() for p in teacher.head.parameters()) / 1e6:.1f}M params, "
          f"XLS-R layers {config.layers}, arcface head {arcface.weight.numel() / 1e6:.1f}M")

    loader = train_loader(train, train.class_id.to_numpy(), args.batch, args.workers,
                          args.lang_temperature)

    def save(path) -> None:
        save_atomic({**teacher.checkpoint(), "step": step, "best": best, "args": vars(args),
                     "config": asdict(config), "arcface": arcface.state_dict(),
                     "optimizer": opt.state_dict()}, path)

    record = run_logger(out)
    teacher.train()
    t0, seen, run_loss, run_acc = time.time(), 0, 0.0, 0.0
    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
            x = augment(x.to(device, non_blocking=True))
            y = y.to(device, non_blocking=True)
            emb = teacher(x)
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
                weights = teacher.head.layer_weights().detach().cpu().numpy()
                record({"step": step, "loss": run_loss / args.log_every,
                        "train_acc": run_acc / args.log_every, "lr": opt.param_groups[0]["lr"],
                        "clips_per_s": round(rate),
                        "top_layers": [int(i) for i in np.argsort(-weights)[:4]]})
                t0, seen, run_loss, run_acc = time.time(), 0, 0.0, 0.0

            if step % args.eval_every == 0 or step == args.steps:
                emb_eval = evaluate.embed(teacher, eval_frame, workers=args.workers)
                res = evaluate.fewshot(emb_eval, eval_frame, episodes=200)
                score = res["mean"]["recall@1%FAR_10shot"]
                rec = fewshot_record(step, res)
                rec["layer_weights"] = [round(float(w), 4) for w in teacher.head.layer_weights()]
                record(rec)
                if score > best:
                    best = score
                    save(out / "best.pt")
                save(out / "last.pt")
                teacher.train()
                t0, seen = time.time(), 0
    print(f"done: best 10-shot recall@1%FAR {best:.4f}")


if __name__ == "__main__":
    main()
