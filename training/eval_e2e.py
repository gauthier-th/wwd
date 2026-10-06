"""End-to-end evaluation of a base model: the user flow on held-out words (PLAN.md §7 step 6).

    uv run --extra training python -m training.eval_e2e models/base/v0.1 [--targets 20]

Per held-out word: enroll 10 clips (as few speakers as possible), run core.finetune exactly as
the server will, calibrate the threshold on the long-form cache, then measure
  - recall on clips of the word by *other* speakers, streamed over consecutive ticks;
  - false accepts per hour on the held-out verification streams (out of sample).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from core.augment import AugmentConfig, Augmenter, NoiseBank, RirBank
from core.calibrate import LongformCache, calibrate, count_triggers, stream_recall
from core.finetune import WINDOW, Bank, Enrollment, FinetuneConfig, finetune
from runtime.spec import DetectorConfig
from training.evaluate import HELDOUT, load_encoder
from training.export import embed_windows, stream_windows
from training.store import ClipReader

SHIFTS = 6  # consecutive ticks during which a 1 s clip sits fully inside the 1.5 s window
HOP = 1536


def load_augmenter(base: Path, device: str) -> Augmenter:
    z = np.load(base / "augment_assets.npz")
    return Augmenter(AugmentConfig(), NoiseBank(z["noise"], device), NoiseBank(z["babble"], device),
                     RirBank(z["rirs"].astype(np.float32), device))


def pick_targets(heldout: pd.DataFrame, per_lang: int, enroll: int, min_test: int,
                 rng: np.random.Generator) -> list[tuple[str, str, pd.DataFrame, pd.DataFrame]]:
    targets = []
    for lang, lf in heldout.groupby("lang"):
        words = lf.word.value_counts()
        words = words[words >= enroll + min_test].index.to_numpy()
        for word in rng.permutation(words):
            wf = lf[lf.word == word].sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
            by_speaker = wf.groupby("speaker").size().sort_values(ascending=False)
            chosen, n = [], 0
            for spk, count in by_speaker.items():
                chosen.append(spk)
                n += count
                if n >= enroll:
                    break
            enroll_df = wf[wf.speaker.isin(chosen)].head(enroll)
            test_df = wf[~wf.speaker.isin(chosen)].head(50)
            if len(test_df) >= min_test:
                targets.append((lang, word, enroll_df, test_df))
            if sum(t[0] == lang for t in targets) >= per_lang:
                break
    return targets


def read_clips(reader: ClipReader, frame: pd.DataFrame) -> list[np.ndarray]:
    return [reader.read(r.blob, r.offset, r.length) for r in frame.itertuples()]


def shifted_windows(clips: list[np.ndarray]) -> torch.Tensor:
    out = torch.zeros(len(clips) * SHIFTS, WINDOW)
    for i, c in enumerate(clips):
        c = torch.from_numpy(c[:16000])
        for k in range(SHIFTS):
            start = WINDOW - len(c) - k * HOP
            out[i * SHIFTS + k, max(0, start) : max(0, start) + len(c)] = c
    return out + torch.randn_like(out) * 10 ** (-70 / 20)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("base", type=Path)
    p.add_argument("--targets", type=int, default=20, help="held-out words per language")
    p.add_argument("--enroll", type=int, default=10)
    p.add_argument("--fa", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    device = "cuda"
    rng = np.random.default_rng(args.seed)

    encoder = load_encoder(args.base / "encoder.pt")
    bank = Bank.load(args.base / "neg_bank.npz")
    cache = LongformCache.load(args.base / "longform_cache.npz")
    augmenter = load_augmenter(args.base, device)
    verify = np.load(args.base / "longform_verify.npz")
    v_emb, v_starts = [], [0]
    for key in verify.files:
        e = embed_windows(encoder, stream_windows(verify[key]), None)
        v_emb.append(e)
        v_starts.append(v_starts[-1] + len(e))
    v_emb = np.concatenate(v_emb)
    v_hours = len(v_emb) * HOP / 16000 / 3600
    print(f"calibration {cache.hours:.1f} h, verification {v_hours:.1f} h, bank {len(bank.words):,}")

    reader = ClipReader()
    targets = pick_targets(pd.read_parquet(HELDOUT), args.targets, args.enroll, 10, rng)
    cfg, det = FinetuneConfig(), DetectorConfig()
    cache_emb = torch.from_numpy(cache.embeddings.astype(np.float32))
    rows = []
    for n, (lang, word, enroll_df, test_df) in enumerate(targets, 1):
        enrollment = Enrollment(word, lang, read_clips(reader, enroll_df))
        res = finetune(encoder, enrollment, bank, augmenter, cfg, device)
        head = res.head.cpu()
        with torch.no_grad():
            s_cache = torch.sigmoid(head(cache_emb)).numpy()
            s_verify = torch.sigmoid(head(torch.from_numpy(v_emb))).numpy()
            test_emb = embed_windows(encoder, shifted_windows(read_clips(reader, test_df)), None)
            s_test = torch.sigmoid(head(torch.from_numpy(test_emb))).numpy().reshape(-1, SHIFTS)
        row = {"lang": lang, "word": word, "enroll_speakers": int(enroll_df.speaker.nunique()),
               "test_clips": len(test_df)}
        for fa in args.fa:
            thr, fa_cal = calibrate(s_cache, cache, fa, det, ignore=cache.mask_word(word))
            row[f"thr@{fa}"] = thr
            row[f"calib_fa@{fa}"] = fa_cal
            row[f"recall@{fa}"] = stream_recall(s_test, thr, det)
            row[f"verify_fa@{fa}"] = count_triggers(s_verify, np.array(v_starts), thr, det) / v_hours
        rows.append(row)
        print(f"[{n}/{len(targets)}] {lang} {word!r}: " + ", ".join(
            f"recall@{fa}={row[f'recall@{fa}']:.2f} (verify {row[f'verify_fa@{fa}']:.2f}/h)"
            for fa in args.fa), flush=True)

    df = pd.DataFrame(rows)
    summary = {
        "targets": len(df),
        "per_lang": df.groupby("lang")[[c for c in df.columns if c.startswith(("recall", "verify"))]]
        .mean().round(4).to_dict(orient="index"),
        "mean": df[[c for c in df.columns if c.startswith(("recall", "verify"))]].mean().round(4).to_dict(),
    }
    df.to_csv(args.base / "eval_e2e.csv", index=False)
    (args.base / "eval_e2e.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["mean"], indent=2))


if __name__ == "__main__":
    main()
