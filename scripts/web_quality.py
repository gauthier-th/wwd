"""Browser training vs the server's, on the 24 held-out words of training.eval_shipped.

    uv run --extra training python -m scripts.web_quality prepare data/web_quality
    WWD_QUALITY=../data/web_quality pnpm --filter @wwd/web exec vitest run quality   # trains
    uv run --extra training python -m scripts.web_quality score data/web_quality

`prepare` writes each word's enrollment clips and test stream exactly as eval_shipped builds
them (same draw, same seed). The browser code (web/src/local/quality.test.ts) trains each word
with the real web assets and writes model.onnx. `score` runs those files through the Python
runtime as eval_shipped scores the server's: recall on other speakers' utterances, gate off
and on, and false accepts per hour on the held-out verification audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.verify import false_accepts
from runtime.spec import DetectorConfig
from server.basemodel import BaseModels
from server.settings import Settings
from training.eval_e2e import pick_targets, read_clips
from training.eval_shipped import detected, test_stream
from training.evaluate import HELDOUT
from training.store import ClipReader

REPO = Path(__file__).resolve().parents[1]


def prepare(out: Path, per_lang: int = 2, seed: int = 0) -> None:
    targets = pick_targets(pd.read_parquet(HELDOUT), 20, 10, 10, np.random.default_rng(seed))
    counts: dict[str, int] = {}
    chosen = []
    for t in targets:  # eval_shipped's selection: the first words per language
        if counts.get(t[0], 0) < per_lang:
            counts[t[0]] = counts.get(t[0], 0) + 1
            chosen.append(t)
    reader, rng = ClipReader(), np.random.default_rng(seed)
    words = []
    for n, (lang, word, enroll, test) in enumerate(chosen, 1):
        d = out / f"{n:02d}_{lang}"
        d.mkdir(parents=True, exist_ok=True)
        clips = read_clips(reader, enroll)
        np.concatenate(clips).astype(np.float32).tofile(d / "enroll.f32")
        audio, spans = test_stream(read_clips(reader, test), rng)
        audio.astype(np.float32).tofile(d / "test.f32")
        meta = {"word": word, "lang": lang, "enroll_lengths": [len(c) for c in clips], "spans": spans}
        (d / "word.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n")
        words.append(d.name)
    (out / "words.json").write_text(json.dumps(words) + "\n")
    print(f"{len(words)} words -> {out}")


def score(out: Path, base: Path) -> None:
    settings = Settings(models_dir=base.parent.parent, device="cpu")
    bm = BaseModels(settings).get(base.name)
    vad = bm.verify_vad(settings.cache_dir, 16)
    server = pd.read_csv(base / "eval_shipped.csv").set_index(["lang", "word"])
    rows = []
    for name in json.loads((out / "words.json").read_text()):
        d = out / name
        meta = json.loads((d / "word.json").read_text())
        report = json.loads((d / "report.json").read_text())
        model = d / "model.onnx"
        audio = np.fromfile(d / "test.f32", np.float32)
        thr = report["threshold"]
        fa = false_accepts(model.read_bytes(), bm.verify_streams, vad,
                           DetectorConfig().with_overrides(threshold=thr), 16)
        row = {"lang": meta["lang"], "word": meta["word"], "threshold": thr,
               "calib_fa": report["calibration"]["fp32"]["fa_per_hour"],
               "browser_verify_fa": report["verification"]["fp32"]["no_gate"]["per_hour"],
               "verify_fa": fa["no_gate"]["per_hour"], "verify_fa_gate": fa["gate"]["per_hour"],
               "cv_recall": (report["cross_validation"] or {}).get("recall_noisy"),
               "seconds": report["timings_s"]["total"]}
        for gate in (False, True):
            row["recall_gate" if gate else "recall"], _ = detected(model, audio, meta["spans"], gate)
        s = server.loc[(meta["lang"], meta["word"])]
        row |= {"server_recall": s["recall_fp32_no_gate"], "server_verify_fa": s["verify_fa_fp32_no_gate"],
                "server_threshold": s["threshold"]}
        rows.append(row)
        print(f"{name} {meta['word']!r}: recall {row['recall']:.2f} (server {row['server_recall']:.2f}), "
              f"FA/h {row['verify_fa']:.2f} (server {row['server_verify_fa']:.2f}), {row['seconds']:.0f}s",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "scores.csv", index=False)
    hours = fa["hours"]
    summary = {"words": len(df), "verify_hours": hours,
               "mean": df.drop(columns=["lang", "word"]).mean().round(4).to_dict(),
               "false_accepts": {"browser": round(float(df.verify_fa.sum() * hours)),
                                 "server": round(float(df.server_verify_fa.sum() * hours))}}
    (out / "scores.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["prepare", "score"])
    p.add_argument("out", type=Path)
    p.add_argument("--base", type=Path, default=REPO / "models" / "base" / "v1")
    args = p.parse_args(argv)
    if args.command == "prepare":
        prepare(args.out)
    else:
        score(args.out, args.base)


if __name__ == "__main__":
    main()
