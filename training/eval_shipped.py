"""The server's quick mode on held-out words, scored on the files it ships (PLAN.md §9.3 step 7).

    uv run --extra training python -m training.eval_shipped models/base/v0.1 [--per-lang 2]

Same targets as training.eval_e2e (same seed, its first words per language), so each word can
be compared with eval_e2e's PyTorch numbers. For each: `server.quick.run` (fine-tune, calibrate
to 0.5 false accepts/h, compose, verify), then the runtime Detector on a stream of other
speakers saying the word, for model.onnx and model.int8.onnx, gate on and off.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from runtime import Detector
from server import quick
from server.basemodel import VAD_FILE, BaseModels
from server.settings import Settings
from training.eval_e2e import pick_targets, read_clips
from training.evaluate import HELDOUT
from training.store import ClipReader

SR = 16000


def test_stream(clips: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, list]:
    gap = np.zeros(2 * SR, np.float32)
    parts, spans, pos = [gap[:SR]], [], SR
    for c in clips:
        c = c[:SR]  # as eval_e2e: the 1 s MSWC clip
        parts += [c, gap]
        spans.append((pos, pos + len(c)))
        pos += len(c) + len(gap)
    audio = np.concatenate(parts)
    return audio + rng.normal(0, 10 ** (-65 / 20), len(audio)).astype(np.float32), spans


def detected(model: Path, audio: np.ndarray, spans: list, gate: bool) -> tuple[float, int]:
    det = Detector(model, VAD_FILE, gate=gate)
    events = det.process(audio)
    w = det.config.window_samples
    hits = [any(a < e.sample <= b + w for e in events) for a, b in spans]
    stray = sum(not any(a < e.sample <= b + w for a, b in spans) for e in events)
    return float(np.mean(hits)), stray


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("base", type=Path)
    p.add_argument("--per-lang", type=int, default=2)
    p.add_argument("--target-fa", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.backends.cudnn.allow_tf32 = False  # the parity check compares against fp32 ONNX

    settings = Settings(models_dir=args.base.parent.parent, device=args.device)
    base = BaseModels(settings).get(args.base.name)
    # eval_e2e's draw: 20 words per language with this seed; take the first ones per language.
    targets = pick_targets(pd.read_parquet(HELDOUT), 20, 10, 10, np.random.default_rng(args.seed))
    per_lang: dict[str, int] = {}
    chosen = []
    for t in targets:
        if per_lang.get(t[0], 0) < args.per_lang:
            per_lang[t[0]] = per_lang.get(t[0], 0) + 1
            chosen.append(t)
    torch_rows = pd.read_csv(args.base / "eval_e2e.csv").set_index(["lang", "word"])

    reader, rows = ClipReader(), []
    rng = np.random.default_rng(args.seed)
    work = Path(tempfile.mkdtemp(prefix="eval_shipped_"))
    for n, (lang, word, enroll, test) in enumerate(chosen, 1):
        job = quick.Job(word, lang, args.target_fa, read_clips(reader, enroll),
                        [str(i) for i in range(len(enroll))])
        out = work / f"{lang}_{n}"
        report = quick.run(base, job, out, settings, lambda *_: None)
        audio, spans = test_stream(read_clips(reader, test), rng)
        v = report["verification"]
        cal = report["calibration"]
        row = {"lang": lang, "word": word, "threshold": cal["fp32"]["threshold"],
               "threshold_int8": cal["int8"]["threshold"],
               "calib_fa": cal["fp32"]["fa_per_hour"], "calib_fa_int8": cal["int8"]["fa_per_hour"],
               "cv_recall": report["cross_validation"]["recall_noisy"],
               "torch_recall": torch_rows["recall@0.5"].get((lang, word), np.nan),
               "torch_verify_fa": torch_rows["verify_fa@0.5"].get((lang, word), np.nan),
               "int8_vs_fp32_max": v["parity"]["int8_vs_fp32_max"],
               "fp32_vs_torch_max": v["parity"]["fp32_vs_torch_max"],
               "seconds": report["timings_s"]["total"]}
        for kind, name in quick.MODEL_FILES.items():
            for gate in (False, True):
                g = "gate" if gate else "no_gate"
                row[f"recall_{kind}_{g}"], row[f"stray_{kind}_{g}"] = detected(out / name, audio, spans, gate)
                row[f"verify_fa_{kind}_{g}"] = v[kind][g]["per_hour"]
        rows.append(row)
        shutil.rmtree(out)
        print(f"[{n}/{len(chosen)}] {lang} {word!r}: thr {row['threshold']:.3f} | recall fp32 "
              f"{row['recall_fp32_no_gate']:.2f} int8 {row['recall_int8_no_gate']:.2f} "
              f"(torch {row['torch_recall']:.2f}) | FA/h fp32 {row['verify_fa_fp32_no_gate']:.2f} "
              f"int8 {row['verify_fa_int8_no_gate']:.2f} | {row['seconds']:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    hours = v["fp32"]["hours"]
    cols = [c for c in df.columns if c not in ("lang", "word")]
    summary = {"targets": len(df), "target_fa_per_hour": args.target_fa, "verify_hours": hours,
               "mean": df[cols].mean().round(4).to_dict(),
               "verify_false_accepts_total": {c: round(float(df[c].sum() * hours))
                                              for c in df.columns if c.startswith("verify_fa_")}}
    df.to_csv(args.base / "eval_shipped.csv", index=False)
    (args.base / "eval_shipped.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
