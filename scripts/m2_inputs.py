"""Inputs for the M2 smoke run (scripts/m2_smoke.sh): what a user would record for one held-out
word, plus test audio.

    uv run --extra training python -m scripts.m2_inputs --lang fr --out data/smoke/fr/inputs

Writes 16 kHz WAVs:
  take.wav          10 enrollment clips in one take, as the web app records them
  other_speech.wav  other words by the same speakers (stands in for the read paragraph)
  ambience.wav      15 s of MUSAN noise
  test_word.wav     other speakers saying the word, 3 s apart (recall through the runtime)
  test_speech.wav   2 min of a held-out long-form stream (parity only)
  meta.json         word, speakers, and where each test clip sits in test_word.wav
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from runtime.wav import write_wav
from training.eval_e2e import pick_targets, read_clips
from training.evaluate import HELDOUT
from training.paths import DATA, MODELS, MUSAN
from training.store import ClipReader, read_wav_16k

SR = 16000
FLOOR = 10 ** (-65 / 20)  # a recording never has digital silence


def joined(clips: list[np.ndarray], gap_s: float, lead_s: float, rng: np.random.Generator
           ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    parts, spans, pos = [np.zeros(int(lead_s * SR), np.float32)], [], int(lead_s * SR)
    for c in clips:
        parts += [c, np.zeros(int(gap_s * SR), np.float32)]
        spans.append((pos, pos + len(c)))
        pos += len(c) + int(gap_s * SR)
    audio = np.concatenate(parts)
    return audio + rng.normal(0, FLOOR, len(audio)).astype(np.float32), spans


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--lang", default="fr")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-chars", type=int, default=7, help="wake words this long or longer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base", default="v0.1")
    args = p.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    heldout = pd.read_parquet(HELDOUT)
    heldout = heldout[(heldout.lang == args.lang) & (heldout.word.str.len() >= args.min_chars)]
    _, word, enroll, test = pick_targets(heldout, 1, 10, 15, rng)[0]
    reader = ClipReader()
    take, _ = joined(read_clips(reader, enroll), 0.8, 0.6, rng)
    write_wav(args.out / "take.wav", take)

    speakers = set(enroll.speaker)
    pool = pd.concat([pd.read_parquet(DATA / "manifests" / "train.parquet",
                                      columns=["lang", "word", "speaker", "blob", "offset", "length"]),
                      heldout[["lang", "word", "speaker", "blob", "offset", "length"]]])
    own = pool[(pool.lang == args.lang) & pool.speaker.isin(speakers) & (pool.word != word)]
    own = own.sample(min(len(own), 40), random_state=args.seed)
    if len(own) < 20:  # these speakers said little else: top up with other speakers
        rest = pool[(pool.lang == args.lang) & (pool.word != word)].sample(40 - len(own),
                                                                             random_state=args.seed)
        own = pd.concat([own, rest])
    other, _ = joined(read_clips(reader, own), 0.25, 0.3, rng)
    write_wav(args.out / "other_speech.wav", other)

    noise, files = [], sorted((MUSAN / "noise" / "free-sound").glob("*.wav"))
    for f in rng.permutation(len(files)):
        noise.append(read_wav_16k(files[f]))
        if sum(map(len, noise)) >= 15 * SR:
            break
    noise = np.concatenate(noise)[: 15 * SR]
    write_wav(args.out / "ambience.wav", noise / max(np.abs(noise).max(), 1e-3) * 0.3)

    test_audio, spans = joined(read_clips(reader, test), 2.0, 1.0, rng)
    write_wav(args.out / "test_word.wav", test_audio)
    verify = np.load(MODELS / "base" / args.base / "longform_verify.npz")
    write_wav(args.out / "test_speech.wav",
              verify[min(verify.files)][: 120 * SR].astype(np.float32) / 32768)

    meta = {"word": word, "language": args.lang, "enroll_speakers": len(speakers),
            "other_speech_own_clips": int(own.speaker.isin(speakers).sum()),
            "test_clips": len(test), "test_spans": {"test_word.wav": spans}}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in meta.items() if k != "test_spans"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
