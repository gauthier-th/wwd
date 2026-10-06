"""Model-independent data for every base model: the negative bank and the long-form sets.

    uv run --extra training python -m training.build_bank [--bank-size 200000]

Needs training.prepare to have run (it defines eval speakers and held-out words). Writes:
  data/bank/manifest.parquet, data/bank/audio.int16      200k single-word clips (PLAN.md §8)
  data/longform/streams.parquet, streams/<id>.npy        ~5-minute 16 kHz int16 streams
  data/longform/occurrences.parquet                      where each MSWC word was placed
Model-dependent embeddings of both are computed by training.export.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from training.paths import DATA, LANGUAGES, LIBRISPEECH, MLS, MUSAN
from training.prepare import MANIFESTS, bucket
from training.store import ClipReader, load_index, read_wav_16k

BANK = DATA / "bank"
LONGFORM = DATA / "longform"
SR = 16000
CLIP = SR  # bank clips are stored as exactly 1 s
STREAM_SECONDS = 300
_reader = ClipReader()


def _decode(args: tuple[str, int, int]) -> np.ndarray:
    a = _reader.read(*args)[:CLIP]
    return np.pad(a, (0, CLIP - len(a)))


def fade(clip: np.ndarray, n: int = 32) -> np.ndarray:
    """Short raised-cosine fade in/out, so concatenated clips don't click."""
    ramp = np.hanning(2 * n)[:n]
    out = clip.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def decode_many(frame: pd.DataFrame, workers: int) -> list[np.ndarray]:
    jobs = list(zip(frame.blob, frame.offset, frame.length, strict=True))
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_decode, jobs, chunksize=256))


# --------------------------------------------------------------------------- bank


def select_bank(size: int, rng: np.random.Generator) -> pd.DataFrame:
    pools = {}
    for lang in LANGUAGES:
        df = load_index(lang)
        if df.empty:
            print(f"{lang}: not downloaded, no bank clips")
            continue
        df = df[df.speaker.map(lambda s: bucket(f"spk:{s}") != 0)]  # no eval speakers
        df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
        df = df.assign(turn=df.groupby(["word", "speaker"]).cumcount())
        df = df.sort_values(["word", "turn"]).groupby("word", group_keys=False).head(3)
        pools[lang] = df[df.groupby("speaker").cumcount() < 200].drop(columns="turn")
    quota = {lang: 0 for lang in pools}
    remaining = size
    while remaining > 0:  # equal shares; what small languages can't fill goes to the others
        open_ = [lang for lang in pools if quota[lang] < len(pools[lang])]
        if not open_:
            break
        share = max(1, remaining // len(open_))
        for lang in open_:
            take = min(share, len(pools[lang]) - quota[lang], remaining)
            quota[lang] += take
            remaining -= take
    parts = [pools[lang].sample(quota[lang], random_state=0) for lang in pools]
    return pd.concat(parts, ignore_index=True)


def build_bank(size: int, workers: int, rng: np.random.Generator) -> None:
    BANK.mkdir(parents=True, exist_ok=True)
    frame = select_bank(size, rng)
    print("bank per language:", frame.lang.value_counts().to_dict())
    audio = np.lib.format.open_memmap(BANK / "audio.npy", mode="w+", dtype=np.int16,
                                      shape=(len(frame), CLIP))
    for i, clip in enumerate(decode_many(frame, workers)):
        audio[i] = np.clip(np.round(clip * 32767), -32768, 32767)
    audio.flush()
    frame.drop(columns=["blob", "offset", "length"]).to_parquet(BANK / "manifest.parquet")
    print(f"bank: {len(frame):,} clips -> {BANK}")


# --------------------------------------------------------------------------- long-form


class StreamWriter:
    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.noise: np.ndarray | None = None  # mixed into 30% of speech streams when set
        self.streams: list[dict] = []
        self.occurrences: list[dict] = []
        (LONGFORM / "streams").mkdir(parents=True, exist_ok=True)

    def write(self, sid: str, lang: str, kind: str, audio: np.ndarray, occ: list[tuple]) -> None:
        noisy = kind == "speech" and self.noise is not None and self.rng.random() < 0.3
        if noisy:
            start = int(self.rng.integers(0, len(self.noise) - len(audio)))
            n = self.noise[start : start + len(audio)]
            speech_rms = np.sqrt(np.mean(audio**2)) + 1e-9
            snr = self.rng.uniform(5, 25)
            audio = audio + n / (np.sqrt(np.mean(n**2)) + 1e-9) * speech_rms / 10 ** (snr / 20)
        audio = audio + self.rng.normal(0, 10 ** (-75 / 20), len(audio))  # never digital silence
        pcm = np.clip(np.round(audio * 32767), -32768, 32767).astype(np.int16)
        np.save(LONGFORM / "streams" / f"{sid}.npy", pcm)
        split = "verify" if bucket(f"stream:{sid}") == 0 else "calib"
        self.streams.append({"id": sid, "lang": lang, "kind": kind, "noisy": noisy,
                             "samples": len(pcm), "split": split})
        self.occurrences += [{"id": sid, "word": w, "start": a, "end": b} for w, a, b in occ]


def word_salad(lang: str, minutes: float, writer: StreamWriter, workers: int) -> None:
    """Eval-speaker clips of training words, as utterances of 3-10 words with pauses."""
    rng = writer.rng
    heldout = set(pd.read_parquet(MANIFESTS / "heldout.parquet", columns=["lang", "word"])
                  .query("lang == @lang").word)
    df = load_index(lang)
    if df.empty:
        return
    df = df[df.speaker.map(lambda s: bucket(f"spk:{s}") == 0) & ~df.word.isin(heldout)]
    df = df.sample(min(len(df), int(minutes * 60 * 1.1)), random_state=int(rng.integers(1 << 31)))
    clips = decode_many(df, workers)
    order = np.argsort(df.speaker.to_numpy(), kind="stable")  # utterances by one speaker
    audio, occ, n, sid = [], [], 0, 0
    i = 0
    while i < len(order):
        k = int(rng.integers(3, 11))
        level = 10 ** (rng.uniform(-35, -15) / 20)
        for j in order[i : i + k]:
            c = fade(clips[j])
            c = c / (np.sqrt(np.mean(c**2)) + 1e-9) * level
            occ.append((df.word.iat[j], n, n + len(c)))
            audio.append(c)
            n += len(c)
            gap = np.zeros(int(rng.uniform(0.0, 0.2) * SR))
            audio.append(gap)
            n += len(gap)
        pause = np.zeros(int(rng.uniform(0.5, 2.5) * SR))
        audio.append(pause)
        n += len(pause)
        i += k
        if n >= STREAM_SECONDS * SR or i >= len(order):
            writer.write(f"{lang}-salad-{sid:03d}", lang, "speech", np.concatenate(audio), occ)
            audio, occ, n, sid = [], [], 0, sid + 1


def real_speech(lang: str, minutes: float, writer: StreamWriter) -> None:
    """Continuous read speech: MLS (de/fr/es/it/pl) or LibriSpeech (en)."""
    if lang == "en":
        files = sorted(LIBRISPEECH.rglob("*.flac"))
    else:
        files = sorted((MLS / lang).rglob("*.flac")) + sorted((MLS / lang).rglob("*.opus"))
    if not files:
        print(f"{lang}: no real long-form speech available, word salad only")
        return
    rng = writer.rng
    audio, n, sid, total = [], 0, 0, 0
    for f in rng.permutation(files):
        a = read_wav_16k(f)
        a = a / (np.sqrt(np.mean(a**2)) + 1e-9) * 10 ** (rng.uniform(-35, -15) / 20)
        audio += [a, np.zeros(int(rng.uniform(0.3, 1.5) * SR))]
        n += len(audio[-2]) + len(audio[-1])
        if n >= STREAM_SECONDS * SR:
            writer.write(f"{lang}-read-{sid:03d}", lang, "speech", np.concatenate(audio), [])
            total += n
            audio, n, sid = [], 0, sid + 1
            if total >= minutes * 60 * SR:
                return


def noise_streams(hours: float, writer: StreamWriter) -> None:
    files = sorted((MUSAN / "noise").rglob("*.wav")) + sorted((MUSAN / "music").rglob("*.wav"))
    if not files:
        print("no MUSAN audio: long-form has no noise/music streams")
        return
    rng = writer.rng
    picked = rng.permutation(files)
    # A separate slice of MUSAN mixes into 30% of the speech streams.
    writer.noise = np.concatenate([read_wav_16k(f) for f in picked[:60]])
    audio, n, sid, total = [], 0, 0, 0
    for f in picked[60:]:
        a = read_wav_16k(f)
        a = a / (np.sqrt(np.mean(a**2)) + 1e-9) * 10 ** (rng.uniform(-40, -15) / 20)
        audio.append(a)
        n += len(a)
        if n >= STREAM_SECONDS * SR:
            writer.write(f"musan-{sid:03d}", "xx", "noise", np.concatenate(audio), [])
            total += n
            audio, n, sid = [], 0, sid + 1
            if total >= hours * 3600 * SR:
                return


def build_longform(minutes_salad: float, minutes_real: float, noise_hours: float, workers: int,
                   rng: np.random.Generator) -> None:
    writer = StreamWriter(rng)
    noise_streams(noise_hours, writer)
    for lang in LANGUAGES:
        word_salad(lang, minutes_salad, writer, workers)
        real_speech(lang, minutes_real, writer)
        print(f"longform {lang}: {sum(s['samples'] for s in writer.streams if s['lang'] == lang) / SR / 60:.0f} min")
    pd.DataFrame(writer.streams).to_parquet(LONGFORM / "streams.parquet")
    pd.DataFrame(writer.occurrences).to_parquet(LONGFORM / "occurrences.parquet")
    hours = sum(s["samples"] for s in writer.streams) / SR / 3600
    print(f"longform: {len(writer.streams)} streams, {hours:.1f} h -> {LONGFORM}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--bank-size", type=int, default=200_000)
    p.add_argument("--salad-minutes", type=float, default=40)
    p.add_argument("--real-minutes", type=float, default=20)
    p.add_argument("--noise-hours", type=float, default=2)
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--only", choices=["bank", "longform"])
    args = p.parse_args(argv)
    rng = np.random.default_rng(0)
    if args.only != "longform":
        build_bank(args.bank_size, args.workers, rng)
    if args.only != "bank":
        build_longform(args.salad_minutes, args.real_minutes, args.noise_hours, args.workers, rng)


if __name__ == "__main__":
    main()
