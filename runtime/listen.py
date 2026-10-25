"""Listen on the microphone and print wake-word events, log a field test, or record a take.

    python -m runtime.listen model.onnx --vad vad.onnx [--no-gate] [--threshold 0.6] [--show]
    python -m runtime.listen a/model.onnx b/model.onnx --vad vad.onnx   # several wake words
    python -m runtime.listen model.onnx --vad vad.onnx --log fieldtest/     # 24 h field test
    python -m runtime.listen model.onnx --vad vad.onnx --input rec.wav --log out/  # replay a file
    python -m runtime.listen --record take.wav [--seconds 30]

Several models listen at once on one stream, each with its own settings; models made from the
same base model share their encoder, so a second wake word costs almost nothing.

--log writes a field-test directory (runtime/fieldlog.py): every trigger with a clip of the audio
around it, per-minute CPU and latency, and every tick's score; `python -m runtime.fieldreport`
summarizes it. With several models, each gets its own directory inside, named after its wake
word. Recording on the device itself (a Pi and its own microphone) gives the closest training
data: upload the take with the app's "Import a file instead".
"""

from __future__ import annotations

import argparse
import queue
import re
import resource
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from runtime.detector import MultiDetector
from runtime.fieldlog import AudioRing, FieldLog, host_info
from runtime.spec import SAMPLE_RATE
from runtime.wav import read_wav, write_wav

STATS_EVERY_S = 60


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("models", nargs="*", metavar="model",
                   help="model.onnx from a downloaded bundle; give several to listen for each")
    p.add_argument("--vad", help="Silero VAD ONNX file; enables the speech gate")
    p.add_argument("--no-gate", action="store_true", help="score every tick even without speech")
    p.add_argument("--threshold", help="override the calibrated threshold: one value, or one per "
                                       "model, comma-separated")
    p.add_argument("--device", help="sounddevice input device name or index")
    p.add_argument("--show", action="store_true", help="print every tick score")
    p.add_argument("--log", metavar="DIR", help="write a field-test log to this directory")
    p.add_argument("--clip-before", type=float, default=3.0, help="seconds of audio kept before a trigger")
    p.add_argument("--clip-after", type=float, default=1.0, help="seconds of audio kept after it")
    p.add_argument("--input", metavar="WAV", help="read a 16 kHz WAV instead of the microphone")
    p.add_argument("--realtime", action="store_true", help="with --input: pace it like a live mic")
    p.add_argument("--record", metavar="WAV", help="record 16 kHz audio to this file instead")
    p.add_argument("--seconds", type=float, default=30.0, help="length of a --record take")
    args = p.parse_args(argv)
    if not args.models and not args.record:
        p.error("give a model to listen with, or --record FILE")
    thresholds = [float(t) for t in args.threshold.split(",")] if args.threshold else [None]
    if len(thresholds) == 1:
        thresholds *= max(len(args.models), 1)
    if len(thresholds) != len(args.models) and not args.record:
        p.error(f"--threshold: one value, or one per model ({len(args.models)})")

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if args.record:
        import sounddevice as sd  # only the microphone needs it

        return record(sd, args.record, args.seconds, device)

    det = MultiDetector(args.models, args.vad, gate=not args.no_gate,
                        overrides=[{"threshold": t} for t in thresholds],
                        record_ticks=bool(args.show or args.log))
    hop = det.words[0].config.hop_samples
    overruns = [0]
    if args.input:
        source = file_chunks(args.input, hop, args.realtime)
    else:
        import sounddevice as sd

        source = mic_chunks(sd, hop, device, overruns)
    several = len(det.words) > 1
    logs = None
    if args.log:
        meta = {"gate": det.gate, "input": args.input or "microphone", "host": host_info(),
                "clip_before_s": args.clip_before, "clip_after_s": args.clip_after}
        if several:
            meta |= {"words": [w.name for w in det.words], "groups": det.groups}
        logs = [FieldLog(log_dir(args.log, det, word.index), {
            "model": Path(model).name, "metadata": word.metadata, "config": word.config.to_metadata(), **meta})
            for word, model in zip(det.words, args.models, strict=True)]
    names = ", ".join(f"'{w.name}' (threshold {w.config.threshold:.3f})" for w in det.words)
    passes = "" if not several else "; one encoder pass for all" if len(det.groups) == 1 else (
        f"; {len(det.groups)} model runs per tick (only models of the same base model share one)")
    print(f"listening for {names}, gate {'on' if det.gate else 'off'}{passes}"
          f"{f'; logging to {args.log}' if logs else ''}; Ctrl+C to stop")
    try:
        run(det, source, logs, args, overruns)
    except KeyboardInterrupt:
        pass
    return 0


def log_dir(root: str, det: MultiDetector, word: int) -> Path:
    """The field log of one wake word: `root` itself, or with several, `root/<n>-<wake word>`."""
    if len(det.words) == 1:
        return Path(root)
    slug = re.sub(r"[^\w-]+", "-", det.words[word].name.lower()).strip("-")
    return Path(root) / f"{word + 1}-{slug or 'word'}"


def mic_chunks(sd, hop: int, device, overruns: list[int]) -> Iterator[np.ndarray]:
    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, frames, t, status):  # PortAudio thread: hand off, never block
        if status:
            overruns[0] += 1
            print(status, file=sys.stderr)
        chunks.put(indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=hop,
                        device=device, callback=on_audio):
        while True:
            yield chunks.get()


def file_chunks(path: str, hop: int, realtime: bool) -> Iterator[np.ndarray]:
    audio, rate = read_wav(path)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{path}: {rate} Hz, expected {SAMPLE_RATE}")
    started = time.perf_counter()
    for i in range(0, len(audio) - hop + 1, hop):
        if realtime:
            time.sleep(max(0.0, started + (i + hop) / SAMPLE_RATE - time.perf_counter()))
        yield audio[i : i + hop]


class Minute:
    """Counters over one stats period."""

    def __init__(self, overruns: int) -> None:
        self.wall, self.cpu = time.time(), time.process_time()
        self.latency: list[float] = []
        self.ticks = self.gate_open = self.events = 0
        self.overruns = overruns

    def record(self, audio_s: float, overruns: int, hop: int) -> dict:
        wall = max(time.time() - self.wall, 1e-9)
        lat = np.array(self.latency or [0.0]) * 1000
        return {
            "time": time.time(), "audio_s": round(audio_s, 3), "ticks": self.ticks,
            "gate_open": self.gate_open, "events": self.events,
            "cpu_pct": round((time.process_time() - self.cpu) / wall * 100, 2),
            # detector time over the audio it covered: the share of one core it needs live
            "busy_pct": round(float(np.sum(self.latency)) / max(self.ticks * hop / SAMPLE_RATE, 1e-9) * 100, 2),
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 3),
                           "p95": round(float(np.percentile(lat, 95)), 3),
                           "max": round(float(lat.max()), 3)},
            "overruns": overruns - self.overruns,
            "rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        }


def run(det: MultiDetector, source: Iterator[np.ndarray], logs: list[FieldLog] | None, args,
        overruns: list[int]) -> None:
    hop = det.words[0].config.hop_samples
    several = len(det.words) > 1
    before, after = int(args.clip_before * SAMPLE_RATE), int(args.clip_after * SAMPLE_RATE)
    ring = AudioRing(args.clip_before + args.clip_after + 1.0) if logs else None
    pending: list[tuple[int, dict, int]] = []  # (word, record, sample) waiting for their audio after
    minutes = [Minute(overruns[0]) for _ in det.words]  # one per log: the CPU is the detector's
    pos = stats_from = 0

    def flush(final: bool = False) -> None:
        while pending and (final or pending[0][2] + after <= pos):
            word, record, sample = pending.pop(0)
            logs[word].event(record, ring.slice(sample - before, sample + after))

    try:
        for chunk in source:
            started = time.perf_counter()
            events = det.process(chunk)
            took = time.perf_counter() - started
            pos += len(chunk)
            ticks = [w.ticks for w in det.words]
            for w in det.words:
                w.ticks = []
            for ev in events:
                name = f"  '{det.words[ev.word].name}'" if several else ""
                print(f"{time.strftime('%H:%M:%S')}  WAKE{name}  score={ev.score:.3f}", flush=True)
            if args.show:
                for i in range(len(ticks[0])):
                    print(f"  tick {ticks[0][i].index:6d}" + " |".join(
                        f"{f' {w.name}:' if several else ''} vad={t[i].vad_max:.2f} "
                        f"{'open' if t[i].gate_open else 'shut'} score={t[i].score:.3f}"
                        for w, t in zip(det.words, ticks, strict=True)))
            if logs is None:
                continue
            ring.push(chunk)
            for w, log, minute, word_ticks in zip(det.words, logs, minutes, ticks, strict=True):
                mine = [ev for ev in events if ev.word == w.index]
                log.ticks(np.array([t.score for t in word_ticks]), np.array([t.vad_max for t in word_ticks]))
                minute.latency.append(took)
                minute.ticks += len(word_ticks)
                minute.gate_open += sum(t.gate_open for t in word_ticks)
                minute.events += len(mine)
                pending += [(w.index, {"time": time.time(), "tick": ev.tick, "sample": ev.sample,
                                       "score": round(ev.score, 5)}, ev.sample) for ev in mine]
            flush()
            if pos - stats_from >= STATS_EVERY_S * SAMPLE_RATE:
                for log, minute in zip(logs, minutes, strict=True):
                    log.stats(minute.record(pos / SAMPLE_RATE, overruns[0], hop))
                minutes, stats_from = [Minute(overruns[0]) for _ in det.words], pos
    finally:
        if logs is not None:
            flush(final=True)
            for log, minute in zip(logs, minutes, strict=True):
                if minute.ticks:
                    log.stats(minute.record(pos / SAMPLE_RATE, overruns[0], hop))
                log.close(audio_s=pos / SAMPLE_RATE, ticks=int(pos // hop))


def record(sd, path: str, seconds: float, device) -> int:
    print(f"recording {seconds:g} s to {path}; Ctrl+C to stop early")
    frames: list[np.ndarray] = []
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device) as stream:
            while sum(len(f) for f in frames) < seconds * SAMPLE_RATE:
                block, _ = stream.read(SAMPLE_RATE // 10)
                frames.append(block[:, 0].copy())
    except KeyboardInterrupt:
        pass
    audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
    write_wav(path, audio[: int(seconds * SAMPLE_RATE)])
    print(f"wrote {len(audio) / SAMPLE_RATE:.1f} s; import it in the app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
