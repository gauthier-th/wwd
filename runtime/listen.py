"""Listen on the microphone and print wake-word events, log a field test, or record a take.

    python -m runtime.listen model.onnx --vad vad.onnx [--no-gate] [--threshold 0.6] [--show]
    python -m runtime.listen model.onnx --vad vad.onnx --log fieldtest/     # 24 h field test
    python -m runtime.listen model.onnx --vad vad.onnx --input rec.wav --log out/  # replay a file
    python -m runtime.listen --record take.wav [--seconds 30]

--log writes a field-test directory (runtime/fieldlog.py): every trigger with a clip of the audio
around it, per-minute CPU and latency, and every tick's score; `python -m runtime.fieldreport`
summarizes it. Recording on the device itself (a Pi and its own microphone) gives the closest
training data: upload the take with the app's "Import a file instead".
"""

from __future__ import annotations

import argparse
import queue
import resource
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from runtime.detector import Detector
from runtime.fieldlog import AudioRing, FieldLog, host_info
from runtime.spec import SAMPLE_RATE
from runtime.wav import read_wav, write_wav

STATS_EVERY_S = 60


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model", nargs="?", help="model.onnx from the downloaded bundle")
    p.add_argument("--vad", help="Silero VAD ONNX file; enables the speech gate")
    p.add_argument("--no-gate", action="store_true", help="score every tick even without speech")
    p.add_argument("--threshold", type=float, help="override the calibrated threshold")
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
    if not args.model and not args.record:
        p.error("give a model to listen with, or --record FILE")

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if args.record:
        import sounddevice as sd  # only the microphone needs it

        return record(sd, args.record, args.seconds, device)

    det = Detector(args.model, args.vad, gate=not args.no_gate, overrides={"threshold": args.threshold},
                   record_ticks=bool(args.show or args.log))
    hop = det.config.hop_samples
    overruns = [0]
    if args.input:
        source = file_chunks(args.input, hop, args.realtime)
    else:
        import sounddevice as sd

        source = mic_chunks(sd, hop, device, overruns)
    log = None
    if args.log:
        log = FieldLog(args.log, {
            "model": Path(args.model).name, "metadata": det.metadata, "config": det.config.to_metadata(),
            "gate": det.gate, "input": args.input or "microphone", "host": host_info(),
            "clip_before_s": args.clip_before, "clip_after_s": args.clip_after})
    name = det.metadata.get("wake_word", "?")
    print(f"listening for '{name}' (threshold {det.config.threshold:.3f}, gate "
          f"{'on' if det.gate else 'off'}){f'; logging to {args.log}' if log else ''}; Ctrl+C to stop")
    try:
        run(det, source, log, args, overruns)
    except KeyboardInterrupt:
        pass
    return 0


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


def run(det: Detector, source: Iterator[np.ndarray], log: FieldLog | None, args, overruns: list[int]) -> None:
    hop = det.config.hop_samples
    before, after = int(args.clip_before * SAMPLE_RATE), int(args.clip_after * SAMPLE_RATE)
    ring = AudioRing(args.clip_before + args.clip_after + 1.0) if log else None
    pending: list[tuple[dict, int]] = []  # triggers waiting for their audio after
    minute = Minute(overruns[0])
    pos = stats_from = 0

    def flush(final: bool = False) -> None:
        while pending and (final or pending[0][1] + after <= pos):
            record, sample = pending.pop(0)
            log.event(record, ring.slice(sample - before, sample + after))

    try:
        for chunk in source:
            started = time.perf_counter()
            events = det.process(chunk)
            minute.latency.append(time.perf_counter() - started)
            pos += len(chunk)
            ticks, det.ticks = det.ticks, []
            for ev in events:
                print(f"{time.strftime('%H:%M:%S')}  WAKE  score={ev.score:.3f}", flush=True)
            if args.show:
                for tick in ticks:
                    gate = "open" if tick.gate_open else "shut"
                    print(f"  tick {tick.index:6d} vad={tick.vad_max:.2f} {gate} score={tick.score:.3f}")
            if log is None:
                continue
            ring.push(chunk)
            log.ticks(np.array([t.score for t in ticks]), np.array([t.vad_max for t in ticks]))
            minute.ticks += len(ticks)
            minute.gate_open += sum(t.gate_open for t in ticks)
            minute.events += len(events)
            pending += [({"time": time.time(), "tick": ev.tick, "sample": ev.sample,
                          "score": round(ev.score, 5)}, ev.sample) for ev in events]
            flush()
            if pos - stats_from >= STATS_EVERY_S * SAMPLE_RATE:
                log.stats(minute.record(pos / SAMPLE_RATE, overruns[0], hop))
                minute, stats_from = Minute(overruns[0]), pos
    finally:
        if log is not None:
            flush(final=True)
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
