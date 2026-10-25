"""Benchmark wake-word files on this machine (PLAN.md M4): latency per tick, the CPU the
detector needs in real time, and the real-time factor, with the speech gate on and off.

    python -m runtime.bench model.onnx [model.int8.onnx] --vad vad.onnx [--seconds 60] [--json out.json]
    python -m runtime.bench a/model.onnx b/model.onnx --vad vad.onnx --together  # several wake words

Audio is streamed one tick (1536 samples) at a time, as `runtime.listen` does live: --wav, or by
default the golden speech file of the runtime tests (speech with pauses), looped. "CPU" is the
process CPU time over the audio duration, so 100% is one core fully busy in real time; sessions
are single-threaded, as on a device. --together also streams all the files at once, as one
detector listening for every wake word, with their encoder shared and without sharing it.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np

from runtime.detector import MultiDetector
from runtime.fieldlog import host_info
from runtime.spec import SAMPLE_RATE
from runtime.vad import SileroVad, make_session
from runtime.wav import read_wav

DEFAULT_WAV = Path(__file__).parent / "testdata" / "audio" / "speech.wav"


def load_audio(path: Path | None, seconds: float) -> np.ndarray:
    if path is None and DEFAULT_WAV.exists():
        path = DEFAULT_WAV
    if path is None:  # no test data shipped: noise with bursts, so the gate opens sometimes
        rng = np.random.default_rng(0)
        audio = rng.normal(0, 0.01, int(seconds * SAMPLE_RATE)).astype(np.float32)
        for start in range(SAMPLE_RATE, len(audio) - SAMPLE_RATE, 3 * SAMPLE_RATE):
            audio[start : start + SAMPLE_RATE // 2] += rng.normal(0, 0.2, SAMPLE_RATE // 2)
        return audio
    audio, rate = read_wav(path)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{path}: {rate} Hz, expected {SAMPLE_RATE}")
    reps = int(np.ceil(seconds * SAMPLE_RATE / len(audio)))
    return np.tile(audio, reps)[: int(seconds * SAMPLE_RATE)]


def ms(values: list[float]) -> dict:
    a = np.array(values) * 1000
    return {"p50": round(float(np.percentile(a, 50)), 3), "p95": round(float(np.percentile(a, 95)), 3),
            "max": round(float(a.max()), 3), "mean": round(float(a.mean()), 3)}


def stream(models: list[str], vad: str | None, audio: np.ndarray, gate: bool,
           share_encoder: bool = True) -> dict:
    det = MultiDetector(models, vad, gate=gate, record_ticks=True, batch_size=1, share_encoder=share_encoder)
    hop = det.words[0].config.hop_samples
    det.process(audio[: 2 * SAMPLE_RATE])  # warm up the sessions
    det.reset()
    latency, gate_open, events = [], 0, 0
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for i in range(0, len(audio) - hop + 1, hop):
        started = time.perf_counter()
        events += len(det.process(audio[i : i + hop]))
        latency.append(time.perf_counter() - started)
        gate_open += sum(any(t) for t in zip(*([t.gate_open for t in w.ticks] for w in det.words)))
        for w in det.words:
            w.ticks.clear()
    cpu, wall = time.process_time() - cpu0, time.perf_counter() - wall0
    audio_s = len(latency) * hop / SAMPLE_RATE
    return {"gate": det.gate, "ticks": len(latency), "audio_s": round(audio_s, 2),
            "gate_open": round(gate_open / len(latency), 3), "events": events,
            "latency_ms": ms(latency), "cpu_pct": round(cpu / audio_s * 100, 2),
            "realtime_factor": round(wall / audio_s, 4), "groups": det.groups}


def model_only(model: str, n: int = 100) -> dict:
    """One window through the wake model alone (the work of an open-gate tick, minus the VAD)."""
    session = make_session(model)
    name = session.get_inputs()[0].name
    x = np.random.default_rng(0).normal(0, 0.05, (1, 24000)).astype(np.float32)
    for _ in range(5):
        session.run(None, {name: x})
    times = []
    for _ in range(n):
        started = time.perf_counter()
        session.run(None, {name: x})
        times.append(time.perf_counter() - started)
    return ms(times)


def vad_only(vad: str, audio: np.ndarray, n: int = 300) -> dict:
    v, times = SileroVad(vad), []
    for i in range(n):
        frame = audio[(i * 512) % (len(audio) - 512) :][:512]
        started = time.perf_counter()
        v(frame)
        times.append(time.perf_counter() - started)
    return ms(times[10:])


def show(r: dict, label: str = "") -> dict:
    lat = r["latency_ms"]
    print(f"{label:<22}{'on' if r['gate'] else 'off':<6}{lat['p50']:>7.2f} {lat['p95']:>7.2f} {lat['max']:>8.2f}"
          f"{'':<9}{r['cpu_pct']:>8.1f}%{'':<10}{r['gate_open']:.0%}")
    return r


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("models", nargs="+", help="model.onnx, model.int8.onnx...")
    p.add_argument("--vad", help="vad.onnx (without it, only gate-off runs)")
    p.add_argument("--wav", type=Path, help="16 kHz WAV to stream (default: the runtime's test speech)")
    p.add_argument("--seconds", type=float, default=60.0, help="audio per run")
    p.add_argument("--json", type=Path, help="also write the results here")
    p.add_argument("--together", action="store_true",
                   help="also listen for all the files at once, with and without a shared encoder")
    args = p.parse_args(argv)

    audio = load_audio(args.wav, args.seconds)
    result = {"host": host_info(), "audio_s": round(len(audio) / SAMPLE_RATE, 2),
              "vad_frame_ms": vad_only(args.vad, audio) if args.vad else None, "models": []}
    host = result["host"]
    print(f"{host['cpu']} [{host['machine']}, {host['cores']} cores], onnxruntime {host['onnxruntime']}, "
          f"{result['audio_s']:.0f} s of audio per run")
    if result["vad_frame_ms"]:
        v = result["vad_frame_ms"]
        print(f"VAD per 32 ms frame: {v['p50']:.3f} ms median, {v['p95']:.3f} ms p95")
    print(f"{'file':<22}{'gate':<6}{'tick latency p50/p95/max (ms)':<32}{'CPU % (real time)':<19}gate open")
    for model in args.models:
        entry = {"file": Path(model).name, "window_ms": model_only(model), "runs": []}
        w = entry["window_ms"]
        print(f"{entry['file']:<22}{'':<6}one window: {w['p50']:.2f} ms median, {w['p95']:.2f} ms p95")
        for gate in ([True, False] if args.vad else [False]):
            entry["runs"].append(show(stream([model], args.vad, audio, gate)))
        result["models"].append(entry)
    if args.together:
        result["together"] = []
        for share in (True, False):
            entry = {"share_encoder": share, "runs": []}
            for gate in ([True, False] if args.vad else [False]):
                entry["runs"].append(r := stream(args.models, args.vad, audio, gate, share))
                passes = len(r["groups"])
                label = f"{len(args.models)} words, {passes} pass{'es' if passes > 1 else ''}"
                show(r, label if gate == bool(args.vad) else "")
            result["together"].append(entry)
    result["rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print(f"peak memory {result['rss_mb']} MB")
    if args.json:
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
