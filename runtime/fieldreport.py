"""Summarize a field-test log (runtime/fieldlog.py): hours of audio, triggers, false accepts
per hour, CPU and latency, and the trigger rate the detector would have had at other thresholds.

    python -m runtime.fieldreport fieldtest/ [--true 3,7] [--json]

Every trigger counts as a false accept unless its number is listed with --true: listen to the
clips in fieldtest/clips/ to decide. The threshold sweep replays the runtime's trigger (k-of-n
smoothing, refractory period) over the logged tick scores, which are 0 while the speech gate was
closed, so it is exact for the gate setting the run used.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from runtime.fieldlog import FieldRun, load
from runtime.spec import DetectorConfig


def trigger_ticks(scores: np.ndarray, threshold: float, cfg: DetectorConfig) -> list[int]:
    """Ticks at which `runtime.detector.Trigger` fires over one continuous stream."""
    hits = (scores.astype(np.float64) >= threshold).astype(np.int32)
    window = np.convolve(hits, np.ones(cfg.smoothing_n, np.int32))[: len(hits)]
    fired, next_allowed = [], 0
    for t in np.flatnonzero(window >= cfg.smoothing_k):
        if t >= next_allowed:
            fired.append(int(t))
            next_allowed = t + cfg.refractory_ticks + 1
    return fired


def _pct(values: list[float | None], q: float) -> float | None:
    values = [v for v in values if v is not None]  # browsers can't measure process CPU
    return round(float(np.percentile(values, q)), 2) if values else None


def summarize(run: FieldRun, true_events: set[int] | frozenset[int] = frozenset()) -> dict:
    cfg = DetectorConfig.from_metadata(run.meta["config"])
    tick_s = cfg.hop_samples / cfg.sample_rate
    hours = len(run.scores) * tick_s / 3600
    gate = run.meta.get("gate", True) and not np.all(np.isnan(run.vad_max))
    speech_hours = (float(np.sum(run.vad_max >= cfg.vad_threshold)) * tick_s / 3600) if gate else hours
    false_accepts = [e for i, e in enumerate(run.events, 1) if i not in true_events]
    stats = run.stats
    sweep = []
    for factor in (0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0):
        thr = cfg.threshold * factor
        if 0 < thr < 1:
            n = len(trigger_ticks(run.scores, thr, cfg))
            sweep.append({"factor": factor, "threshold": round(thr, 4), "triggers": n,
                          "per_hour": round(n / hours, 3) if hours else None})
    return {
        "wake_word": run.meta.get("metadata", {}).get("wake_word"),
        "model": run.meta.get("model"),
        "host": run.meta.get("host"),
        "input": run.meta.get("input"),
        "gate": gate,
        "threshold": cfg.threshold,
        "hours": round(hours, 3),
        "speech_hours": round(speech_hours, 3),
        "triggers": len(run.events),
        "true_accepts": len(run.events) - len(false_accepts),
        "false_accepts": len(false_accepts),
        "false_accepts_per_hour": round(len(false_accepts) / hours, 3) if hours else None,
        "false_accepts_per_speech_hour": round(len(false_accepts) / speech_hours, 3) if speech_hours else None,
        "cpu_pct": {"mean": _pct([s["cpu_pct"] for s in stats], 50), "p95": _pct([s["cpu_pct"] for s in stats], 95)},
        "busy_pct": {"mean": round(float(np.mean([s["busy_pct"] for s in stats])), 2) if stats else None,
                     "max": max((s["busy_pct"] for s in stats), default=None)},
        "latency_ms": {"p50": _pct([s["latency_ms"]["p50"] for s in stats], 50),
                       "p95": _pct([s["latency_ms"]["p95"] for s in stats], 50),
                       "max": max((s["latency_ms"]["max"] for s in stats), default=None)},
        "overruns": sum(s.get("overruns", 0) for s in stats),
        "rss_mb": max((s.get("rss_mb", 0) for s in stats), default=None),
        "battery": _battery(stats),
        "sweep": sweep,
        "events": [{"n": i, **e, "false_accept": i not in true_events} for i, e in enumerate(run.events, 1)],
    }


def _battery(stats: list[dict]) -> dict | None:
    """Phones (web app) log the battery level; a Pi doesn't."""
    levels = [(s["time"], s["battery"]) for s in stats if s.get("battery") is not None]
    if len(levels) < 2:
        return None
    (t0, b0), (t1, b1) = levels[0], levels[-1]
    return {"start": b0, "end": b1, "per_hour": round((b1 - b0) / max((t1 - t0) / 3600, 1e-9), 3)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("log", help="directory written by runtime.listen --log (or the web device test)")
    p.add_argument("--true", default="", help="comma-separated numbers of triggers that were real wake words")
    p.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    args = p.parse_args(argv)
    true_events = {int(x) for x in args.true.split(",") if x.strip()}
    r = summarize(load(args.log), true_events)
    if args.json:
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0
    host = r["host"] or {}
    print(f"'{r['wake_word']}' ({r['model']}) on {host.get('cpu', '?')} [{host.get('machine', '?')}], "
          f"input {r['input']}, gate {'on' if r['gate'] else 'off'}, threshold {r['threshold']:.4f}")
    print(f"{r['hours']:.2f} h of audio, {r['speech_hours']:.2f} h with the gate open")
    print(f"triggers {r['triggers']} ({r['true_accepts']} marked true): "
          f"{r['false_accepts']} false accepts = {r['false_accepts_per_hour']} per hour, "
          f"{r['false_accepts_per_speech_hour']} per speech hour")
    print(f"CPU {r['cpu_pct']['mean']}% median, {r['cpu_pct']['p95']}% p95 (process, one core = 100%); "
          f"detector busy {r['busy_pct']['mean']}% of real time")
    print(f"latency per tick: {r['latency_ms']['p50']} ms median, {r['latency_ms']['p95']} ms p95, "
          f"{r['latency_ms']['max']} ms max; overruns {r['overruns']}; memory {r['rss_mb']} MB")
    if r["battery"]:
        b = r["battery"]
        print(f"battery {b['start']:.0%} -> {b['end']:.0%} ({b['per_hour']:+.1%} per hour)")
    print("threshold sweep (triggers per hour if the threshold had been...):")
    for s in r["sweep"]:
        mark = "  <- this run" if s["factor"] == 1.0 else ""
        print(f"  {s['threshold']:.4f}: {s['triggers']:5d} triggers, {s['per_hour']} per hour{mark}")
    for e in r["events"]:
        when = f"{e['sample'] / 16000:9.1f} s"
        print(f"  #{e['n']:<3d} {when}  score {e['score']:.3f}  {e.get('clip', '')}"
              f"{'' if e['false_accept'] else '  (true)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
