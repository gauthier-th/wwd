"""Field-test tooling (PLAN.md M4): the log written by runtime.listen, the report, the bench."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from runtime import Detector
from runtime.fieldlog import AudioRing, load
from runtime.fieldreport import summarize, trigger_ticks
from runtime.spec import DetectorConfig
from runtime.wav import read_wav

RUNTIME = Path(__file__).resolve().parents[1]
DATA = RUNTIME / "testdata"
MODEL = DATA / "models" / "model.onnx"
VAD = RUNTIME / "assets" / "silero_vad_16k_op15.onnx"
SPEECH = DATA / "audio" / "speech.wav"


def test_audio_ring_keeps_the_last_seconds_by_absolute_position():
    ring = AudioRing(1.0)  # 16000 samples
    stream = np.arange(40000, dtype=np.float32)
    for i in range(0, len(stream), 1536):
        ring.push(stream[i : i + 1536])
    np.testing.assert_array_equal(ring.slice(30000, 30010), stream[30000:30010])
    assert len(ring.slice(0, 100)) == 0  # long gone
    np.testing.assert_array_equal(ring.slice(39990, 50000), stream[39990:40000])  # not there yet


def test_trigger_replay_matches_the_runtime_trigger():
    from runtime.detector import Trigger

    rng = np.random.default_rng(0)
    scores = rng.random(3000) ** 4
    cfg = DetectorConfig()
    for thr in (0.2, 0.5, 0.8):
        trig = Trigger(cfg.with_overrides(threshold=thr))
        assert trigger_ticks(scores, thr, cfg) == [t for t, s in enumerate(scores) if trig.update(t, s)]


def test_listen_replays_a_file_into_a_field_log(tmp_path, capsys):
    from runtime.listen import main

    out = tmp_path / "run"
    assert main([str(MODEL), "--vad", str(VAD), "--input", str(SPEECH), "--log", str(out)]) == 0
    expected = Detector(MODEL, VAD).process(read_wav(SPEECH)[0])
    run = load(out)
    assert [e["tick"] for e in run.events] == [e.tick for e in expected]
    assert len(run.scores) == len(read_wav(SPEECH)[0]) // 1536
    for i, e in enumerate(run.events, 1):  # 3 s before + 1 s after each trigger, when available
        clip, rate = read_wav(out / e["clip"])
        assert rate == 16000 and 0 < len(clip) <= 4 * 16000
        assert e["clip"] == f"clips/event-{i:04d}.wav"
    assert run.stats and run.stats[-1]["ticks"] > 0
    meta = json.loads((out / "meta.json").read_text())
    assert meta["metadata"]["wake_word"] == "golden-stub" and meta["ended_at"] >= meta["started_at"]
    assert meta["host"]["onnxruntime"]

    r = summarize(run, {1})
    assert r["triggers"] == len(expected) and r["false_accepts"] == len(expected) - 1
    at_threshold = next(s for s in r["sweep"] if s["factor"] == 1.0)
    assert at_threshold["triggers"] == len(expected)  # the replay reproduces the live run
    assert 0 < r["speech_hours"] < r["hours"]


def test_bench_runs_and_writes_json(tmp_path, capsys):
    from runtime.bench import main

    out = tmp_path / "bench.json"
    assert main([str(MODEL), "--vad", str(VAD), "--seconds", "4", "--json", str(out)]) == 0
    r = json.loads(out.read_text())
    [entry] = r["models"]
    assert [run["gate"] for run in entry["runs"]] == [True, False]
    for run in entry["runs"]:
        assert run["ticks"] == (4 * 16000) // 1536 and run["cpu_pct"] > 0
        assert run["latency_ms"]["p50"] <= run["latency_ms"]["max"]
    assert r["vad_frame_ms"]["p50"] > 0 and r["host"]["cores"] >= 1
