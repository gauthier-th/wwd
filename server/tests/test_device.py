"""Device measurements uploaded by the web app (PLAN.md M4): benchmarks and field tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from runtime.fieldlog import load
from server import audio
from server.api import create_app
from server.tests.tiny_base import take
from server.worker import Worker


@pytest.fixture
def trained(settings, speech):
    app = create_app(settings)
    client = TestClient(app)
    wid = client.post("/api/wakewords", json={"name": "x", "language": "en"}).json()["id"]
    client.post(f"/api/wakewords/{wid}/samples", data={"kind": "positive"},
                files={"file": ("t.wav", audio.encode_wav(take(speech, 3)), "audio/wav")})
    tid = client.post(f"/api/wakewords/{wid}/trainings", json={}).json()["id"]
    Worker(settings, engine=app.state.engine).run_once()
    assert client.get(f"/api/trainings/{tid}").json()["status"] == "done"
    return client, wid, tid


def test_field_test_upload_report_and_delete(trained, settings):
    client, wid, tid = trained
    report = client.get(f"/api/trainings/{tid}").json()["report"]
    r = client.post(f"/api/trainings/{tid}/fieldtests",
                    json={"model": "int8", "gate": True, "fa_per_hour": 1.0,
                          "device": {"userAgent": "Mozilla/5.0 (Linux; Android 15) Chrome/140",
                                     "platform": "Linux armv8l", "hardwareConcurrency": 8}})
    assert r.status_code == 201, r.text
    rid = r.json()["id"]
    point = next(p for p in report["operating_points"] if p["fa_per_hour"] == 1.0)
    assert r.json()["threshold"] == point["int8"]["threshold"]
    assert client.post(f"/api/trainings/{tid}/fieldtests", json={"fa_per_hour": 0.3}).status_code == 422

    # two minutes of uploads: ticks, one trigger with its clip, stats with a battery level
    rng = np.random.default_rng(0)
    for minute in range(2):
        ticks = np.stack([rng.random(625) * 0.2, rng.random(625)], axis=1).astype("<f4")
        events = [{"time": 1000.0 + minute * 60, "tick": 625 * minute + 100, "sample": 1536 * 101,
                   "score": 0.9, "clip": "event.wav"}] if minute == 1 else []
        stats = [{"time": 1000.0 + 60 * (minute + 1), "audio_s": 60.0 * (minute + 1), "ticks": 625,
                  "gate_open": 300, "events": len(events), "cpu_pct": None, "busy_pct": 3.5,
                  "latency_ms": {"p50": 6.0, "p95": 9.0, "max": 20.0}, "overruns": 0,
                  "battery": 0.9 - 0.01 * minute}]
        files = [("ticks", ("ticks.f32", ticks.tobytes(), "application/octet-stream"))]
        if events:
            files.append(("clips", ("event.wav", audio.encode_wav(np.zeros(16000, np.float32)), "audio/wav")))
        r = client.post(f"/api/fieldtests/{rid}/chunk", data={"data": json.dumps({"stats": stats, "events": events})},
                        files=files)
        assert r.status_code == 200, r.text
    bad = client.post(f"/api/fieldtests/{rid}/chunk", data={"data": "{}"},
                      files=[("ticks", ("t", b"123", "application/octet-stream"))])
    assert bad.status_code == 422

    summary = client.post(f"/api/fieldtests/{rid}/finish").json()
    assert summary["triggers"] == 1 and summary["hours"] == pytest.approx(1250 * 0.096 / 3600, abs=1e-3)
    assert summary["busy_pct"]["mean"] == 3.5 and summary["cpu_pct"]["mean"] is None
    assert summary["battery"]["start"] == 0.9 and summary["battery"]["per_hour"] < 0
    assert summary["host"]["runtime"] == "onnxruntime-web (wasm)"
    assert client.get(f"/api/fieldtests/{rid}/report?true=1").json()["false_accepts"] == 0
    clip = client.get(f"/api/fieldtests/{rid}/clips/event-0001.wav")
    assert clip.status_code == 200 and clip.content[:4] == b"RIFF"
    assert client.post(f"/api/fieldtests/{rid}/chunk", data={"data": "{}"}).status_code == 409

    bench = client.post(f"/api/trainings/{tid}/benchmarks", json={"runs": [{"gate": True, "p50": 5.1}]})
    assert bench.status_code == 201
    runs = client.get(f"/api/trainings/{tid}/device-runs").json()
    assert [r["kind"] for r in runs] == ["fieldtest", "benchmark"]
    assert runs[0]["summary"]["triggers"] == 1 and runs[1]["data"]["runs"][0]["p50"] == 5.1

    log_dir = next((settings.files_dir / "wakewords" / wid).rglob("events.jsonl")).parent
    assert len(load(log_dir).scores) == 1250  # the same layout runtime.listen --log writes
    assert client.delete(f"/api/wakewords/{wid}").status_code == 204
    assert not log_dir.exists()
    assert client.get(f"/api/fieldtests/{rid}/report").status_code == 404
