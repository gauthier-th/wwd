"""The whole server flow against the tiny base model: create, upload, train (in-process
worker), stream progress, download, check the files with the runtime, delete."""

from __future__ import annotations

import io
import json
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.export import check_graph, read_metadata
from runtime import Detector
from server import audio
from server.api import create_app
from server.bundle import ARTIFACTS, BUNDLE_FILES
from server.tests.tiny_base import take
from server.worker import Worker


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
def client(app):
    return TestClient(app)


def wav(x: np.ndarray) -> bytes:
    return audio.encode_wav(x)


def upload(client, wid, kind, data, name="take.wav"):
    return client.post(f"/api/wakewords/{wid}/samples", data={"kind": kind},
                       files={"file": (name, data, "audio/wav")})


def sse(client, tid) -> list[tuple[str, dict]]:
    events, current = [], {}
    with client.stream("GET", f"/api/trainings/{tid}/events") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("event: "):
                current["event"] = line[7:]
            elif line.startswith("data: "):
                current["data"] = json.loads(line[6:])
            elif not line and current:
                events.append((current["event"], current["data"]))
                current = {}
    return events


def read_metadata_bytes(data: bytes) -> dict[str, str]:
    import onnx

    return {p.key: p.value for p in onnx.load_from_string(data).metadata_props}


def test_languages_and_base_models(client):
    langs = client.get("/api/languages").json()
    assert [x["code"] for x in langs] == ["en", "de", "fr", "ca", "rw", "es", "fa", "it", "ru",
                                          "pl", "eu", "cy"]
    assert next(x for x in langs if x["code"] == "fa")["rtl"]
    assert all(len(x["paragraph"].split()) >= 50 for x in langs if x["code"] != "fa")
    [base] = client.get("/api/base-models").json()
    assert base["version"] == "tiny" and base["default"]


def test_validation(client):
    assert client.post("/api/wakewords", json={"name": "x", "language": "xx"}).status_code == 422
    assert client.post("/api/wakewords", json={"name": "  ", "language": "en"}).status_code == 422
    wid = client.post("/api/wakewords", json={"name": " Hey   Nova ", "language": "en"}).json()["id"]
    assert client.get(f"/api/wakewords/{wid}").json()["name"] == "Hey Nova"
    r = upload(client, wid, "positive", wav(np.zeros(32000, np.float32)))
    assert r.status_code == 422 and "silent" in r.json()["detail"]
    assert upload(client, wid, "positive", b"garbage").status_code == 422
    assert upload(client, wid, "bogus", b"x").status_code == 422
    r = client.post(f"/api/wakewords/{wid}/trainings", json={"mode": "quick"})
    assert r.status_code == 422 and "at least 3" in r.json()["detail"]
    assert client.post(f"/api/wakewords/{wid}/trainings", json={"mode": "quality"}).status_code == 422
    assert client.get("/api/wakewords/nope").status_code == 404


def test_full_flow(client, app, settings, speech, noise):
    ww = client.post("/api/wakewords", json={"name": "abad", "language": "fr"})
    assert ww.status_code == 201
    wid = ww.json()["id"]

    r = upload(client, wid, "positive", wav(take(speech)))
    assert r.status_code == 201, r.text
    samples = r.json()["samples"]
    assert len(samples) == 6 and all(s["accepted"] for s in samples)
    assert all(s["similarity"] is not None for s in samples)
    assert upload(client, wid, "other_speech", wav(speech)).status_code == 201
    assert upload(client, wid, "ambience", wav(noise)).status_code == 201
    clip = client.get(f"/api/wakewords/{wid}/samples/{samples[0]['id']}/audio")
    assert clip.status_code == 200 and clip.content[:4] == b"RIFF"

    # drop one recording: 5 remain, an odd count for the leave-2-out folds
    assert client.delete(f"/api/wakewords/{wid}/samples/{samples[-1]['id']}").status_code == 204
    r = client.post(f"/api/wakewords/{wid}/trainings", json={"target_fa_per_hour": 2.0})
    assert r.status_code == 202, r.text
    tid = r.json()["id"]
    assert r.json()["status"] == "queued" and r.json()["base_version"] == "tiny"
    assert client.post(f"/api/wakewords/{wid}/trainings", json={}).status_code == 409
    assert client.get(f"/api/trainings/{tid}/artifacts/model.onnx").status_code == 404

    worker = Worker(settings, engine=app.state.engine)
    assert worker.run_once() and not worker.run_once()

    t = client.get(f"/api/trainings/{tid}").json()
    assert t["status"] == "done", t["error"]
    assert set(t["artifacts"]) == set(ARTIFACTS)
    report = t["report"]
    thresholds = {k: report["calibration"][k]["threshold"] for k in ("fp32", "int8")}
    assert report["calibration"]["int8"]["own_cache"]
    assert report["recordings"] == 5
    assert report["negative_seconds"]["other_speech"] > 10
    assert report["cross_validation"]["folds"] == 3
    assert len(report["cross_validation"]["per_recording"]) == 5
    for kind in ("fp32", "int8"):
        v = report["verification"][kind]
        assert v["hours"] > 0 and v["enrollment_total"] == 5
        assert v["gate"]["false_accepts"] <= v["no_gate"]["false_accepts"]
    assert report["verification"]["parity"]["fp32_vs_torch_max"] < 1e-4

    events = sse(client, tid)
    stages = [d["stage"] for e, d in events if e == "progress"]
    assert stages[0] == "queued" and stages[-1] == "done"
    assert {"embedding", "cross_validating", "calibrating", "composing", "verifying"} <= set(stages)
    assert events[-1][0] == "done" and events[-1][1]["status"] == "done"

    files = {}
    for name in ("model.onnx", "model.int8.onnx", "vad.onnx", "model.json", "bundle.zip"):
        r = client.get(f"/api/trainings/{tid}/artifacts/{name}")
        assert r.status_code == 200
        files[name] = r.content
    path = settings.data_dir / "check.onnx"
    for kind, name in (("fp32", "model.onnx"), ("int8", "model.int8.onnx")):
        path.write_bytes(files[name])
        check_graph(path, outputs=["score"])
        meta = read_metadata(path)
        assert float(meta["threshold"]) == thresholds[kind]
        assert meta["wake_word"] == "abad" and meta["language"] == "fr"
        assert meta["base_model_version"] == "tiny" and meta["vad_file"] == "vad.onnx"
        det = Detector(files[name], files["vad.onnx"])
        assert det.config.threshold == thresholds[kind]
        det.process(speech)
    model_json = json.loads(files["model.json"])
    assert model_json["threshold"] == str(thresholds["fp32"])
    assert model_json["files"]["model.int8.onnx"]["threshold"] == str(thresholds["int8"])
    assert set(model_json["files"]) == {"model.onnx", "model.int8.onnx", "vad.onnx"}
    with zipfile.ZipFile(io.BytesIO(files["bundle.zip"])) as zf:
        assert sorted(zf.namelist()) == sorted(f"wake-word/{n}" for n in BUNDLE_FILES)
        assert zf.read("wake-word/model.onnx") == files["model.onnx"]

    # another operating point: same files, only the thresholds in their metadata change
    points = report["operating_points"]
    assert [p["fa_per_hour"] for p in points] == [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    for kind in ("fp32", "int8"):
        thr = [p[kind]["threshold"] for p in points]
        assert thr == sorted(thr, reverse=True)  # more false accepts allowed, lower threshold
        assert next(p for p in points if p["fa_per_hour"] == 2.0)[kind]["threshold"] == thresholds[kind]
    r = client.get(f"/api/trainings/{tid}/bundle.zip", params={"fa_per_hour": 0.1})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for kind, name in (("fp32", "model.onnx"), ("int8", "model.int8.onnx")):
            path.write_bytes(zf.read(f"wake-word/{name}"))
            meta = read_metadata(path)
            assert float(meta["threshold"]) == points[0][kind]["threshold"]
            assert {k: v for k, v in meta.items() if k != "threshold"} == {
                k: v for k, v in read_metadata_bytes(files[name]).items() if k != "threshold"}
        retargeted = json.loads(zf.read("wake-word/model.json"))
        assert retargeted["threshold"] == str(points[0]["fp32"]["threshold"])
        assert retargeted["evaluation"]["target_fa_per_hour"] == 0.1
        assert "0.1 false accepts per hour" in zf.read("wake-word/README.md").decode()
    assert client.get(f"/api/trainings/{tid}/bundle.zip?fa_per_hour=0.3").status_code == 422

    assert client.delete(f"/api/wakewords/{wid}").status_code == 204
    assert client.get(f"/api/wakewords/{wid}").status_code == 404
    assert client.get(f"/api/trainings/{tid}").status_code == 404
    assert not (settings.files_dir / "wakewords" / wid).exists()


def test_deleting_a_wake_word_cancels_its_training(client, app, settings, speech):
    wid = client.post("/api/wakewords", json={"name": "x", "language": "en"}).json()["id"]
    upload(client, wid, "positive", wav(take(speech, 3)))
    tid = client.post(f"/api/wakewords/{wid}/trainings", json={}).json()["id"]
    worker = Worker(settings, engine=app.state.engine)
    assert worker.claim() == tid
    client.delete(f"/api/wakewords/{wid}")
    assert worker.cancelled(tid)
    worker.execute(tid)
    assert not (settings.files_dir / "wakewords" / wid).exists()


def test_worker_requeues_interrupted_jobs(client, app, settings, speech):
    wid = client.post("/api/wakewords", json={"name": "x", "language": "en"}).json()["id"]
    upload(client, wid, "positive", wav(take(speech, 3)))
    tid = client.post(f"/api/wakewords/{wid}/trainings", json={}).json()["id"]
    worker = Worker(settings, engine=app.state.engine)
    assert worker.claim() == tid
    assert client.get(f"/api/trainings/{tid}").json()["status"] == "running"
    Worker(settings, engine=app.state.engine).recover()  # a new worker after a crash
    assert client.get(f"/api/trainings/{tid}").json()["status"] == "queued"


def test_release_without_int8_cache_shares_the_fp32_threshold(models_dir, tmp_path, speech):
    import shutil

    from server.settings import Settings

    old = tmp_path / "models"
    shutil.copytree(models_dir, old)
    (old / "base" / "tiny" / "longform_cache.int8.npz").unlink()
    manifest = json.loads((old / "base" / "tiny" / "base_model.json").read_text())
    del manifest["files"]["longform_cache.int8.npz"]
    (old / "base" / "tiny" / "base_model.json").write_text(json.dumps(manifest))
    settings = Settings(data_dir=tmp_path / "server", models_dir=old, verify_workers=2,
                        web_dist=None, finetune={"augment_per_clip": 4, "partial_per_clip": 2,
                                  "hard_negatives": 50, "random_negatives": 200})
    app = create_app(settings)
    client = TestClient(app)
    wid = client.post("/api/wakewords", json={"name": "x", "language": "en"}).json()["id"]
    upload(client, wid, "positive", wav(take(speech, 3)))
    tid = client.post(f"/api/wakewords/{wid}/trainings", json={}).json()["id"]
    Worker(settings, engine=app.state.engine).run_once()
    cal = client.get(f"/api/trainings/{tid}").json()["report"]["calibration"]
    assert not cal["int8"]["own_cache"]
    assert cal["int8"]["threshold"] == cal["fp32"]["threshold"]


def test_serves_the_built_app_next_to_the_api(settings, tmp_path):
    from dataclasses import replace

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>app</title>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    client = TestClient(create_app(replace(settings, web_dist=dist)))
    for path in ("/", "/w/abc", "/w/abc/t/def"):  # client-side routes get the app
        r = client.get(path)
        assert r.status_code == 200 and "<title>app</title>" in r.text, path
    assert client.get("/assets/app.js").text == "console.log(1)"
    assert client.get("/assets/missing.js").status_code == 404
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/nope").status_code == 404


def test_worker_thread_runs_jobs_inside_the_api_process(settings, speech):
    import time
    from dataclasses import replace

    with TestClient(create_app(replace(settings, worker_thread=True))) as client:
        wid = client.post("/api/wakewords", json={"name": "x", "language": "en"}).json()["id"]
        upload(client, wid, "positive", wav(take(speech, 3)))
        tid = client.post(f"/api/wakewords/{wid}/trainings", json={}).json()["id"]
        deadline = time.time() + 120
        while client.get(f"/api/trainings/{tid}").json()["status"] not in ("done", "failed"):
            assert time.time() < deadline
            time.sleep(0.2)
        assert client.get(f"/api/trainings/{tid}").json()["status"] == "done"
