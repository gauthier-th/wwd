"""Device measurements (PLAN.md M4): benchmarks and field tests run by the web app on a phone.

A field test is uploaded a minute at a time, so a 24 h run survives the page being closed or
killed, into the same directory layout `runtime.listen --log` writes on a Pi
(runtime/fieldlog.py); `runtime.fieldreport` summarizes both.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Annotated, Literal

import numpy as np
import onnx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from runtime.fieldlog import FieldLog, load
from runtime.fieldreport import summarize
from runtime.spec import DetectorConfig
from server.db import DeviceRun, Training, now
from server.storage import LocalStorage
from server.worker import training_prefix

MODEL_FILES = {"fp32": "model.onnx", "int8": "model.int8.onnx"}
MAX_CHUNK_TICKS = 20_000  # a minute is 625


class FieldTestIn(BaseModel):
    model: Literal["fp32", "int8"] = "fp32"
    gate: bool = True
    fa_per_hour: float
    device: dict = Field(default_factory=dict)  # user agent, cores, memory, as the browser reports


class ChunkIn(BaseModel):
    stats: list[dict] = Field(default_factory=list, max_length=60)
    events: list[dict] = Field(default_factory=list, max_length=100)


def device_routes(app: FastAPI, session: Callable[[], Session], storage: LocalStorage) -> None:
    def done_training(s: Session, tid: str) -> Training:
        t = s.get(Training, tid)
        if t is None or t.status != "done":
            raise HTTPException(404, "no finished training with this id")
        return t

    def run_dir(run: DeviceRun):
        return storage.path(f"{training_prefix(run.wakeword_id, run.training_id)}/device/{run.id}")

    def get_run(s: Session, rid: str, kind: str) -> DeviceRun:
        run = s.get(DeviceRun, rid)
        if run is None or run.kind != kind:
            raise HTTPException(404, f"no {kind} with this id")
        return run

    @app.post("/api/trainings/{tid}/benchmarks", status_code=201)
    def save_benchmark(tid: str, body: dict) -> dict:
        if len(json.dumps(body)) > 1_000_000:
            raise HTTPException(413, "benchmark too large")
        with session() as s:
            t = done_training(s, tid)
            run = DeviceRun(wakeword_id=t.wakeword_id, training_id=tid, kind="benchmark", data=body,
                            finished=True)
            s.add(run)
            s.commit()
            return {"id": run.id}

    @app.post("/api/trainings/{tid}/fieldtests", status_code=201)
    def start_field_test(tid: str, body: FieldTestIn) -> dict:
        with session() as s:
            t = done_training(s, tid)
            wid, points = t.wakeword_id, t.report["operating_points"]
            point = next((p for p in points if abs(p["fa_per_hour"] - body.fa_per_hour) < 1e-9), None)
            if point is None:
                raise HTTPException(422, f"no operating point at {body.fa_per_hour} false accepts/h")
            run = DeviceRun(wakeword_id=wid, training_id=tid, kind="fieldtest")
            s.add(run)
            s.commit()
            s.refresh(run)
        name = MODEL_FILES[body.model]
        model = onnx.load(str(storage.path(f"{training_prefix(wid, tid)}/{name}")),
                          load_external_data=False)
        metadata = {p.key: p.value for p in model.metadata_props}
        config = DetectorConfig.from_metadata(metadata).with_overrides(threshold=point[body.model]["threshold"])
        d = body.device
        FieldLog(run_dir(run), {
            "model": name, "metadata": metadata, "config": config.to_metadata(), "gate": body.gate,
            "fa_per_hour": body.fa_per_hour, "input": "microphone (browser)",
            "host": {"machine": d.get("platform", "?"), "cpu": d.get("userAgent", "?"),
                     "cores": d.get("hardwareConcurrency"), "memory_gb": d.get("deviceMemory"),
                     "runtime": "onnxruntime-web (wasm)"},
        })
        return {"id": run.id, "threshold": config.threshold}

    @app.post("/api/fieldtests/{rid}/chunk")
    async def field_test_chunk(
        rid: str,
        data: Annotated[str, Form()],
        ticks: Annotated[UploadFile | None, File()] = None,
        clips: Annotated[list[UploadFile] | None, File()] = None,
    ) -> dict:
        """One minute (or so) of a running field test: stats, triggers with their clips (in
        order: each event naming a clip gets the next uploaded file), every tick's (score, VAD)."""
        try:
            chunk = ChunkIn.model_validate_json(data)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        with session() as s:
            run = get_run(s, rid, "fieldtest")
            if run.finished:
                raise HTTPException(409, "this field test is finished")
        log = FieldLog.reopen(run_dir(run))
        if ticks is not None:
            raw = await ticks.read(MAX_CHUNK_TICKS * 8 + 1)
            if len(raw) > MAX_CHUNK_TICKS * 8 or len(raw) % 8:
                raise HTTPException(422, "ticks: float32 (score, vad) pairs, at most 20000")
            pairs = np.frombuffer(raw, "<f4").reshape(-1, 2)
            log.ticks(pairs[:, 0], pairs[:, 1])
        files = list(clips or [])
        for event in chunk.events:
            clip = None
            if event.pop("clip", None) and files:
                clip = await files.pop(0).read(2_000_000)
            log.event({k: event[k] for k in ("time", "tick", "sample", "score") if k in event}, clip)
        for record in chunk.stats:
            log.stats(record)
        with session() as s:
            run = s.get(DeviceRun, rid)
            run.updated_at = now()
            s.add(run)
            s.commit()
        return {"events": log.events}

    @app.post("/api/fieldtests/{rid}/finish")
    def finish_field_test(rid: str) -> dict:
        with session() as s:
            run = get_run(s, rid, "fieldtest")
            if not run.finished:
                FieldLog.reopen(run_dir(run)).close()
                run.finished = True
                run.updated_at = now()
                s.add(run)
                s.commit()
                s.refresh(run)
        return summarize(load(run_dir(run)))

    @app.get("/api/fieldtests/{rid}/report")
    def field_test_report(rid: str, true: str = "") -> dict:
        with session() as s:
            run = get_run(s, rid, "fieldtest")
        return summarize(load(run_dir(run)), {int(x) for x in true.split(",") if x.strip().isdigit()})

    @app.get("/api/fieldtests/{rid}/clips/{name}")
    def field_test_clip(rid: str, name: str) -> FileResponse:
        if "/" in name or not name.endswith(".wav"):
            raise HTTPException(404, "no such clip")
        with session() as s:
            run = get_run(s, rid, "fieldtest")
        path = run_dir(run) / "clips" / name
        if not path.is_file():
            raise HTTPException(404, "no such clip")
        return FileResponse(path, media_type="audio/wav")

    @app.get("/api/trainings/{tid}/device-runs")
    def device_runs(tid: str) -> list[dict]:
        with session() as s:
            runs = s.exec(select(DeviceRun).where(DeviceRun.training_id == tid)
                          .order_by(DeviceRun.created_at)).all()
        out = []
        for run in runs:
            item = {"id": run.id, "kind": run.kind, "finished": run.finished,
                    "created_at": run.created_at, "updated_at": run.updated_at}
            if run.kind == "benchmark":
                item["data"] = run.data
            else:
                r = summarize(load(run_dir(run)))
                item["summary"] = {k: r[k] for k in ("hours", "triggers", "false_accepts_per_hour",
                                                      "busy_pct", "latency_ms", "battery", "host")}
            out.append(item)
        return out
