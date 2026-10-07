"""HTTP API (PLAN.md §9.1).

    uv run --extra server uvicorn server.api:app        # plus: python -m server.worker
"""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from fastapi.sse import EventSourceResponse, ServerSentEvent
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete
from sqlmodel import Session, col, select
from starlette.exceptions import HTTPException as StarletteHTTPException

from runtime.vad import SileroVad
from server import audio
from server.basemodel import VAD_FILE, BaseModels
from server.bundle import ARTIFACTS, retargeted_bundle
from server.db import DeviceRun, Sample, Training, TrainingEvent, Upload, WakeWord, make_engine
from server.device import device_routes
from server.languages import LANGUAGES
from server.settings import Settings
from server.storage import LocalStorage
from server.worker import FINAL, Worker, training_prefix

MIN_POSITIVES = 3
EXTENSIONS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/wave": "wav", "audio/flac": "flac",
              "audio/ogg": "ogg"}


class LanguageOut(BaseModel):
    code: str
    name: str
    native_name: str
    rtl: bool
    needs_review: bool
    paragraph: str


class WakeWordIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    language: str

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("empty name")
        return v

    @field_validator("language")
    @classmethod
    def _known(cls, v: str) -> str:
        if v not in LANGUAGES:
            raise ValueError(f"language must be one of {sorted(LANGUAGES)}")
        return v


class SampleOut(BaseModel):
    id: str
    upload_id: str
    kind: str
    start_s: float
    duration_s: float
    speech_s: float
    peak_dbfs: float
    rms_dbfs: float
    accepted: bool
    reason: str | None
    similarity: float | None
    outlier: bool
    created_at: datetime


class TrainingIn(BaseModel):
    mode: Literal["quick", "quality"] = "quick"
    target_fa_per_hour: float = Field(0.5, gt=0, le=10)
    base_version: str | None = None


class TrainingOut(BaseModel):
    id: str
    wakeword_id: str
    mode: str
    target_fa_per_hour: float
    base_version: str
    status: str
    stage: str
    progress: float
    error: str | None
    report: dict | None
    artifacts: list[str]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class WakeWordOut(BaseModel):
    id: str
    name: str
    language: str
    created_at: datetime
    samples: list[SampleOut]
    trainings: list[TrainingOut]


class UploadOut(BaseModel):
    upload_id: str
    samples: list[SampleOut]


def training_out(t: Training) -> TrainingOut:
    artifacts = list(ARTIFACTS) if t.status == "done" else []
    return TrainingOut(**t.model_dump(exclude={"sample_ids"}), artifacts=artifacts)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(settings.db_url)
    storage = LocalStorage(settings.files_dir)
    bases = BaseModels(settings)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stop = threading.Event()
        if settings.worker_thread:
            worker = Worker(settings, engine=engine)
            worker.recover()

            def loop() -> None:
                while not stop.is_set():
                    if not worker.run_once():
                        stop.wait(0.5)

            threading.Thread(target=loop, name="wwd-worker", daemon=True).start()
        yield
        stop.set()

    app = FastAPI(title="wwd", summary="Few-shot wake-word trainer (PLAN.md §9.1)",
                  lifespan=lifespan)
    app.state.settings, app.state.engine = settings, engine

    def session() -> Session:
        return Session(engine)

    def get_wakeword(s: Session, wid: str) -> WakeWord:
        ww = s.get(WakeWord, wid)
        if ww is None:
            raise HTTPException(404, "wake word not found")
        return ww

    def wakeword_out(s: Session, ww: WakeWord) -> WakeWordOut:
        samples = s.exec(select(Sample).where(Sample.wakeword_id == ww.id)
                         .order_by(Sample.created_at, Sample.start_s)).all()
        trainings = s.exec(select(Training).where(Training.wakeword_id == ww.id)
                           .order_by(Training.created_at)).all()
        return WakeWordOut(**ww.model_dump(), samples=[SampleOut(**x.model_dump()) for x in samples],
                           trainings=[training_out(t) for t in trainings])

    def refresh_outliers(s: Session, ww: WakeWord) -> None:
        """Similarity of each accepted recording to the others (PLAN.md §9.2 step 6)."""
        rows = s.exec(select(Sample).where(Sample.wakeword_id == ww.id, Sample.kind == "positive",
                                           Sample.accepted)).all()
        if not rows or not bases.versions():
            return
        windows = np.stack([audio.center_window(audio.decode(storage.read(r.key), 10.0))
                            for r in rows])
        sims, flags = audio.outliers(bases.get().embed_windows(windows))
        for r, sim, flag in zip(rows, sims, flags, strict=True):
            r.similarity, r.outlier = float(sim), bool(flag)
            s.add(r)

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok", "base_models": bases.versions()}

    @app.get("/api/languages")
    def languages() -> list[LanguageOut]:
        return [LanguageOut(code=code, **lang) for code, lang in LANGUAGES.items()]

    @app.get("/api/base-models")
    def base_models() -> list[dict]:
        default = bases.default() if bases.versions() else None
        return [bases.get(v).summary() | {"default": v == default} for v in bases.versions()]

    @app.post("/api/wakewords", status_code=201)
    def create_wakeword(body: WakeWordIn) -> WakeWordOut:
        with session() as s:
            ww = WakeWord(name=body.name, language=body.language)
            s.add(ww)
            s.commit()
            s.refresh(ww)
            return wakeword_out(s, ww)

    @app.get("/api/wakewords/{wid}")
    def get_wakeword_route(wid: str) -> WakeWordOut:
        with session() as s:
            return wakeword_out(s, get_wakeword(s, wid))

    @app.delete("/api/wakewords/{wid}", status_code=204)
    def delete_wakeword(wid: str) -> Response:
        """Deletes every recording, model and record of this wake word (GDPR). A running
        training stops at its next step and removes what it wrote."""
        with session() as s:
            get_wakeword(s, wid)
            tids = s.exec(select(Training.id).where(Training.wakeword_id == wid)).all()
            s.execute(delete(TrainingEvent).where(col(TrainingEvent.training_id).in_(tids)))
            for table in (DeviceRun, Training, Sample, Upload):
                s.execute(delete(table).where(table.wakeword_id == wid))
            s.execute(delete(WakeWord).where(WakeWord.id == wid))
            s.commit()
        storage.delete_prefix(f"wakewords/{wid}")
        return Response(status_code=204)

    @app.post("/api/wakewords/{wid}/samples", status_code=201)
    async def upload_sample(
        wid: str,
        file: Annotated[UploadFile, File()],
        kind: Annotated[Literal["positive", "other_speech", "ambience"], Form()],
    ) -> UploadOut:
        limit = int(settings.max_upload_mb * 1e6)
        data = await file.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"file larger than {settings.max_upload_mb:g} MB")
        return await asyncio.to_thread(store_upload, wid, kind, file.filename or "upload",
                                       file.content_type or "", data)

    def store_upload(wid: str, kind: str, filename: str, content_type: str, data: bytes) -> UploadOut:
        with session() as s:
            get_wakeword(s, wid)
        try:
            pcm = audio.decode(data, settings.max_upload_s)
            vad = SileroVad(VAD_FILE)
            clips = (audio.split_positives(pcm, vad) if kind == "positive"
                     else [audio.prepare_other(pcm, kind, vad)])
        except audio.UploadError as e:
            raise HTTPException(422, str(e)) from e
        ext = EXTENSIONS.get(content_type) or (re.findall(r"\.(\w{1,5})$", filename) or ["bin"])[0]
        with session() as s:
            ww = get_wakeword(s, wid)
            up = Upload(wakeword_id=wid, kind=kind, filename=filename[:200], key="",
                        duration_s=len(pcm) / audio.SR)
            up.key = f"wakewords/{wid}/uploads/{up.id}.{ext.lower()}"
            storage.write(up.key, data)
            s.add(up)
            rows = []
            for clip in clips:
                row = Sample(wakeword_id=wid, upload_id=up.id, kind=kind, key="",
                             start_s=clip.start_s, duration_s=clip.duration_s,
                             speech_s=clip.speech_s, peak_dbfs=clip.peak_dbfs,
                             rms_dbfs=clip.rms_dbfs, accepted=clip.accepted, reason=clip.reason)
                row.key = f"wakewords/{wid}/samples/{row.id}.wav"
                storage.write(row.key, audio.encode_wav(clip.audio))
                s.add(row)
                rows.append(row)
            s.flush()
            if kind == "positive":
                refresh_outliers(s, ww)
            s.commit()
            for r in rows:
                s.refresh(r)
            return UploadOut(upload_id=up.id, samples=[SampleOut(**r.model_dump()) for r in rows])

    @app.get("/api/wakewords/{wid}/samples/{sid}/audio")
    def sample_audio(wid: str, sid: str) -> FileResponse:
        with session() as s:
            row = s.get(Sample, sid)
            if row is None or row.wakeword_id != wid:
                raise HTTPException(404, "sample not found")
            return FileResponse(storage.path(row.key), media_type="audio/wav")

    @app.delete("/api/wakewords/{wid}/samples/{sid}", status_code=204)
    def delete_sample(wid: str, sid: str) -> Response:
        with session() as s:
            ww = get_wakeword(s, wid)
            row = s.get(Sample, sid)
            if row is None or row.wakeword_id != wid:
                raise HTTPException(404, "sample not found")
            s.delete(row)
            storage.delete(row.key)
            remaining = s.exec(select(Sample).where(Sample.upload_id == row.upload_id,
                                                    Sample.id != sid)).first()
            if remaining is None:  # the last clip of an upload takes the original with it
                up = s.get(Upload, row.upload_id)
                if up is not None:
                    storage.delete(up.key)
                    s.delete(up)
            s.flush()
            if row.kind == "positive":
                refresh_outliers(s, ww)
            s.commit()
        return Response(status_code=204)

    @app.post("/api/wakewords/{wid}/trainings", status_code=202)
    def create_training(wid: str, body: TrainingIn) -> TrainingOut:
        if body.mode == "quality":
            raise HTTPException(422, "quality mode arrives in M6; use mode=quick")
        try:
            version = body.base_version or bases.default()
            bases.get(version)
        except LookupError as e:
            raise HTTPException(422, str(e)) from e
        with session() as s:
            get_wakeword(s, wid)
            active = s.exec(select(Training).where(Training.wakeword_id == wid,
                                                   col(Training.status).in_(("queued", "running")))).first()
            if active is not None:
                raise HTTPException(409, f"training {active.id} is already {active.status}")
            samples = s.exec(select(Sample).where(Sample.wakeword_id == wid, Sample.accepted)).all()
            positives = [x for x in samples if x.kind == "positive" and not x.outlier]
            if len(positives) < MIN_POSITIVES:
                raise HTTPException(422, f"need at least {MIN_POSITIVES} accepted recordings of "
                                         f"the wake word (10 recommended), have {len(positives)}")
            ids = [x.id for x in positives] + [x.id for x in samples if x.kind != "positive"]
            t = Training(wakeword_id=wid, mode=body.mode, target_fa_per_hour=body.target_fa_per_hour,
                         base_version=version, sample_ids=ids)
            s.add(t)
            s.add(TrainingEvent(training_id=t.id, stage="queued", progress=0.0, message="Queued"))
            s.commit()
            s.refresh(t)
            return training_out(t)

    @app.get("/api/trainings/{tid}")
    def get_training(tid: str) -> TrainingOut:
        with session() as s:
            t = s.get(Training, tid)
            if t is None:
                raise HTTPException(404, "training not found")
            return training_out(t)

    @app.get("/api/trainings/{tid}/events", response_class=EventSourceResponse)
    async def training_events(
        tid: str, request: Request,
        last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    ) -> AsyncIterator[ServerSentEvent]:
        """Progress as server-sent events (`progress`), then one final `done`, `failed` or
        `cancelled` event carrying the training. Resumes after `Last-Event-ID`."""
        after = int(last_event_id) if last_event_id and last_event_id.isdigit() else 0
        while True:
            with session() as s:
                t = s.get(Training, tid)  # read before the events: none can be missed
                if t is None:
                    yield ServerSentEvent(event="gone", data={"id": tid})
                    return
                rows = s.exec(select(TrainingEvent).where(TrainingEvent.training_id == tid,
                                                          TrainingEvent.id > after)
                              .order_by(TrainingEvent.id)).all()
            for r in rows:
                after = r.id
                yield ServerSentEvent(id=str(r.id), event="progress", data={
                    "stage": r.stage, "progress": r.progress, "message": r.message,
                    "data": r.data, "at": r.created_at.isoformat()})
            if t.status in FINAL:
                yield ServerSentEvent(event=t.status, data=training_out(t).model_dump(mode="json"))
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.5)

    @app.get("/api/trainings/{tid}/artifacts/{name}")
    def artifact(tid: str, name: str) -> FileResponse:
        if name not in ARTIFACTS:
            raise HTTPException(404, f"artifacts: {', '.join(ARTIFACTS)}")
        with session() as s:
            t = s.get(Training, tid)
            if t is None or t.status != "done":
                raise HTTPException(404, "no finished training with this id")
            path = storage.path(f"{training_prefix(t.wakeword_id, tid)}/{name}")
        media = {"json": "application/json", "md": "text/markdown", "zip": "application/zip"}
        return FileResponse(path, filename=name,
                            media_type=media.get(name.rsplit(".", 1)[-1], "application/octet-stream"))

    @app.get("/api/trainings/{tid}/bundle.zip")
    def bundle_at(tid: str, fa_per_hour: float) -> Response:
        """The bundle with every model file's threshold set for another of the training's
        operating points (`report.operating_points`): only metadata changes."""
        with session() as s:
            t = s.get(Training, tid)
            if t is None or t.status != "done":
                raise HTTPException(404, "no finished training with this id")
        try:
            data = retargeted_bundle(storage.path(training_prefix(t.wakeword_id, tid)), t.report,
                                     fa_per_hour)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        name = f"wake-word-{fa_per_hour:g}fa.zip"
        return Response(data, media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    device_routes(app, session, storage)

    if settings.web_dist is not None and (settings.web_dist / "index.html").exists():
        app.mount("/", SpaFiles(directory=settings.web_dist, html=True), name="web")
    return app


class SpaFiles(StaticFiles):
    """The built app: files as they are, and index.html for client-side routes (/w/<id>...)."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            if e.status_code != 404 or path.startswith("api/") or "." in path.rsplit("/", 1)[-1]:
                raise
            return await super().get_response("index.html", scope)


def __getattr__(name: str):  # `uvicorn server.api:app` builds the app from the environment
    if name == "app":
        return create_app()
    raise AttributeError(name)
