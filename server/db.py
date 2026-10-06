"""Database tables (SQLModel on SQLite; Postgres later). Jobs are rows of `Training`."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, Column, event
from sqlalchemy.engine import Engine
from sqlmodel import Field, SQLModel, create_engine


def new_id() -> str:
    return uuid.uuid4().hex  # unguessable: there is no auth before M6


def now() -> datetime:
    return datetime.now(UTC)


class WakeWord(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    name: str
    language: str
    created_at: datetime = Field(default_factory=now)


class Upload(SQLModel, table=True):
    """An uploaded file, kept as received (PLAN.md §9.2 step 1)."""

    id: str = Field(default_factory=new_id, primary_key=True)
    wakeword_id: str = Field(index=True)
    kind: str
    filename: str
    key: str
    duration_s: float
    created_at: datetime = Field(default_factory=now)


class Sample(SQLModel, table=True):
    """One processed clip: a wake-word utterance cut from an upload, or a whole
    other-speech / ambience recording. Rejected clips are kept so the UI can say why."""

    id: str = Field(default_factory=new_id, primary_key=True)
    wakeword_id: str = Field(index=True)
    upload_id: str = Field(index=True)
    kind: str
    key: str
    start_s: float
    duration_s: float
    speech_s: float
    peak_dbfs: float
    rms_dbfs: float
    accepted: bool
    reason: str | None = None
    similarity: float | None = None  # positives: cosine to the other recordings' prototype
    outlier: bool = False
    created_at: datetime = Field(default_factory=now)


class Training(SQLModel, table=True):
    id: str = Field(default_factory=new_id, primary_key=True)
    wakeword_id: str = Field(index=True)
    mode: str
    target_fa_per_hour: float
    base_version: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    stage: str = "queued"
    progress: float = 0.0
    error: str | None = None
    sample_ids: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    report: dict | None = Field(default=None, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class TrainingEvent(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    training_id: str = Field(index=True)
    created_at: datetime = Field(default_factory=now)
    stage: str
    progress: float
    message: str
    data: dict | None = Field(default=None, sa_column=Column(JSON))


def make_engine(url: str) -> Engine:
    engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _record) -> None:  # the API and the worker share the file
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=30000")
        cur.close()

    SQLModel.metadata.create_all(engine)
    return engine
