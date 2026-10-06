"""Fine-tune worker: takes queued trainings from the database, one at a time (PLAN.md §9).

    uv run --extra server python -m server.worker [--once]

Progress goes to `TrainingEvent` rows, which the API streams as server-sent events.
"""

from __future__ import annotations

import argparse
import io
import logging
import time
import traceback

import numpy as np
import soundfile as sf
from sqlalchemy import delete, update
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from server import quick
from server.basemodel import BaseModels
from server.db import Sample, Training, TrainingEvent, WakeWord, make_engine, now
from server.settings import Settings
from server.storage import LocalStorage

log = logging.getLogger("wwd.worker")
FINAL = ("done", "failed", "cancelled")


def training_prefix(wakeword_id: str, training_id: str) -> str:
    return f"wakewords/{wakeword_id}/trainings/{training_id}"


class Worker:
    def __init__(self, settings: Settings, engine: Engine | None = None) -> None:
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.engine = engine or make_engine(settings.db_url)
        self.storage = LocalStorage(settings.files_dir)
        self.bases = BaseModels(settings)

    def event(self, tid: str, stage: str, progress: float, message: str,
              data: dict | None = None, **fields) -> None:
        with Session(self.engine) as s:
            s.add(TrainingEvent(training_id=tid, stage=stage, progress=progress, message=message,
                                data=data))
            s.execute(update(Training).where(Training.id == tid)
                      .values(stage=stage, progress=progress, **fields))
            s.commit()

    def recover(self) -> None:
        """A job left running was interrupted (the worker died): queue it again."""
        with Session(self.engine) as s:
            stuck = s.exec(select(Training).where(Training.status == "running")).all()
        for t in stuck:
            self.event(t.id, "queued", 0.0, "Requeued after a worker restart", status="queued")

    def claim(self) -> str | None:
        with Session(self.engine) as s:
            queued = s.exec(select(Training.id).where(Training.status == "queued")
                            .order_by(Training.created_at)).all()
            for tid in queued:
                res = s.execute(update(Training)
                                .where(Training.id == tid, Training.status == "queued")
                                .values(status="running", started_at=now()))
                s.commit()
                if res.rowcount == 1:
                    return tid
        return None

    def cancelled(self, tid: str) -> bool:
        with Session(self.engine) as s:
            t = s.get(Training, tid)
            return t is None or t.status == "cancelled" or s.get(WakeWord, t.wakeword_id) is None

    def load_job(self, t: Training, ww: WakeWord) -> quick.Job:
        with Session(self.engine) as s:
            samples = [s.get(Sample, sid) for sid in t.sample_ids]
        samples = [x for x in samples if x is not None and self.storage.exists(x.key)]

        def audio(x: Sample) -> np.ndarray:
            data, rate = sf.read(io.BytesIO(self.storage.read(x.key)), dtype="float32")
            assert rate == 16000
            return data

        positives = [x for x in samples if x.kind == "positive"]
        negatives = [x for x in samples if x.kind != "positive"]
        seconds: dict[str, float] = {}
        for x in negatives:
            seconds[x.kind] = round(seconds.get(x.kind, 0.0) + x.duration_s, 2)
        return quick.Job(ww.name, ww.language, t.target_fa_per_hour,
                         [audio(x) for x in positives], [x.id for x in positives],
                         [audio(x) for x in negatives], seconds)

    def execute(self, tid: str) -> None:
        with Session(self.engine) as s:
            t = s.get(Training, tid)
            ww = s.get(WakeWord, t.wakeword_id) if t else None
        if t is None or ww is None:
            return
        prefix = training_prefix(ww.id, tid)
        out = self.storage.path(prefix)
        log.info("training %s: %r (%s), base %s", tid, ww.name, ww.language, t.base_version)
        try:
            job = self.load_job(t, ww)
            self.event(tid, "loading", 0.01,
                       f"Loading base model {t.base_version} ({len(job.positives)} recordings)")
            base = self.bases.get(t.base_version)
            report = quick.run(base, job, out, self.settings,
                               lambda stage, p, msg: self.event(tid, stage, p, msg),
                               lambda: self.cancelled(tid))
        except quick.Cancelled:
            log.info("training %s cancelled", tid)
            self.storage.delete_prefix(prefix)
            self.event(tid, "cancelled", 0.0, "Cancelled", status="cancelled", finished_at=now())
        except Exception as e:  # noqa: BLE001 - any failure ends this job, not the worker
            log.error("training %s failed:\n%s", tid, traceback.format_exc())
            self.event(tid, "failed", 0.0, f"{type(e).__name__}: {e}", status="failed",
                       error=f"{type(e).__name__}: {e}", finished_at=now())
        else:
            summary = {"threshold": report["threshold"],
                       "recall": (report["cross_validation"] or {}).get("recall_noisy"),
                       "fa_per_hour": report["verification"]["fp32"]["no_gate"]["per_hour"]}
            self.event(tid, "done", 1.0, "Done", data=summary, status="done", report=report,
                       finished_at=now())
            log.info("training %s done in %.0f s", tid, report["timings_s"]["total"])
        if self.cancelled(tid):  # deleted while running: leave nothing behind
            self.storage.delete_prefix(prefix)
            with Session(self.engine) as s:
                if s.get(Training, tid) is None:
                    s.execute(delete(TrainingEvent).where(TrainingEvent.training_id == tid))
                    s.commit()

    def run_once(self) -> bool:
        tid = self.claim()
        if tid is None:
            return False
        self.execute(tid)
        return True

    def run_forever(self, poll_s: float = 1.0) -> None:
        self.recover()
        log.info("worker ready: data %s, models %s", self.settings.data_dir,
                 self.settings.models_dir)
        while True:
            if not self.run_once():
                time.sleep(poll_s)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--once", action="store_true", help="run queued jobs, then exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    worker = Worker(Settings.from_env())
    if args.once:
        worker.recover()
        while worker.run_once():
            pass
    else:
        worker.run_forever()


if __name__ == "__main__":
    main()
