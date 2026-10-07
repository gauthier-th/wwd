"""Server configuration from environment variables; the defaults suit local development."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    data_dir: Path = REPO / "data" / "server"  # WWD_SERVER_DATA: SQLite database + files
    models_dir: Path = REPO / "models"  # WWD_MODELS: base models live in <models>/base/<version>/
    base_version: str | None = None  # WWD_BASE_VERSION: default base model (newest if unset)
    device: str = "cpu"  # WWD_DEVICE: torch device for fine-tuning (PLAN.md §9.3: CPU)
    verify_workers: int = os.cpu_count() or 1  # WWD_VERIFY_WORKERS: parallel runtime sessions
    max_upload_mb: float = 25.0
    max_upload_s: float = 180.0
    # FinetuneConfig overrides, as JSON in WWD_FINETUNE (tests use small ones).
    finetune: dict = field(default_factory=dict)
    # The built web app, served at / when present (WWD_WEB_DIST; empty to disable): one
    # origin for the app and the API, so a phone needs a single HTTPS tunnel.
    web_dist: Path | None = REPO / "web" / "dist"
    # Run the fine-tune worker in a thread of the API process (WWD_WORKER_THREAD=1): handy for
    # development and end-to-end tests; production runs `python -m server.worker` separately.
    worker_thread: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        kwargs: dict = {}
        if "WWD_SERVER_DATA" in env:
            kwargs["data_dir"] = Path(env["WWD_SERVER_DATA"])
        if "WWD_MODELS" in env:
            kwargs["models_dir"] = Path(env["WWD_MODELS"])
        if env.get("WWD_BASE_VERSION"):
            kwargs["base_version"] = env["WWD_BASE_VERSION"]
        if "WWD_DEVICE" in env:
            kwargs["device"] = env["WWD_DEVICE"]
        if "WWD_VERIFY_WORKERS" in env:
            kwargs["verify_workers"] = int(env["WWD_VERIFY_WORKERS"])
        if "WWD_FINETUNE" in env:
            kwargs["finetune"] = json.loads(env["WWD_FINETUNE"])
        if "WWD_WEB_DIST" in env:
            kwargs["web_dist"] = Path(env["WWD_WEB_DIST"]) if env["WWD_WEB_DIST"] else None
        if env.get("WWD_WORKER_THREAD", "") not in ("", "0"):
            kwargs["worker_thread"] = True
        return cls(**kwargs)

    @property
    def db_url(self) -> str:
        return f"sqlite:///{self.data_dir / 'wwd.sqlite'}"

    @property
    def files_dir(self) -> Path:
        return self.data_dir / "files"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"
