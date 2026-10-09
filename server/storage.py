"""Blob storage behind a small interface: the local filesystem (PLAN.md §9).

Keys are relative POSIX paths such as `wakewords/<id>/samples/<sid>.wav`. Everything a user
owns lives under `wakewords/<id>/`, so deleting a wake word is one `delete_prefix`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path, PurePosixPath


class LocalStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        parts = PurePosixPath(key).parts
        if not parts or key.startswith("/") or any(p in ("..", ".") for p in parts):
            raise ValueError(f"bad storage key {key!r}")
        return self.root.joinpath(*parts)

    def write(self, key: str, data: bytes) -> None:
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def read(self, key: str) -> bytes:
        return self.path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def delete(self, key: str) -> None:
        self.path(key).unlink(missing_ok=True)

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self.path(prefix), ignore_errors=True)
