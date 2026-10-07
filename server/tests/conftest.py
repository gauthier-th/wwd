"""Shared fixtures: settings over a tiny base model (server/tests/tiny_base.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from runtime.wav import read_wav
from server.settings import Settings
from server.tests.tiny_base import AUDIO, SR, build_tiny_base


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("models")
    build_tiny_base(root / "base" / "tiny")
    return root


@pytest.fixture
def settings(models_dir, tmp_path) -> Settings:
    return Settings(data_dir=tmp_path / "server", models_dir=models_dir, verify_workers=2,
                    web_dist=None,
                    finetune={"augment_per_clip": 8, "partial_per_clip": 4, "hard_negatives": 50,
                              "random_negatives": 200})


@pytest.fixture(scope="session")
def speech() -> np.ndarray:
    audio, rate = read_wav(AUDIO / "speech.wav")
    assert rate == SR
    return audio


@pytest.fixture(scope="session")
def noise() -> np.ndarray:
    return read_wav(AUDIO / "noise.wav")[0]
