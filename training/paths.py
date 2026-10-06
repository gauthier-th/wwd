"""Where training data and base models live. Override the data root with WWD_DATA."""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("WWD_DATA", REPO / "data"))
MODELS = Path(os.environ.get("WWD_MODELS", REPO / "models"))

MSWC = DATA / "mswc"
MUSAN = DATA / "musan"
RIRS = DATA / "RIRS_NOISES"
MLS = DATA / "mls"
LIBRISPEECH = DATA / "LibriSpeech"
DOWNLOADS = DATA / "_downloads"

# MSWC high-resourced languages (> 100 h), PLAN.md §6.1. Ordered smallest first.
LANGUAGES = ["cy", "eu", "pl", "ru", "it", "fa", "es", "rw", "ca", "fr", "de", "en"]
SPLITS = ["train", "dev", "test"]
