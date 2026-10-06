"""Base models as `training.export` writes them (`models/base/<version>/`), loaded lazily and
once per process. The server only reads them: it never imports `training`."""

from __future__ import annotations

import hashlib
import json
from functools import cached_property
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from core.augment import AugmentConfig, Augmenter, NoiseBank, RirBank
from core.calibrate import LongformCache
from core.encoder import Encoder, load_encoder
from core.finetune import Bank
from core.verify import parallel, stream_vad
from server.settings import Settings

RUNTIME_ASSETS = Path(__file__).resolve().parents[1] / "runtime" / "assets"
VAD_FILE = RUNTIME_ASSETS / "silero_vad_16k_op15.onnx"
VAD_LICENSE = RUNTIME_ASSETS / "LICENSE-silero-vad"
SHIPPED = ("encoder.onnx", "encoder.int8.onnx", "longform_cache.int8.npz")  # checked by hash


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


class BaseModel:
    def __init__(self, path: Path, device: str = "cpu") -> None:
        self.path, self.device = path, device
        self.version = path.name
        self.manifest = json.loads((path / "base_model.json").read_text())
        self._checked = False

    def summary(self) -> dict:
        evaluation = self.manifest.get("evaluation", {})
        return {
            "version": self.version,
            "created_at": self.manifest.get("created_at"),
            "encoder_config": self.manifest.get("encoder_config"),
            "longform_hours": self.manifest.get("longform_hours"),
            "evaluation": {
                "fewshot": evaluation.get("fewshot_heldout_words", {}).get("mean"),
                "e2e": evaluation.get("e2e", {}).get("mean"),
            },
        }

    def check_shipped(self) -> None:
        """The encoders go into every user's bundle and the int8 cache sets its threshold:
        refuse if they changed on disk since the release. Checked once per process."""
        if self._checked:
            return
        for name in SHIPPED:
            expected = self.manifest["files"].get(name)
            if expected and (self.path / name).exists() and sha256(self.path / name) != expected:
                raise RuntimeError(f"{self.path / name} doesn't match base_model.json")
        self._checked = True

    @cached_property
    def encoder(self) -> Encoder:
        return load_encoder(self.path / "encoder.pt", self.device)

    @cached_property
    def embedding_session(self) -> ort.InferenceSession:
        """encoder.onnx on plain onnxruntime, for upload checks in the API process."""
        return ort.InferenceSession(str(self.path / "encoder.onnx"),
                                    providers=["CPUExecutionProvider"])

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        return self.embedding_session.run(None, {"audio": windows.astype(np.float32)})[0]

    @cached_property
    def bank(self) -> Bank:
        return Bank.load(self.path / "neg_bank.npz")

    @cached_property
    def cache(self) -> LongformCache:
        return LongformCache.load(self.path / "longform_cache.npz")

    @cached_property
    def cache_embeddings(self) -> torch.Tensor:
        return torch.from_numpy(self.cache.embeddings.astype(np.float32))

    @cached_property
    def cache_int8_embeddings(self) -> torch.Tensor | None:
        """The calibration streams through encoder.int8.onnx, when the release has them
        (base models exported before M2 share the fp32 threshold)."""
        path = self.path / "longform_cache.int8.npz"
        if not path.exists():
            return None
        cache = LongformCache.load(path)
        if not np.array_equal(cache.stream_starts, self.cache.stream_starts):
            raise RuntimeError(f"{path} doesn't match longform_cache.npz")
        return torch.from_numpy(cache.embeddings.astype(np.float32))

    @cached_property
    def augmenter(self) -> Augmenter:
        z = np.load(self.path / "augment_assets.npz")
        return Augmenter(AugmentConfig(), NoiseBank(z["noise"], self.device),
                         NoiseBank(z["babble"], self.device),
                         RirBank(z["rirs"].astype(np.float32), self.device))

    @cached_property
    def verify_streams(self) -> list[np.ndarray]:
        z = np.load(self.path / "longform_verify.npz")
        return [z[k] for k in sorted(z.files)]

    def verify_vad(self, cache_dir: Path, workers: int) -> list[np.ndarray]:
        """VAD probabilities of the verification streams: model-independent, so computed
        once per base model and cached on disk."""
        path = cache_dir / self.version / "verify_vad.npz"
        if path.exists():
            z = np.load(path)
            return [z[str(i)] for i in range(len(z.files))]
        probs = parallel(lambda a: stream_vad(VAD_FILE, a), self.verify_streams, workers)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name("verify_vad.tmp.npz")
        np.savez(tmp, **{str(i): p for i, p in enumerate(probs)})
        tmp.replace(path)
        return probs


class BaseModels:
    """The base models available under `<models>/base/`."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = settings.models_dir / "base"
        self._loaded: dict[str, BaseModel] = {}

    def versions(self) -> list[str]:
        found = [d for d in self.root.glob("*/base_model.json")] if self.root.exists() else []
        manifests = [(json.loads(f.read_text()).get("created_at", ""), f.parent.name) for f in found]
        return [v for _, v in sorted(manifests)]

    def default(self) -> str:
        if self.settings.base_version:
            return self.settings.base_version
        versions = self.versions()
        if not versions:
            raise LookupError(f"no base model in {self.root}")
        return versions[-1]

    def get(self, version: str | None = None) -> BaseModel:
        version = version or self.default()
        if version not in self._loaded:
            path = self.root / version
            if not (path / "base_model.json").exists():
                raise LookupError(f"unknown base model {version!r}")
            self._loaded[version] = BaseModel(path, self.settings.device)
        return self._loaded[version]
