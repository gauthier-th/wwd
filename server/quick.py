"""Quick mode (PLAN.md §9.3): frozen encoder, pushed-prototype head, threshold calibrated on
long-form audio, ONNX composed from the validated encoder files, then verified by running the
shipped files through the Python runtime."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from core.calibrate import calibrate, stream_recall
from core.export import compose_head
from core.finetune import (
    HOP,
    SHIFTS,
    WINDOW,
    Enrollment,
    EnrollmentEmbeddings,
    FinetuneConfig,
    embed,
    embed_enrollment,
    fit_head,
    tick_windows,
)
from core.verify import as_float, false_accepts, parallel
from runtime.detector import Detector
from runtime.spec import FORMAT_VERSION, DetectorConfig
from server.basemodel import VAD_FILE, BaseModel
from server.bundle import write_bundle
from server.settings import Settings

RECALL_COPIES = 10  # augmented copies of each held-out recording in the fold recall estimate
PARITY_TICKS = 128
MODEL_FILES = {"fp32": "model.onnx", "int8": "model.int8.onnx"}

Progress = Callable[[str, float, str], None]


class Cancelled(Exception):
    pass


@dataclass
class Job:
    wake_word: str
    language: str
    target_fa_per_hour: float
    positives: list[np.ndarray]  # processed clips, 16 kHz float32
    positive_ids: list[str]
    negatives: list[np.ndarray] = field(default_factory=list)  # other speech + ambience
    negative_seconds: dict[str, float] = field(default_factory=dict)


def head_scores(head: torch.nn.Module, emb: np.ndarray | torch.Tensor) -> np.ndarray:
    x = emb if isinstance(emb, torch.Tensor) else torch.from_numpy(np.asarray(emb, np.float32))
    with torch.no_grad():
        return torch.sigmoid(head(x)).numpy()


def folds_of(n: int) -> list[list[int]]:
    """Leave-2-out (consecutive pairs; an odd one out is a fold of its own), or leave-1-out
    below 5 recordings, so every fold keeps at least 2 recordings to fit on."""
    size = 1 if n < 5 else 2
    return [list(range(i, min(i + size, n))) for i in range(0, n, size)]


def cross_validate(base: BaseModel, emb: EnrollmentEmbeddings, noisy: np.ndarray,
                   clean: np.ndarray, job: Job, ignore: np.ndarray, cfg: FinetuneConfig,
                   det: DetectorConfig) -> dict:
    """Recall on recordings the head didn't see (PLAN.md §9.3 step 4), each fold calibrated to
    the target like the final model. noisy: [C, copies, SHIFTS, D], clean: [C, 1, SHIFTS, D]."""
    n = len(emb.clean)
    recall_noisy, recall_clean = np.zeros(n), np.zeros(n)
    folds = folds_of(n)
    for held in folds:
        train = [i for i in range(n) if i not in held]
        res = fit_head(emb, base.bank, job.wake_word, job.language, cfg, recordings=train)
        thr, _ = calibrate(head_scores(res.head, base.cache_embeddings), base.cache,
                           job.target_fa_per_hour, det, ignore=ignore)
        for i in held:
            recall_noisy[i] = stream_recall(head_scores(res.head, noisy[i].reshape(-1, noisy.shape[-1]))
                                            .reshape(-1, SHIFTS), thr, det)
            recall_clean[i] = stream_recall(head_scores(res.head, clean[i].reshape(-1, clean.shape[-1]))
                                            .reshape(-1, SHIFTS), thr, det)
    weak = (recall_noisy < 0.5) & (recall_noisy < np.median(recall_noisy) - 0.25)
    return {
        "folds": len(folds),
        "recall_noisy": float(recall_noisy.mean()),
        "recall_clean": float(recall_clean.mean()),
        "per_recording": [
            {"sample_id": sid, "recall_noisy": float(rn), "recall_clean": float(rc), "weak": bool(w)}
            for sid, rn, rc, w in zip(job.positive_ids, recall_noisy, recall_clean, weak, strict=True)
        ],
    }


def fires_on(model: bytes, vad: bytes, clip: np.ndarray, seed: int) -> bool:
    """The full runtime path, VAD gate included, on one recording between pauses."""
    rng = np.random.default_rng(seed)
    stream = np.concatenate([np.zeros(16000, np.float32), clip, np.zeros(24000, np.float32)])
    stream = stream + rng.normal(0, 10 ** (-60 / 20), len(stream)).astype(np.float32)
    return bool(Detector(model, vad, gate=True).process(stream))


def parity(base: BaseModel, head: torch.nn.Module, scores: dict[str, list[np.ndarray]],
           seed: int = 0) -> dict:
    """Runtime scores of the shipped files against PyTorch on the same windows: the ticks the
    fp32 model scored highest (where thresholds bite) plus random ones."""
    streams = base.verify_streams
    flat = np.concatenate(scores["fp32"])
    offsets = np.cumsum([0] + [len(s) for s in scores["fp32"]])
    rng = np.random.default_rng(seed)
    top = np.argsort(-flat)[: PARITY_TICKS // 2]
    pick = np.unique(np.concatenate([top, rng.choice(len(flat), PARITY_TICKS // 2, replace=False)]))
    windows = np.zeros((len(pick), WINDOW), np.float32)
    for row, g in enumerate(pick):
        s = int(np.searchsorted(offsets, g, side="right") - 1)
        end = (g - offsets[s] + 1) * HOP  # runtime/SPEC.md §3: window ends at (tick + 1) * hop
        audio = as_float(streams[s][max(0, end - WINDOW) : end])
        windows[row, WINDOW - len(audio) :] = audio
    ref = head_scores(head, embed(base.encoder, torch.from_numpy(windows), None, 64, base.device))
    int8 = np.concatenate(scores["int8"])
    return {
        "ticks": len(pick),
        "fp32_vs_torch_max": float(np.abs(flat[pick] - ref).max()),
        "int8_vs_torch_max": float(np.abs(int8[pick] - ref).max()),
        "int8_vs_fp32_max": float(np.abs(int8 - flat).max()),
        "int8_vs_fp32_mean": float(np.abs(int8 - flat).mean()),
    }


def run(base: BaseModel, job: Job, out: Path, settings: Settings, progress: Progress,
        cancelled: Callable[[], bool] = lambda: False) -> dict:
    """Train, calibrate, compose and verify one wake word into `out`; returns the report."""
    timings: dict[str, float] = {}
    started = last = time.time()
    current = "loading"

    def step(stage: str, frac: float, message: str) -> None:
        nonlocal last, current
        if cancelled():
            raise Cancelled
        timings[current] = round(time.time() - last, 2)
        last, current = time.time(), stage
        progress(stage, frac, message)

    cfg = FinetuneConfig(**settings.finetune)
    det = DetectorConfig()
    device = settings.device
    n = len(job.positives)
    base.check_shipped()
    step("embedding", 0.05, f"Embedding {n} recordings x {cfg.augment_per_clip} augmentations")

    emb = embed_enrollment(base.encoder, Enrollment(job.wake_word, job.language, job.positives,
                                                    job.negatives), base.augmenter, cfg, device)
    torch.manual_seed(cfg.seed)
    dim = emb.clean.shape[1]
    noisy = embed(base.encoder, tick_windows(job.positives, RECALL_COPIES, base.augmenter, device),
                  None, cfg.embed_batch, device).reshape(n, RECALL_COPIES, SHIFTS, dim)
    clean = embed(base.encoder, tick_windows(job.positives, 1, None), None, cfg.embed_batch,
                  device).reshape(n, 1, SHIFTS, dim)
    step("cross_validating", 0.35, f"Leave-2-out folds over {n} recordings")

    ignore = base.cache.mask_word(job.wake_word)
    folds = cross_validate(base, emb, noisy, clean, job, ignore, cfg, det) if n >= 3 else None
    step("calibrating", 0.5, f"Calibrating to {job.target_fa_per_hour} false accepts/h")

    result = fit_head(emb, base.bank, job.wake_word, job.language, cfg)
    caches = {"fp32": base.cache_embeddings, "int8": base.cache_int8_embeddings}
    calibration: dict = {"hours": base.cache.hours, "target_fa_per_hour": job.target_fa_per_hour}
    for kind, cache_emb in caches.items():
        if cache_emb is None:  # a release without longform_cache.int8.npz: share fp32's
            calibration[kind] = {**calibration["fp32"], "own_cache": False}
            continue
        thr, fa = calibrate(head_scores(result.head, cache_emb), base.cache,
                            job.target_fa_per_hour, det, ignore=ignore)
        calibration[kind] = {"threshold": thr, "fa_per_hour": fa,
                             "target_met": fa <= job.target_fa_per_hour, "own_cache": True}
    step("composing", 0.55, "Composing ONNX files")

    created = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {kind: {
        "format_version": FORMAT_VERSION,
        **det.with_overrides(threshold=calibration[kind]["threshold"]).to_metadata(),
        "wake_word": job.wake_word,
        "language": job.language,
        "base_model_version": base.version,
        "created_at": created,
        "vad_file": "vad.onnx",
    } for kind in MODEL_FILES}
    out.mkdir(parents=True, exist_ok=True)
    encoders = {"fp32": "encoder.onnx", "int8": "encoder.int8.onnx"}
    for kind, name in MODEL_FILES.items():
        compose_head(base.path / encoders[kind], result.head, out / name, metadata=metadata[kind])
    step("verifying", 0.6, "Running the shipped files through the runtime")

    vad = base.verify_vad(settings.cache_dir, settings.verify_workers)
    vad_bytes = VAD_FILE.read_bytes()
    verify: dict = {}
    scores: dict[str, list[np.ndarray]] = {}
    for kind, name in MODEL_FILES.items():
        model = (out / name).read_bytes()
        run_cfg = det.with_overrides(threshold=calibration[kind]["threshold"])
        fa = false_accepts(model, base.verify_streams, vad, run_cfg, settings.verify_workers)
        scores[kind] = fa.pop("scores")
        hits = parallel(lambda a, m=model: fires_on(m, vad_bytes, a[1], a[0]),
                        list(enumerate(job.positives)), settings.verify_workers)
        verify[kind] = {**fa, "enrollment_detected": int(sum(hits)), "enrollment_total": n}
        if cancelled():
            raise Cancelled
    verify["parity"] = parity(base, result.head, scores)
    step("bundling", 0.97, "Writing the bundle")

    report = {
        "wake_word": job.wake_word,
        "language": job.language,
        "base_model_version": base.version,
        "mode": "quick",
        "recordings": n,
        "negative_seconds": job.negative_seconds,
        "threshold": calibration["fp32"]["threshold"],
        "target_fa_per_hour": job.target_fa_per_hour,
        "calibration": calibration,
        "cross_validation": folds,
        "verification": verify,
        "finetune": result.report,
        "timings_s": timings,
    }
    timings[current] = round(time.time() - last, 2)
    timings["total"] = round(time.time() - started, 2)
    write_bundle(out, metadata, report)
    return report
