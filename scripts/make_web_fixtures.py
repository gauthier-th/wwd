"""Fixtures for the browser-only app's tests, under web/.e2e/ (gitignored):

    uv run --no-sync python -m scripts.make_web_fixtures

  models/base/tiny/   the tiny base model (server/tests/tiny_base.py)
  fixtures/           take.wav, other_speech.wav, ambience.wav
  public/base/        the tiny base exported as web assets (scripts/export_web_assets.py), with
                      small fine-tuning settings so a whole training takes seconds
  golden.json         Python's own results for the code web/src/local/ ports: trigger counts,
                      calibration, stream recall, quantiles, folds, word normalization, speed
                      perturbation, the pushed-prototype head, and a composed model's scores
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from core.calibrate import calibrate, count_triggers, stream_recall
from core.export import compose_head, read_metadata
from core.finetune import FinetuneConfig, _speed, normalize_word, prototype_head
from core.model import PrototypeHead
from runtime.spec import DetectorConfig
from runtime.vad import make_session
from scripts import export_web_assets
from server.quick import folds_of
from server.tests import tiny_base

REPO = Path(__file__).resolve().parents[1]
E2E = REPO / "web" / ".e2e"
# Fine-tuning settings for the tiny base (the server's e2e API uses the same, e2e/api-server.sh).
TINY_FINETUNE = {"augment_per_clip": 16, "partial_per_clip": 4, "hard_negatives": 50,
                 "random_negatives": 200}


def test_window(i: int, n: int = 24000) -> np.ndarray:
    """Deterministic windows the TypeScript test rebuilds: a chirp-ish tone and a ramp."""
    t = np.arange(n, dtype=np.float64) / 16000
    return (0.1 * np.sin(2 * np.pi * (200 + 150 * i) * t) + 0.02 * (t - 0.75)).astype(np.float32)


def golden(base: Path) -> dict:
    rng = np.random.default_rng(0)
    det = DetectorConfig()
    starts = [0, 400, 1000, 1300]
    scores = (rng.random(1300) * 0.3).astype(np.float32)
    for p in rng.choice(1300 - 4, 40, replace=False):
        scores[p : p + rng.integers(1, 5)] = rng.uniform(0.4, 1.0)
    hours = 1300 * det.hop_samples / det.sample_rate / 3600
    cache = SimpleNamespace(stream_starts=np.array(starts), hours=hours)
    ignore = np.zeros(1300, bool)
    ignore[100:300] = True
    shifted = rng.random((40, 6)).astype(np.float32)
    values = rng.normal(size=101)

    cfg = replace(FinetuneConfig(), hard_negatives=30)
    dim = 8
    proto = rng.normal(size=dim).astype(np.float32)
    proto /= np.linalg.norm(proto)
    pos = (proto + 0.3 * rng.normal(size=(20, dim))).astype(np.float32)
    neg = rng.normal(size=(300, dim)).astype(np.float32)
    head = prototype_head(proto, pos, neg, cfg)

    clip = rng.normal(size=1000).astype(np.float32)
    encoder = base / "encoder.onnx"
    direction = rng.normal(size=32).astype(np.float32)
    direction /= np.linalg.norm(direction)
    meta = {"threshold": "0.25", "wake_word": "abad", "language": "fr"}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.onnx"
        compose_head(encoder, PrototypeHead(torch.from_numpy(direction), 20.0, -7.3), path, metadata=meta)
        session = make_session(path.read_bytes())
        windows = np.stack([test_window(i) for i in range(3)])
        composed = session.run(None, {session.get_inputs()[0].name: windows})[0]
        assert read_metadata(path) == meta

    return {
        "triggers": {"scores": scores.tolist(), "starts": starts, "k": det.smoothing_k, "n": det.smoothing_n,
                     "refractory_ticks": det.refractory_ticks, "hours": hours,
                     "counts": {str(t): count_triggers(scores, np.array(starts), t, det)
                                for t in (0.35, 0.5, 0.7, 0.9)},
                     "calibrate": {str(fa): list(calibrate(scores, cache, fa, det)) for fa in (30, 100, 300)},
                     "calibrate_ignore": list(calibrate(scores, cache, 100, det, ignore=ignore)),
                     "ignore": [100, 300]},
        "recall": {"scores": shifted.tolist(), "results": {str(t): stream_recall(shifted, t, det)
                                                           for t in (0.3, 0.6, 0.8)}},
        "quantile": {"values": values.tolist(),
                     "results": {str(q): float(np.quantile(values, q)) for q in (0, 0.25, 0.5, 0.999, 1)}},
        "folds": {str(n): folds_of(n) for n in range(1, 11)},
        "normalize": {w: normalize_word(w) for w in ("Straße", "  HÉllo ", "Morceaux", "İstanbul", "ریزد")},
        "speed": {"clip": clip.tolist(),
                  "results": {str(f): _speed(torch.from_numpy(clip), f).numpy().tolist() for f in (0.9, 1.1)}},
        "head": {"proto": proto.tolist(), "pos": pos.tolist(), "neg": neg.tolist(),
                 "hard_negatives": cfg.hard_negatives, "rocchio": cfg.rocchio, "score_scale": cfg.score_scale,
                 "weight": head.linear.weight.detach().numpy()[0].tolist(),
                 "bias": float(head.linear.bias.item())},
        "compose": {"direction": direction.tolist(), "scale": 20.0, "bias": -7.3, "metadata": meta,
                    "scores": composed.tolist()},
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out", type=Path, default=E2E)
    args = p.parse_args(argv)
    tiny_base.main([str(args.out / "models"), "--fixtures", str(args.out / "fixtures")])
    base = args.out / "models" / "base" / "tiny"
    export_web_assets.main([str(base), "--out", str(args.out / "public" / "base"), "--workers", "4",
                            "--finetune", json.dumps(TINY_FINETUNE)])
    (args.out / "golden.json").write_text(json.dumps(golden(base)) + "\n")
    print(f"web fixtures -> {args.out}")


if __name__ == "__main__":
    main()
