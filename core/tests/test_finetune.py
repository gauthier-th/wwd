from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
import torch

from core.augment import AugmentConfig, Augmenter, NoiseBank, RirBank
from core.calibrate import LongformCache, calibrate, count_triggers, stream_recall
from core.encoder import Encoder, EncoderConfig
from core.finetune import (
    WINDOW,
    Bank,
    Enrollment,
    FinetuneConfig,
    finetune,
    normalize_word,
    partial_windows,
    positive_windows,
    prototype_head,
)
from runtime.detector import Trigger
from runtime.spec import DetectorConfig


@pytest.mark.parametrize("refractory_ms", [0, 192, 1500])
def test_count_triggers_matches_runtime_trigger(refractory_ms):
    rng = np.random.default_rng(0)
    cfg = DetectorConfig(refractory_ms=refractory_ms)
    scores = rng.random(5000) ** 3
    starts = np.array([0, 1200, 1201, 3000, 5000])
    for thr in (0.2, 0.5, 0.9):
        expected = 0
        for a, b in pairwise(starts):
            trig = Trigger(cfg.with_overrides(threshold=thr))
            expected += sum(trig.update(t, s) for t, s in enumerate(scores[a:b]))
        assert count_triggers(scores, starts, thr, cfg) == expected


def test_calibrate_hits_target_and_ignores_masked_ticks():
    rng = np.random.default_rng(1)
    n = 40000  # ~1.07 h of ticks
    cache = LongformCache(np.zeros((n, 4), np.float16), np.array([0, n]), np.array(["x"]),
                          np.array([[100, 110]]))
    scores = rng.random(n) ** 8
    thr, fa = calibrate(scores, cache, 2.0, DetectorConfig())
    assert fa <= 2.0
    assert count_triggers(scores, cache.stream_starts, thr - 1e-3, DetectorConfig()) / cache.hours > 2.0
    scores[100:110] = 1.0  # the wake word spoken in the long-form audio
    assert calibrate(scores, cache, 2.0, DetectorConfig(), ignore=cache.mask_word("X"))[0] == thr


def test_stream_recall():
    cfg = DetectorConfig()
    s = np.array([[0.9, 0.1, 0.9, 0.1, 0.1, 0.1],  # 2 hits within 3 ticks -> detected
                  [0.9, 0.1, 0.1, 0.9, 0.1, 0.1]])  # hits too far apart -> missed
    assert stream_recall(s, 0.5, cfg) == 0.5


def test_windows_keep_clip_inside():
    rng = np.random.default_rng(0)
    clip = np.ones(14000, np.float32)
    cfg = FinetuneConfig()
    w = positive_windows([clip], 50, cfg, rng)
    assert w.shape == (50, WINDOW)
    lengths = (w > 0.5).sum(1)
    assert lengths.min() >= 14000 / cfg.speed[1] - 2 and lengths.max() <= 14000 / cfg.speed[0] + 2
    p = partial_windows([clip], 50, cfg, rng)
    assert (p > 0.5).sum(1).max() <= 14000 * cfg.partial_fraction[1] / cfg.speed[0] + 2


def test_normalize_word():
    assert normalize_word(" Ça ") == normalize_word("ça") == normalize_word("Ça")


def test_finetune_runs_end_to_end():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    encoder = Encoder(EncoderConfig(width=8, embedding_dim=32)).eval()
    t = np.arange(12000) / 16000
    word = (0.3 * np.sin(2 * np.pi * 300 * t) * np.hanning(len(t))).astype(np.float32)
    bank_emb = rng.normal(size=(500, 32)).astype(np.float32)
    bank_emb /= np.linalg.norm(bank_emb, axis=1, keepdims=True)
    bank = Bank(bank_emb.astype(np.float16), np.array(["hello"] * 10 + ["w"] * 490),
                np.array(["en"] * 250 + ["fr"] * 250))
    aug = Augmenter(AugmentConfig(), NoiseBank(rng.normal(0, 0.1, 64000)), None,
                    RirBank([np.r_[1.0, np.zeros(100)]]))
    cfg = FinetuneConfig(augment_per_clip=8, partial_per_clip=4, hard_negatives=50,
                         random_negatives=200, steps=20)
    res = finetune(encoder, Enrollment("Hello", "en", [word, word * 0.5]), bank, aug, cfg)
    assert res.report["scorer"] == "prototype"
    assert res.report["bank_same_word"] == 10
    assert res.report["positives"] == 16 and res.report["partial_negatives"] == 8
    assert np.isclose(np.linalg.norm(res.prototype), 1.0)
    assert res.head(torch.from_numpy(res.prototype[None])).shape == (1,)


def test_prototype_head_is_a_cosine_against_a_pushed_prototype():
    rng = np.random.default_rng(0)
    proto = np.zeros(8, np.float32)
    proto[0] = 1.0
    hard = np.tile(np.r_[0.8, 0.6, np.zeros(6)].astype(np.float32), (20, 1))
    pos = proto[None] + rng.normal(0, 0.05, (30, 8)).astype(np.float32)
    head = prototype_head(proto, pos, hard, FinetuneConfig(hard_negatives=10, rocchio=0.5))
    direction = head.linear.weight[0].detach().numpy() / FinetuneConfig().score_scale
    assert np.isclose(np.linalg.norm(direction), 1.0, atol=1e-5)
    assert direction[1] < 0  # pushed away from the negatives' second axis
    with torch.no_grad():
        p_score = head(torch.from_numpy(proto[None])).item()
        n_score = head(torch.from_numpy(hard[:1])).item()
    assert p_score > 0 > n_score  # sigmoid midpoint between positives and hard negatives


def test_tick_windows_follow_the_runtime_windowing():
    from core.finetune import HOP, SHIFTS, tick_windows

    clip = np.arange(1, 8001, dtype=np.float32)  # distinguishable samples
    w = tick_windows([clip], 2, None).numpy()
    assert w.shape == (2 * SHIFTS, WINDOW)
    for k in range(SHIFTS):  # shift k: the clip ended k ticks before the window's end
        end = WINDOW - k * HOP
        np.testing.assert_allclose(w[k, end - 8000 : end], clip, atol=1e-2)
        assert np.abs(w[k, end:]).max(initial=0.0) < 1e-2


def test_fit_head_on_a_subset_matches_finetune_on_that_subset():
    from core.finetune import embed_enrollment, fit_head

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    encoder = Encoder(EncoderConfig(width=8, embedding_dim=32)).eval()
    t = np.arange(12000) / 16000
    words = [(0.3 * np.sin(2 * np.pi * f * t) * np.hanning(len(t))).astype(np.float32)
             for f in (300, 320, 340)]
    bank_emb = rng.normal(size=(500, 32)).astype(np.float32)
    bank = Bank((bank_emb / np.linalg.norm(bank_emb, axis=1, keepdims=True)).astype(np.float16),
                np.array(["w"] * 500), np.array(["en"] * 500))
    aug = Augmenter(AugmentConfig(), NoiseBank(rng.normal(0, 0.1, 64000)), None, None)
    cfg = FinetuneConfig(augment_per_clip=4, partial_per_clip=2, hard_negatives=50,
                         random_negatives=200)
    emb = embed_enrollment(encoder, Enrollment("x", "en", words), aug, cfg)
    assert emb.positives.shape == (3, 4, 32) and emb.partial.shape == (3, 2, 32)
    full, part = fit_head(emb, bank, "x", "en", cfg), fit_head(emb, bank, "x", "en", cfg, [0, 2])
    np.testing.assert_allclose(part.prototype, emb.clean[[0, 2]].mean(0)
                               / np.linalg.norm(emb.clean[[0, 2]].mean(0)), atol=1e-6)
    assert part.report["positives"] == 8 and full.report["positives"] == 12
    assert part.report["partial_negatives"] == 4
