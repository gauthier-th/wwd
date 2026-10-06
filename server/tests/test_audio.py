from __future__ import annotations

import numpy as np
import pytest

from runtime.vad import SileroVad
from server import audio
from server.basemodel import VAD_FILE


@pytest.fixture
def vad() -> SileroVad:
    return SileroVad(VAD_FILE)


def test_speech_regions_hysteresis():
    probs = np.r_[[0] * 10, [0.9] * 10, [0.2] * 3, [0.9] * 5, [0] * 20, [0.9] * 2, [0] * 15]
    # the 3-frame dip doesn't split the word; the 2-frame blip is too short to count
    assert audio.speech_regions(probs) == [(10 * 512, 28 * 512)]


def word_from(speech: np.ndarray, vad: SileroVad) -> np.ndarray:
    clips = [c for c in audio.split_positives(speech, vad) if c.accepted and c.speech_s < 0.9]
    return clips[0].audio


def test_split_positives_cuts_a_take_into_normalized_clips(speech, vad):
    word = word_from(speech, vad)
    gap = np.zeros(int(0.8 * audio.SR), np.float32)
    take = np.concatenate([gap] + [x for g in (0.3, 1.0, 0.6, 0.15, 0.8) for x in (word * g, gap)])
    clips = audio.split_positives(take, vad)
    assert len(clips) == 5 and all(c.accepted for c in clips)
    starts = [c.start_s for c in clips]
    assert starts == sorted(starts)
    for c in clips:
        assert c.duration_s <= audio.WINDOW_S
        assert np.abs(c.audio).max() <= audio.PEAK_LIMIT + 1e-4
        assert -26 < c.rms_dbfs < -14  # normalized whatever the take's level


def test_split_positives_rejects_long_and_clipped(speech, vad):
    regions = audio.speech_regions(audio.vad_probs(speech, vad))
    run_on = np.concatenate([speech[a:b] for a, b in regions[:4]])  # > 1.3 s without a pause
    pad = np.zeros(audio.SR, np.float32)
    [long] = audio.split_positives(np.concatenate([pad, run_on, pad]), vad)
    assert not long.accepted and long.reason == "too_long"
    word = word_from(speech, vad)
    hot = np.clip(word / np.abs(word).max() * 3, -1, 1)
    [clipped] = audio.split_positives(np.concatenate([pad, hot, pad]), vad)
    assert clipped.reason == "clipped"


def test_upload_errors(vad, noise):
    with pytest.raises(audio.UploadError, match="unreadable"):
        audio.decode(b"not audio at all", 60)
    silence = np.zeros(audio.SR * 2, np.float32)
    with pytest.raises(audio.UploadError, match="silent"):
        audio.split_positives(silence, vad)
    with pytest.raises(audio.UploadError, match="speech"):
        audio.prepare_other(noise, "other_speech", vad)
    assert audio.prepare_other(noise, "ambience", vad).accepted
    with pytest.raises(audio.UploadError, match="too long"):
        audio.decode(audio.encode_wav(np.zeros(audio.SR * 5, np.float32)), 4)


def test_decode_resamples_to_16k():
    import io

    import soundfile as sf

    t = np.arange(44100) / 44100
    buf = io.BytesIO()
    sf.write(buf, np.stack([np.sin(2 * np.pi * 440 * t)] * 2, axis=1) * 0.5, 44100, format="FLAC")
    pcm = audio.decode(buf.getvalue(), 10)
    assert len(pcm) == 16000
    assert abs(audio.rms(pcm) - 0.5 / np.sqrt(2)) < 1e-2


def test_outliers_flag_the_odd_one_out():
    rng = np.random.default_rng(0)
    base = rng.normal(size=16)
    e = np.stack([base + rng.normal(0, 0.2, 16) for _ in range(6)] + [rng.normal(size=16)])
    sims, flags = audio.outliers(e)
    assert flags.tolist() == [False] * 6 + [True]
    assert sims[:6].min() > 0.9
