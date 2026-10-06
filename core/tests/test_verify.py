"""core.verify: replaying gate-off scores through the runtime's Gate and Trigger must give
exactly what the runtime Detector does with the gate on."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.verify import false_accepts, replay, stream_scores, stream_vad
from runtime import Detector
from runtime.wav import read_wav

RUNTIME = Path(__file__).resolve().parents[2] / "runtime"
MODEL = RUNTIME / "testdata" / "models" / "model.onnx"
VAD = RUNTIME / "assets" / "silero_vad_16k_op15.onnx"


@pytest.mark.parametrize("gate", [True, False])
def test_replay_matches_detector(gate):
    audio, _ = read_wav(RUNTIME / "testdata" / "audio" / "speech.wav")
    det = Detector(MODEL, VAD, gate=gate, record_ticks=True)
    expected = [e.tick for e in det.process(audio)]
    scores, vad = stream_scores(MODEL, audio), stream_vad(VAD, audio)
    gated = [t.score for t in det.ticks]
    np.testing.assert_allclose(np.where([t.gate_open for t in det.ticks], scores, 0.0), gated,
                               atol=1e-6)
    assert replay(scores, vad, det.config, gate) == expected
    assert expected  # the golden stub fires on this file


def test_false_accepts_counts_both_gate_settings():
    audio, _ = read_wav(RUNTIME / "testdata" / "audio" / "speech.wav")
    noise, _ = read_wav(RUNTIME / "testdata" / "audio" / "noise.wav")
    det = Detector(MODEL, VAD)
    streams = [audio, noise]
    vad = [stream_vad(VAD, a) for a in streams]
    fa = false_accepts(MODEL, streams, vad, det.config, workers=2)
    n_gate = len(Detector(MODEL, VAD, gate=True).process(audio)) + \
        len(Detector(MODEL, VAD, gate=True).process(noise))
    assert fa["gate"]["false_accepts"] == n_gate
    assert fa["no_gate"]["false_accepts"] >= n_gate
    assert fa["hours"] == pytest.approx(sum(len(s) for s in fa["scores"]) * 1536 / 16000 / 3600)


def test_stream_embeddings_window_like_the_runtime():
    import json

    from core.verify import stream_embeddings

    golden = json.loads((RUNTIME / "testdata" / "golden.json").read_text())["encoder"]
    audio, _ = read_wav(RUNTIME / "testdata" / "audio" / golden["file"])
    emb = stream_embeddings(RUNTIME / "testdata" / "models" / "encoder.onnx", audio)
    assert len(emb) == len(audio) // 1536
    np.testing.assert_allclose(emb[golden["ticks"]], golden["embeddings"], atol=1e-4)
