"""Python side of the dual-runtime contract (runtime/js/test mirrors these tests)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runtime import Detector, DetectorConfig, Gate, SileroVad, Trigger
from runtime.wav import read_wav

DATA = Path(__file__).resolve().parents[1] / "testdata"
VAD = Path(__file__).resolve().parents[1] / "assets" / "silero_vad_16k_op15.onnx"
GOLDEN = json.loads((DATA / "golden.json").read_text())
CASES = json.loads((DATA / "logic_cases.json").read_text())
TOL = GOLDEN["tolerance"]
FILES = sorted(GOLDEN["files"])
MODELS = {"fp32": DATA / "models" / "model.onnx", "int8": DATA / "models" / "model.int8.onnx"}


def audio(name: str) -> np.ndarray:
    samples, rate = read_wav(DATA / "audio" / name)
    assert rate == 16000
    return samples


# ----------------------------------------------------------------------------- logic


@pytest.mark.parametrize("case", CASES["trigger"], ids=lambda c: c["name"])
def test_trigger_cases(case):
    trigger = Trigger(DetectorConfig().with_overrides(**case["config"]))
    fired = [t for t, s in enumerate(case["scores"]) if trigger.update(t, s)]
    assert fired == case["expected"]


@pytest.mark.parametrize("case", CASES["gate"], ids=lambda c: c["name"])
def test_gate_cases(case):
    cfg = DetectorConfig().with_overrides(**case["config"])
    gate, opened, maxes = Gate(cfg), [], []
    for i, p in enumerate(case["vad"]):
        gate.push(p)
        if (i + 1) % cfg.frames_per_tick == 0:
            opened.append(gate.open)
            maxes.append(gate.vad_max)
    assert opened == case["expected_open"]
    assert maxes == pytest.approx(case["expected_vad_max"])


def test_config_roundtrip():
    cfg = DetectorConfig(threshold=0.625, refractory_ms=1000)
    assert DetectorConfig.from_metadata({**cfg.to_metadata(), "wake_word": "x"}) == cfg


# ----------------------------------------------------------------------------- golden


@pytest.mark.parametrize("name", FILES)
def test_vad_matches_official_wrapper(name):
    vad, a = SileroVad(VAD), audio(name)
    got = [vad(a[i * 512 : (i + 1) * 512]) for i in range(len(a) // 512)]
    np.testing.assert_allclose(got, GOLDEN["files"][name]["vad"], atol=TOL["vad"], rtol=0)


@pytest.mark.parametrize("kind", ["fp32", "int8"])
@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("gate", [True, False], ids=["gate", "no_gate"])
@pytest.mark.parametrize("chunk", [None, 1000], ids=["whole", "chunked"])
def test_detector_golden(kind, name, gate, chunk):
    expected = GOLDEN["files"][name]
    det = Detector(MODELS[kind], VAD, gate=gate, record_ticks=True)
    assert det.config == DetectorConfig.from_metadata(GOLDEN["config"])
    a = audio(name)
    events = []
    for start in range(0, len(a), chunk or len(a)):
        events += det.process(a[start : start + (chunk or len(a))])

    scores = np.array(expected["scores"][kind])
    gate_open = np.array(expected["gate_open"]) if gate else np.ones(len(scores), bool)
    assert [t.gate_open for t in det.ticks] == gate_open.tolist()
    np.testing.assert_allclose([t.vad_max for t in det.ticks], expected["vad_max"],
                               atol=TOL["vad"], rtol=0)
    got, want = np.array([t.score for t in det.ticks]), np.where(gate_open, scores, 0.0)
    np.testing.assert_allclose(got, want, atol=TOL[kind], rtol=0)
    if kind == "int8":  # rare one-step requantization flips are allowed, systematic drift is not
        assert np.abs(got - want).mean() <= TOL["int8_mean"]
    assert [e.tick for e in events] == expected["events"][kind]["gate" if gate else "no_gate"]


def test_encoder_golden():
    import onnxruntime as ort

    enc = GOLDEN["encoder"]
    a = audio(enc["file"])
    padded = np.concatenate([np.zeros(24000, np.float32), a])
    windows = np.stack([padded[(t + 1) * 1536 : (t + 1) * 1536 + 24000] for t in enc["ticks"]])
    sess = ort.InferenceSession(str(DATA / "models" / "encoder.onnx"), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"audio": windows})[0]
    np.testing.assert_allclose(got, enc["embeddings"], atol=TOL["embedding"], rtol=0)
    np.testing.assert_allclose(np.linalg.norm(got, axis=1), 1.0, atol=1e-5)


def test_listen_record_writes_a_16k_take(tmp_path):
    from runtime.listen import record

    class FakeStream:  # sounddevice.InputStream, delivering a 440 Hz tone
        def __init__(self, **kwargs):
            assert kwargs["samplerate"] == 16000 and kwargs["channels"] == 1
            self.t = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, n):
            block = 0.5 * np.sin(2 * np.pi * 440 * (self.t + np.arange(n)) / 16000)
            self.t += n
            return block.astype(np.float32)[:, None], False

    class FakeSd:
        InputStream = FakeStream

    out = tmp_path / "take.wav"
    assert record(FakeSd, str(out), 2.5, None) == 0
    audio, rate = read_wav(out)
    assert rate == 16000 and len(audio) == 40000
    assert abs(np.abs(audio).max() - 0.5) < 1e-3
