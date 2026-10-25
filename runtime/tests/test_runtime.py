"""Python side of the dual-runtime contract (runtime/js/test mirrors these tests)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from runtime import Detector, DetectorConfig, Gate, MultiDetector, SileroVad, Trigger
from runtime.onnxfile import read_metadata, split_model
from runtime.vad import make_session
from runtime.wav import read_wav

DATA = Path(__file__).resolve().parents[1] / "testdata"
VAD = Path(__file__).resolve().parents[1] / "assets" / "silero_vad_16k_op15.onnx"
GOLDEN = json.loads((DATA / "golden.json").read_text())
CASES = json.loads((DATA / "logic_cases.json").read_text())
TOL = GOLDEN["tolerance"]
FILES = sorted(GOLDEN["files"])
MODELS = {"fp32": DATA / "models" / "model.onnx", "int8": DATA / "models" / "model.int8.onnx"}
WORDS = {name: DATA / "models" / name for name in GOLDEN["words"]}  # composed like shipped files


def audio(name: str) -> np.ndarray:
    samples, rate = read_wav(DATA / "audio" / name)
    assert rate == 16000
    return samples


def golden(model: str, name: str) -> dict:
    """Expected settings, ticks and events of one model file (a MODELS kind or a WORDS file)."""
    if model in WORDS:
        return {"config": GOLDEN["words"][model]["config"], **GOLDEN["words"][model]["files"][name]}
    f = GOLDEN["files"][name]
    return {"config": GOLDEN["config"], "vad_max": f["vad_max"], "gate_open": f["gate_open"],
            "scores": f["scores"][model], "events": f["events"][model]}


def check_golden(model: str, name: str, gate: bool, ticks: list, events: list) -> None:
    want = golden(model, name)
    scores = np.array(want["scores"])
    gate_open = np.array(want["gate_open"]) if gate else np.ones(len(scores), bool)
    assert [t.gate_open for t in ticks] == gate_open.tolist()
    np.testing.assert_allclose([t.vad_max for t in ticks], want["vad_max"], atol=TOL["vad"], rtol=0)
    got, expected = np.array([t.score for t in ticks]), np.where(gate_open, scores, 0.0)
    np.testing.assert_allclose(got, expected, atol=TOL["int8" if model == "int8" else "fp32"], rtol=0)
    if model == "int8":  # rare one-step requantization flips are allowed, systematic drift is not
        assert np.abs(got - expected).mean() <= TOL["int8_mean"]
    assert [e.tick for e in events] == want["events"]["gate" if gate else "no_gate"]


def stream(det, a: np.ndarray, chunk: int | None) -> list:
    events = []
    for start in range(0, len(a), chunk or len(a)):
        events += det.process(a[start : start + (chunk or len(a))])
    return events


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


@pytest.mark.parametrize("model", ["fp32", "int8", *WORDS])
@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("gate", [True, False], ids=["gate", "no_gate"])
@pytest.mark.parametrize("chunk", [None, 1000], ids=["whole", "chunked"])
def test_detector_golden(model, name, gate, chunk):
    det = Detector(MODELS.get(model) or WORDS[model], VAD, gate=gate, record_ticks=True)
    assert det.config == DetectorConfig.from_metadata(golden(model, name)["config"])
    events = stream(det, audio(name), chunk)
    check_golden(model, name, gate, det.ticks, events)


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


# ----------------------------------------------------------------------------- several wake words


def test_split_shares_the_encoder_of_composed_files():
    a, b = (split_model(path.read_bytes()) for path in WORDS.values())
    assert a and b and a.encoder == b.encoder and a.head != b.head
    for path in (*MODELS.values(), DATA / "models" / "encoder.onnx", VAD):
        assert split_model(path.read_bytes()) is None  # no `embedding` tensor between two parts
    assert split_model(b"\x3a\xff") is None  # truncated


def test_split_encoder_gives_the_golden_embeddings():
    enc = GOLDEN["encoder"]
    padded = np.concatenate([np.zeros(24000, np.float32), audio(enc["file"])])
    windows = np.stack([padded[(t + 1) * 1536 : (t + 1) * 1536 + 24000] for t in enc["ticks"]])
    encoder = split_model(WORDS["word_a.onnx"].read_bytes()).encoder
    got = make_session(encoder).run(None, {"audio": windows})[0]
    np.testing.assert_allclose(got, enc["embeddings"], atol=TOL["embedding"], rtol=0)


def test_metadata_reader_matches_onnxruntime():
    for path in (*MODELS.values(), *WORDS.values()):
        meta = dict(make_session(str(path)).get_modelmeta().custom_metadata_map)
        assert read_metadata(path.read_bytes()) == meta


@pytest.mark.parametrize("name", FILES)
@pytest.mark.parametrize("gate", [True, False], ids=["gate", "no_gate"])
@pytest.mark.parametrize("chunk", [None, 1000], ids=["whole", "chunked"])
def test_multi_detector_golden(name, gate, chunk):
    """The two words share their encoder; the stub has no `embedding` and runs whole. Each gets
    exactly its own golden ticks and events."""
    models = ["word_a.onnx", "fp32", "word_b.onnx"]
    det = MultiDetector([MODELS.get(m) or WORDS[m] for m in models], VAD, gate=gate, record_ticks=True)
    assert det.groups == [[0, 2], [1]]
    assert [w.name for w in det.words] == ["golden-a", "golden-stub", "golden-b"]
    events = stream(det, audio(name), chunk)
    assert events == sorted(events, key=lambda e: (e.tick, e.word))
    for word, model in zip(det.words, models, strict=True):
        assert word.config == DetectorConfig.from_metadata(golden(model, name)["config"])
        check_golden(model, name, gate, word.ticks, [e for e in events if e.word == word.index])


def test_multi_detector_without_sharing_gives_the_same_results():
    a = audio("speech.wav")
    runs = [MultiDetector(list(WORDS.values()), VAD, record_ticks=True, share_encoder=share)
            for share in (True, False)]
    assert [r.groups for r in runs] == [[[0, 1]], [[0], [1]]]
    events = [stream(r, a, 1536) for r in runs]
    assert [(e.tick, e.word) for e in events[0]] == [(e.tick, e.word) for e in events[1]]
    for shared, alone in zip(runs[0].words, runs[1].words, strict=True):
        np.testing.assert_allclose([t.score for t in shared.ticks], [t.score for t in alone.ticks],
                                   atol=TOL["fp32"], rtol=0)


def test_multi_detector_settings():
    paths = list(WORDS.values())
    det = MultiDetector(paths, overrides={"threshold": 0.9})  # one dict for every word
    assert [w.config.threshold for w in det.words] == [0.9, 0.9]
    det = MultiDetector(paths, overrides=[None, {"threshold": 0.9}])  # or one per word
    calibrated = float(GOLDEN["words"]["word_a.onnx"]["config"]["threshold"])
    assert [w.config.threshold for w in det.words] == [calibrated, 0.9]
    assert MultiDetector(paths[:1]).groups == [[0]]  # one file runs whole
    with pytest.raises(ValueError, match="same window_samples and hop_samples"):
        MultiDetector(paths, overrides=[None, {"hop_samples": 3072}])
    with pytest.raises(ValueError, match="overrides"):
        MultiDetector(paths, overrides=[None])
    with pytest.raises(ValueError, match="at least one"):
        MultiDetector([])


def test_detector_config_replaces_the_metadata():
    det = Detector(MODELS["fp32"], config=DetectorConfig(threshold=0.9, smoothing_k=1),
                   overrides={"smoothing_n": 4, "refractory_ms": None})
    assert det.config == DetectorConfig(threshold=0.9, smoothing_k=1, smoothing_n=4)


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
