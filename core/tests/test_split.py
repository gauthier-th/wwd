"""Files as core.export composes them, split by the runtime to listen for several wake words
(runtime/SPEC.md §5): int8 encoders share too, and never with float32 ones."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from core.export import compose_head, quantize_int8
from core.model import Head, PrototypeHead
from runtime import Detector, MultiDetector
from runtime.wav import read_wav

RUNTIME = Path(__file__).resolve().parents[2] / "runtime"
ENCODER = RUNTIME / "testdata" / "models" / "encoder.onnx"
VAD = RUNTIME / "assets" / "silero_vad_16k_op15.onnx"


def test_int8_and_float32_files_share_their_own_encoder(tmp_path):
    audio, _ = read_wav(RUNTIME / "testdata" / "audio" / "speech.wav")
    calibration = np.stack([audio[s : s + 24000] for s in range(0, len(audio) - 24000, 24000)])
    quantize_int8(ENCODER, tmp_path / "encoder.int8.onnx", calibration, output_name="embedding")
    torch.manual_seed(0)
    heads = {"a": PrototypeHead(torch.nn.functional.normalize(torch.randn(256), dim=0), 20.0, 0.0),
             "b": Head()}
    files = []
    for encoder in (ENCODER, tmp_path / "encoder.int8.onnx"):
        for name, head in heads.items():
            files.append(tmp_path / f"{name}.{encoder.stem}.onnx")
            compose_head(encoder, head, files[-1], metadata={"wake_word": name, "threshold": "0.5"})

    det = MultiDetector(files, VAD, gate=False, record_ticks=True)
    assert det.groups == [[0, 1], [2, 3]]
    events = det.process(audio)
    for word, path in zip(det.words, files, strict=True):
        alone = Detector(path, VAD, gate=False, record_ticks=True)
        assert [e.tick for e in alone.process(audio)] == [e.tick for e in events if e.word == word.index]
        np.testing.assert_allclose([t.score for t in word.ticks], [t.score for t in alone.ticks],
                                   atol=1e-6, rtol=0)
