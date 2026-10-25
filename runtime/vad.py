"""Streaming Silero VAD (16 kHz) on plain onnxruntime."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .spec import SAMPLE_RATE, VAD_CONTEXT, VAD_FRAME


def make_session(model: str | Path | bytes) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3  # Silero ships unused initializers; silence the warnings
    opts.intra_op_num_threads = 1
    # x86 without VNNI: the default u8s8 int8 matmul can overflow 16 bits and drift every score
    opts.add_session_config_entry("session.x64quantprecision", "1")
    src = model if isinstance(model, bytes) else str(model)
    return ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])


class SileroVad:
    """One call per 512-sample frame. Like the official wrapper, each call prepends the last
    64 samples of the previous call and carries the recurrent state forward."""

    def __init__(self, model: str | Path | bytes) -> None:
        self._session = make_session(model)
        self._has_sr = any(i.name == "sr" for i in self._session.get_inputs())
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (VAD_FRAME,):
            raise ValueError(f"expected {VAD_FRAME} samples, got {frame.shape}")
        x = np.concatenate([self._context, frame.astype(np.float32, copy=False)[None]], axis=1)
        feeds = {"input": x, "state": self._state}
        if self._has_sr:
            feeds["sr"] = np.array(SAMPLE_RATE, dtype=np.int64)
        prob, self._state = self._session.run(None, feeds)
        self._context = x[:, -VAD_CONTEXT:]
        return float(prob.reshape(-1)[0])
