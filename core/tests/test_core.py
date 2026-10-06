from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper

from core.encoder import Encoder, EncoderConfig, SubSpectralNorm
from core.export import (
    GraphCheckError,
    check_graph,
    export_module,
    freeze_for_export,
    quantize_int8,
    read_metadata,
)
from core.frontend import Frontend, mel_filterbank
from core.model import Head, WakeWordModel


@pytest.fixture(scope="module")
def model() -> WakeWordModel:
    torch.manual_seed(0)
    m = WakeWordModel(Encoder(EncoderConfig(width=16)), Head()).eval()
    for mod in m.modules():  # non-trivial BN statistics so freezing is exercised
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
    return m


def test_conv_stft_mel_matches_numpy():
    fe = Frontend()
    x = np.random.default_rng(0).normal(0, 0.1, 24000).astype(np.float32)
    got = fe.mel_energies(torch.from_numpy(x)[None])[0].numpy()

    n_fft, hop = 400, 160
    frames = np.stack([x[i : i + n_fft] for i in range(0, len(x) - n_fft + 1, hop)])
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    mel = mel_filterbank(n_fft, 40, 16000, 20.0, 7600.0).numpy()
    np.testing.assert_allclose(got, (power @ mel.T).T, rtol=1e-3, atol=1e-4)
    assert got.shape == (40, fe.n_frames(24000)) == (40, 148)


def test_pcen_fir_tracks_iir_smoother():
    fe = Frontend()
    e = torch.rand(1, 40, 148) + 0.1
    fir = torch.nn.functional.conv1d(
        torch.nn.functional.pad(e, (63, 0), mode="replicate"), fe.smoother(), groups=40)
    s = 0.04
    iir = torch.empty_like(e)
    m = e[..., 0]
    for t in range(e.shape[-1]):
        m = (1 - s) * m + s * e[..., t]
        iir[..., t] = m
    # Only the truncated tail (0.96^64 ~ 7%) differs; warm-up differs too, so skip it.
    assert torch.allclose(fir[..., 64:], iir[..., 64:], rtol=0.1)


def test_subspectral_norm_freeze_is_exact():
    ssn = SubSpectralNorm(8, 5).eval()
    ssn.bn.running_mean.uniform_(-1, 1)
    ssn.bn.running_var.uniform_(0.5, 2)
    x = torch.randn(2, 8, 20, 30)
    ref = ssn(x)
    ssn.freeze(20)
    assert torch.allclose(ssn(x), ref, atol=1e-6)


def test_freeze_for_export_preserves_outputs(model):
    x = torch.randn(3, 24000) * 0.1
    with torch.no_grad():
        assert torch.allclose(freeze_for_export(model)(x), model(x), atol=1e-6)


def test_export_parity_and_metadata(model, tmp_path):
    path = tmp_path / "m.onnx"
    export_module(model, path, output_name="score", metadata={"threshold": "0.6"})
    assert read_metadata(path) == {"threshold": "0.6"}
    x = (torch.randn(4, 24000) * 0.1).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"audio": x})[0]
    with torch.no_grad():
        np.testing.assert_allclose(got, model(torch.from_numpy(x)).numpy(), atol=1e-5)
    assert got.shape == (4,)


def test_int8_keeps_frontend_float(model, tmp_path):
    src, dst = tmp_path / "m.onnx", tmp_path / "m.int8.onnx"
    export_module(model, src, output_name="score", metadata={"k": "v"})
    quantize_int8(src, dst, np.random.default_rng(0).normal(0, 0.1, (8, 24000)), output_name="score")
    graph = onnx.load(str(dst)).graph
    dequantized = {o for n in graph.node if n.op_type == "DequantizeLinear" for o in n.output}
    frontend = [n for n in graph.node if "/frontend/" in n.name
                and n.op_type not in ("QuantizeLinear", "DequantizeLinear")]
    assert frontend, "frontend nodes not found by name"
    for node in frontend:  # neither activations nor weights of the frontend went through int8
        assert not set(node.input) & dequantized, node.name
    assert any(n.op_type == "QuantizeLinear" for n in graph.node)
    assert read_metadata(dst) == {"k": "v"}


def _tiny_model(path, node, *, domain_opset=None):
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("audio", TensorProto.FLOAT, ["batch", 24000])],
        [helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 24000])],
    )
    opsets = [helper.make_opsetid("", 17)] + ([domain_opset] if domain_opset else [])
    onnx.save(helper.make_model(graph, opset_imports=opsets), str(path))


def test_checker_rejects_non_portable_graphs(tmp_path):
    ok = tmp_path / "ok.onnx"
    _tiny_model(ok, helper.make_node("Relu", ["audio"], ["score"]))
    check_graph(ok, outputs=["score"])

    bad = tmp_path / "bad.onnx"
    _tiny_model(bad, helper.make_node("Gelu", ["audio"], ["score"], domain="com.microsoft"),
                domain_opset=helper.make_opsetid("com.microsoft", 1))
    with pytest.raises(GraphCheckError, match="com.microsoft"):
        check_graph(bad, outputs=["score"])

    with pytest.raises(GraphCheckError, match="outputs must be"):
        check_graph(ok, outputs=["embedding"])


@pytest.mark.parametrize("int8", [False, True], ids=["fp32", "int8"])
def test_compose_head_matches_encoder_plus_head(model, tmp_path, int8):
    from core.export import compose_head
    from core.model import PrototypeHead

    enc = tmp_path / "encoder.onnx"
    export_module(model.encoder, enc, output_name="embedding", metadata={"base_model_version": "x"})
    if int8:
        calib = np.random.default_rng(0).normal(0, 0.1, (8, 24000)).astype(np.float32)
        quantize_int8(enc, tmp_path / "encoder.int8.onnx", calib, output_name="embedding")
        enc = tmp_path / "encoder.int8.onnx"
    direction = torch.nn.functional.normalize(torch.randn(256), dim=0)
    head = PrototypeHead(direction, 20.0, -4.0)
    compose_head(enc, head, tmp_path / "model.onnx", metadata={"threshold": "0.75", "wake_word": "ça"})
    assert read_metadata(tmp_path / "model.onnx") == {"threshold": "0.75", "wake_word": "ça"}
    x = np.random.default_rng(1).normal(0, 0.1, (3, 24000)).astype(np.float32)
    emb = ort.InferenceSession(str(enc)).run(None, {"audio": x})[0]
    got = ort.InferenceSession(str(tmp_path / "model.onnx")).run(None, {"audio": x})[0]
    want = torch.sigmoid(head(torch.from_numpy(emb))).detach().numpy()
    assert got.shape == (3,)
    np.testing.assert_allclose(got, want, atol=1e-6)
