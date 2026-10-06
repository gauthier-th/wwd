"""ONNX export, metadata and the portability checker that enforces PLAN.md §3."""

from __future__ import annotations

import copy
import io
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnx
import torch

OPSET = 17
WINDOW_SAMPLES = 24000

# Ops our exported graphs may contain. Everything here runs on the onnxruntime CPU kernels
# that both onnxruntime (Python) and onnxruntime-web (wasm) ship.
ALLOWED_OPS = frozenset(
    {
        # math
        "Add", "Sub", "Mul", "Div", "Pow", "Sqrt", "Exp", "Log", "Neg", "Abs", "Reciprocal",
        "Max", "Min", "Clip",
        # nn
        "Conv", "MatMul", "Gemm", "BatchNormalization", "Relu", "Sigmoid", "Tanh", "Softmax",
        "ReduceMean", "ReduceSum", "ReduceMax",
        # layout (static except batch)
        "Pad", "Reshape", "Transpose", "Squeeze", "Unsqueeze", "Flatten", "Concat", "Slice",
        "Split", "Gather", "Shape", "Expand", "Cast", "Constant", "ConstantOfShape", "Identity",
        # int8 QDQ
        "QuantizeLinear", "DequantizeLinear",
    }
)
BANNED_OPS = frozenset({"STFT", "DFT", "Loop", "Scan", "If"})


class GraphCheckError(ValueError):
    pass


@torch.no_grad()
def freeze_for_export(module: torch.nn.Module) -> torch.nn.Module:
    """Eval-mode copy with every parameter-derived constant precomputed, so the traced graph
    holds plain initializers instead of parameter math or shape-dependent ops."""
    from core.encoder import SubSpectralNorm
    from core.frontend import Frontend

    module = copy.deepcopy(module).eval()
    freq: dict[SubSpectralNorm, int] = {}

    def record_freq(m: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        freq[m] = args[0].shape[2]

    hooks = [
        m.register_forward_pre_hook(record_freq)
        for m in module.modules()
        if isinstance(m, SubSpectralNorm)
    ]
    module(torch.zeros(1, WINDOW_SAMPLES))
    for h in hooks:
        h.remove()
    for m in module.modules():
        if isinstance(m, SubSpectralNorm):
            m.freeze(freq[m])
        elif isinstance(m, Frontend):
            m.freeze()
    return module


def export_module(module: torch.nn.Module, path: str | Path, *, output_name: str,
                  metadata: dict[str, str] | None = None) -> onnx.ModelProto:
    """Export `module` ([N, 24000] float32 audio -> output) to a single self-contained file."""
    module = freeze_for_export(module)
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module,
            (torch.zeros(1, WINDOW_SAMPLES),),
            buf,
            dynamo=False,
            opset_version=OPSET,
            input_names=["audio"],
            output_names=[output_name],
            dynamic_axes={"audio": {0: "batch"}, output_name: {0: "batch"}},
            do_constant_folding=True,
        )
    model = onnx.load_from_string(buf.getvalue())
    inline_identity_initializers(model)
    if metadata:
        set_metadata(model, metadata)
    onnx.save(model, str(path))
    check_graph(path, outputs=[output_name])
    return model


class _ScoredHead(torch.nn.Module):
    def __init__(self, head: torch.nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(embedding))


def compose_head(encoder: str | Path, head: torch.nn.Module, path: str | Path, *,
                 metadata: dict[str, str]) -> onnx.ModelProto:
    """encoder file ([N, 24000] -> embedding) + head (embedding -> logit) + Sigmoid -> `score`.

    The encoder graph is reused as validated (fp32 or int8, PLAN.md §9.3 step 6); only the head
    is exported here, in float32, under a `head/` name prefix.
    """
    enc = onnx.load(str(encoder))
    dim = next(o for o in enc.graph.output if o.name == "embedding")
    dim = dim.type.tensor_type.shape.dim[1].dim_value
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            _ScoredHead(copy.deepcopy(head).eval()), (torch.zeros(1, dim),), buf,
            dynamo=False, opset_version=OPSET, input_names=["embedding"], output_names=["score"],
            dynamic_axes={"embedding": {0: "batch"}, "score": {0: "batch"}},
            do_constant_folding=True,
        )
    scored = onnx.compose.add_prefix(onnx.load_from_string(buf.getvalue()), "head/",
                                     rename_inputs=False, rename_outputs=False)
    scored.ir_version = enc.ir_version
    del enc.metadata_props[:]
    model = onnx.compose.merge_models(enc, scored, io_map=[("embedding", "embedding")],
                                      outputs=["score"])
    set_metadata(model, metadata)
    onnx.save(model, str(path))
    check_graph(path, outputs=["score"])
    return model


def inline_identity_initializers(model: onnx.ModelProto) -> None:
    """Replace `Identity(initializer)` nodes (a TorchScript-exporter artifact of folded
    BatchNorm biases) by initializers, so the quantizer sees every Conv bias as constant."""
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}
    keep = []
    for node in graph.node:
        if node.op_type == "Identity" and node.input[0] in inits:
            copy_ = onnx.TensorProto()
            copy_.CopyFrom(inits[node.input[0]])
            copy_.name = node.output[0]
            graph.initializer.append(copy_)
        else:
            keep.append(node)
    del graph.node[:]
    graph.node.extend(keep)


def set_metadata(model: onnx.ModelProto, metadata: dict[str, str]) -> None:
    existing = {p.key: p for p in model.metadata_props}
    for key, value in metadata.items():
        if key in existing:
            existing[key].value = str(value)
        else:
            model.metadata_props.add(key=key, value=str(value))


def read_metadata(path: str | Path) -> dict[str, str]:
    return {p.key: p.value for p in onnx.load(str(path), load_external_data=False).metadata_props}


def _ops(graph: onnx.GraphProto) -> Iterable[onnx.NodeProto]:
    for node in graph.node:
        yield node
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                yield from _ops(attr.g)


def check_graph(path: str | Path, *, outputs: list[str]) -> None:
    """Raise GraphCheckError unless the file follows the dual-runtime graph rules."""
    path = Path(path)
    model = onnx.load(str(path), load_external_data=False)
    problems: list[str] = []
    onnx.checker.check_model(model)

    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and imp.version > OPSET:
            problems.append(f"opset {imp.version} > {OPSET}")
        elif imp.domain not in ("", "ai.onnx"):
            problems.append(f"non-standard opset domain {imp.domain!r}")
    for node in _ops(model.graph):
        if node.domain not in ("", "ai.onnx"):
            problems.append(f"{node.op_type}: custom/contrib domain {node.domain!r}")
        elif node.op_type in BANNED_OPS:
            problems.append(f"banned op {node.op_type}")
        elif node.op_type not in ALLOWED_OPS:
            problems.append(f"op {node.op_type} not in allowlist")
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            problems.append(f"external data in initializer {init.name}")

    inputs = [i for i in model.graph.input if i.name not in {t.name for t in model.graph.initializer}]
    if [i.name for i in inputs] != ["audio"]:
        problems.append(f"inputs must be ['audio'], got {[i.name for i in inputs]}")
    else:
        t = inputs[0].type.tensor_type
        dims = [d.dim_value or d.dim_param for d in t.shape.dim]
        if t.elem_type != onnx.TensorProto.FLOAT or len(dims) != 2 or dims[1] != WINDOW_SAMPLES:
            problems.append(f"audio must be float32 [batch, {WINDOW_SAMPLES}], got {dims}")
    got_outputs = [o.name for o in model.graph.output]
    if got_outputs != outputs:
        problems.append(f"outputs must be {outputs}, got {got_outputs}")
    for o in model.graph.output:
        if o.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
            problems.append(f"output {o.name} must be float32")

    if problems:
        raise GraphCheckError(f"{path.name}: " + "; ".join(dict.fromkeys(problems)))


def quantize_int8(src: str | Path, dst: str | Path, calibration: np.ndarray, *,
                  output_name: str) -> None:
    """Static int8 quantization in QDQ format, calibrated on [M, 24000] audio windows.

    The frontend stays float32: quantizing raw audio to 8 bits would erase quiet speech.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self._it = iter(calibration.astype(np.float32)[:, None, :])

        def get_next(self):
            window = next(self._it, None)
            return None if window is None else {"audio": window}

    dst = Path(dst)
    prepped = dst.with_suffix(".prep.onnx")
    quant_pre_process(str(src), str(prepped), skip_symbolic_shape=True)
    frontend = [n.name for n in onnx.load(str(prepped)).graph.node if "/frontend/" in n.name]
    if not frontend:
        raise ValueError("no frontend nodes found; refusing to quantize raw audio")
    try:
        quantize_static(
            str(prepped), str(dst), Reader(),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            nodes_to_exclude=frontend,
        )
    finally:
        prepped.unlink(missing_ok=True)
    model = onnx.load(str(dst))
    del model.metadata_props[:]  # drop the quantizer's own keys; keep exactly the source's
    set_metadata(model, read_metadata(src))
    onnx.save(model, str(dst))
    check_graph(dst, outputs=[output_name])
