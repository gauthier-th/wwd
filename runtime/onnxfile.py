"""What the runtime reads from ONNX files itself, without an ONNX library: the metadata, and the
split of a wake-word file into its encoder and its head (runtime/SPEC.md §5).

Only the protobuf wire format is parsed. Fields that a split doesn't change are copied as they
are, so files that share an encoder give byte-identical encoder parts. Mirrors
runtime/js/src/onnxMeta.ts and runtime/js/src/onnxSplit.ts.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

EMBEDDING = "embedding"  # the encoder's output, which the head reads (core.export.compose_head)

# Field numbers from onnx.proto
MODEL_GRAPH, MODEL_METADATA = 7, 14
GRAPH_NODE, GRAPH_INITIALIZER, GRAPH_INPUT, GRAPH_OUTPUT = 1, 5, 11, 12
GRAPH_VALUE_INFO, GRAPH_SPARSE_INITIALIZER = 13, 15
NODE_INPUT, NODE_OUTPUT, NODE_ATTRIBUTE = 1, 2, 5
ATTRIBUTE_GRAPH, ATTRIBUTE_GRAPHS = 6, 11
TENSOR_NAME = 8
VALUE_INFO_NAME = 1
ENTRY_KEY, ENTRY_VALUE = 1, 2
LEN = 2  # wire type of strings and messages


def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7


def _fields(buf: memoryview) -> Iterator[tuple[int, int, memoryview, memoryview | None]]:
    """(field number, wire type, the field's bytes, its payload if length-delimited)."""
    pos = 0
    while pos < len(buf):
        start = pos
        tag, pos = _varint(buf, pos)
        wire, payload = tag & 7, None
        if wire == 0:
            _, pos = _varint(buf, pos)
        elif wire == 1:
            pos += 8
        elif wire == LEN:
            size, pos = _varint(buf, pos)
            payload = buf[pos : pos + size]
            pos += size
        elif wire == 5:
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if pos > len(buf):
            raise ValueError("truncated protobuf")
        yield tag >> 3, wire, buf[start:pos], payload


def _strings(msg: memoryview, field: int) -> list[str]:
    return [bytes(p).decode() for f, w, _, p in _fields(msg) if f == field and w == LEN]


def _name(msg: memoryview, field: int) -> str:
    return next(iter(_strings(msg, field)), "")


def _encode(field: int, payload: bytes) -> bytes:
    out, n = bytearray(), len(payload)
    for value in (field << 3 | LEN, n):
        while value > 0x7F:
            out.append(value & 0x7F | 0x80)
            value >>= 7
        out.append(value)
    return bytes(out) + payload


def read_metadata(model: bytes) -> dict[str, str]:
    """ModelProto.metadata_props (a later entry wins over an earlier one with the same key)."""
    meta = {}
    for field, wire, _, entry in _fields(memoryview(model)):
        if field == MODEL_METADATA and wire == LEN:
            meta[_name(entry, ENTRY_KEY)] = _name(entry, ENTRY_VALUE)
    return meta


@dataclass(frozen=True)
class Split:
    """Two ModelProtos, without the file's metadata."""

    encoder: bytes  # audio -> embedding
    head: bytes  # embedding -> score


def split_model(model: bytes) -> Split | None:
    """The encoder and head of a wake-word file, or None when no `embedding` tensor separates
    them: the file then runs whole. Never raises: a broken file fails in onnxruntime instead."""
    try:
        return _split(memoryview(model))
    except (ValueError, IndexError, UnicodeDecodeError):
        return None


def _split(buf: memoryview) -> Split | None:
    top = list(_fields(buf))
    graphs = [p for f, w, _, p in top if f == MODEL_GRAPH and w == LEN]
    if len(graphs) != 1:
        return None
    graph = list(_fields(graphs[0]))
    if any(f == GRAPH_SPARSE_INITIALIZER for f, *_ in graph):
        return None
    reads, writes = [], []
    for f, _, _, node in graph:
        if f != GRAPH_NODE:
            continue
        fields = list(_fields(node))
        # A subgraph can read outer tensors without listing them as inputs: don't split.
        if any(f == NODE_ATTRIBUTE and any(a in (ATTRIBUTE_GRAPH, ATTRIBUTE_GRAPHS)
                                           for a, *_ in _fields(p)) for f, _, _, p in fields):
            return None
        reads.append([bytes(p).decode() for f, w, _, p in fields if f == NODE_INPUT])
        writes.append([bytes(p).decode() for f, w, _, p in fields if f == NODE_OUTPUT])
    producer = {name: i for i, names in enumerate(writes) for name in names if name}
    initializers = {_name(p, TENSOR_NAME) for f, _, _, p in graph if f == GRAPH_INITIALIZER}
    declared = {f: [p for g, _, _, p in graph
                    if g == f and _name(p, VALUE_INFO_NAME) not in initializers]
                for f in (GRAPH_INPUT, GRAPH_OUTPUT)}
    inputs = {_name(p, VALUE_INFO_NAME) for p in declared[GRAPH_INPUT]}
    outputs = [_name(p, VALUE_INFO_NAME) for p in declared[GRAPH_OUTPUT]]
    if EMBEDDING not in producer or EMBEDDING in outputs:
        return None

    def upstream(targets: list[str], stop: str | None) -> tuple[set[int], set[str]]:
        """Nodes needed to compute `targets`, and the tensors they read from outside."""
        nodes, leaves, seen, todo = set(), set(), set(), list(targets)
        while todo:
            name = todo.pop()
            if not name or name in seen:
                continue
            seen.add(name)
            if name == stop or name not in producer:
                leaves.add(name)
            else:
                nodes.add(producer[name])
                todo += reads[producer[name]]
        return nodes, leaves - initializers

    encoder, encoder_reads = upstream([EMBEDDING], None)
    head, head_reads = upstream(outputs, EMBEDDING)
    if not encoder_reads <= inputs or head_reads != {EMBEDDING}:
        return None
    embedding = next((p for f, _, _, p in graph
                      if f == GRAPH_VALUE_INFO and _name(p, VALUE_INFO_NAME) == EMBEDDING), None)
    # Undeclared: float [N, D] with both sizes unknown. ValueInfoProto{name, type{tensor_type{
    # elem_type: FLOAT, shape{dim {}, dim {}}}}}
    embedding = bytes(embedding) if embedding is not None else (
        _encode(VALUE_INFO_NAME, EMBEDDING.encode())
        + _encode(2, _encode(1, b"\x08\x01" + _encode(2, _encode(1, b"") * 2))))

    def part(nodes: set[int], new_inputs: list[bytes], new_outputs: list[bytes]) -> bytes:
        used = {name for i in nodes for name in reads[i]}
        out, index, placed = [], 0, set()
        for f, _, raw, p in graph:
            if f == GRAPH_NODE:
                if index in nodes:
                    out.append(raw)
                index += 1
            elif f == GRAPH_INITIALIZER:
                if _name(p, TENSOR_NAME) in used:
                    out.append(raw)
            elif f in (GRAPH_INPUT, GRAPH_OUTPUT):
                if _name(p, VALUE_INFO_NAME) in initializers:  # IR < 4 lists them as inputs too
                    if _name(p, VALUE_INFO_NAME) in used:
                        out.append(raw)
                elif f not in placed:
                    placed.add(f)
                    out += [_encode(f, v) for v in (new_inputs if f == GRAPH_INPUT else new_outputs)]
            elif f != GRAPH_VALUE_INFO:  # shapes of tensors that may be gone: onnxruntime infers them
                out.append(raw)
        return _encode(MODEL_GRAPH, b"".join(bytes(x) for x in out))

    header = b"".join(bytes(raw) for f, _, raw, _ in top if f not in (MODEL_GRAPH, MODEL_METADATA))
    return Split(
        encoder=header + part(encoder, [bytes(p) for p in declared[GRAPH_INPUT]], [embedding]),
        head=header + part(head, [embedding], [bytes(p) for p in declared[GRAPH_OUTPUT]]),
    )
