// Splits a wake-word file into its encoder and its head (runtime/SPEC.md §5), straight from the
// protobuf bytes like onnxMeta.ts. Mirrors runtime/onnxfile.py: fields a split doesn't change
// are copied as they are, so files that share an encoder give byte-identical encoder parts.

import { Reader } from "./onnxMeta.js";

export const EMBEDDING = "embedding"; // the encoder's output, which the head reads

// Field numbers from onnx.proto
const MODEL_GRAPH = 7;
const MODEL_METADATA = 14;
const GRAPH_NODE = 1;
const GRAPH_INITIALIZER = 5;
const GRAPH_INPUT = 11;
const GRAPH_OUTPUT = 12;
const GRAPH_VALUE_INFO = 13;
const GRAPH_SPARSE_INITIALIZER = 15;
const NODE_INPUT = 1;
const NODE_OUTPUT = 2;
const NODE_ATTRIBUTE = 5;
const ATTRIBUTE_GRAPH = 6;
const ATTRIBUTE_GRAPHS = 11;
const TENSOR_NAME = 8;
const VALUE_INFO_NAME = 1;
const LEN = 2; // wire type of strings and messages

interface Field {
  field: number;
  wire: number;
  raw: Uint8Array; // the whole field
  payload: Uint8Array | null; // if length-delimited
}

function fields(buf: Uint8Array): Field[] {
  const r = new Reader(buf);
  const out: Field[] = [];
  while (!r.done) {
    const start = r.pos;
    const tag = r.varint();
    const wire = tag % 8;
    let payload: Uint8Array | null = null;
    if (wire === LEN) payload = r.bytes();
    else r.skip(wire);
    if (r.pos > buf.length) throw new Error("truncated protobuf");
    out.push({ field: Math.floor(tag / 8), wire, raw: buf.subarray(start, r.pos), payload });
  }
  return out;
}

const utf8 = new TextDecoder("utf-8", { fatal: true });

const strings = (msg: Uint8Array, field: number) =>
  fields(msg).flatMap((f) => (f.field === field && f.payload ? [utf8.decode(f.payload)] : []));

const nameOf = (msg: Uint8Array, field: number) => strings(msg, field)[0] ?? "";

function varint(n: number): number[] {
  const out: number[] = [];
  while (n > 0x7f) {
    out.push((n % 128) | 0x80);
    n = Math.floor(n / 128);
  }
  out.push(n);
  return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}

const encode = (field: number, payload: Uint8Array) =>
  concat([Uint8Array.from([...varint((field << 3) | LEN), ...varint(payload.length)]), payload]);

/** Two ModelProtos, without the file's metadata. */
export interface Split {
  encoder: Uint8Array; // audio -> embedding
  head: Uint8Array; // embedding -> score
}

/**
 * The encoder and head of a wake-word file, or null when no `embedding` tensor separates them:
 * the file then runs whole. Never throws: a broken file fails in onnxruntime instead.
 */
export function splitModel(model: Uint8Array): Split | null {
  try {
    return split(model);
  } catch {
    return null;
  }
}

function split(model: Uint8Array): Split | null {
  const top = fields(model);
  const graphs = top.filter((f) => f.field === MODEL_GRAPH && f.payload);
  if (graphs.length !== 1) return null;
  const graph = fields(graphs[0].payload!);
  if (graph.some((f) => f.field === GRAPH_SPARSE_INITIALIZER)) return null;
  const reads: string[][] = [];
  const writes: string[][] = [];
  for (const f of graph) {
    if (f.field !== GRAPH_NODE || !f.payload) continue;
    const node = fields(f.payload);
    // A subgraph can read outer tensors without listing them as inputs: don't split.
    const subgraph = (a: Field) =>
      a.field === NODE_ATTRIBUTE && fields(a.payload!).some((g) => g.field === ATTRIBUTE_GRAPH || g.field === ATTRIBUTE_GRAPHS);
    if (node.some(subgraph)) return null;
    reads.push(node.flatMap((n) => (n.field === NODE_INPUT && n.payload ? [utf8.decode(n.payload)] : [])));
    writes.push(node.flatMap((n) => (n.field === NODE_OUTPUT && n.payload ? [utf8.decode(n.payload)] : [])));
  }
  const producer = new Map<string, number>();
  writes.forEach((names, i) => names.forEach((name) => name && producer.set(name, i)));
  const initializers = new Set(graph.filter((f) => f.field === GRAPH_INITIALIZER).map((f) => nameOf(f.payload!, TENSOR_NAME)));
  const declared = (field: number) =>
    graph.filter((f) => f.field === field && !initializers.has(nameOf(f.payload!, VALUE_INFO_NAME))).map((f) => f.payload!);
  const inputs = new Set(declared(GRAPH_INPUT).map((p) => nameOf(p, VALUE_INFO_NAME)));
  const outputs = declared(GRAPH_OUTPUT).map((p) => nameOf(p, VALUE_INFO_NAME));
  if (!producer.has(EMBEDDING) || outputs.includes(EMBEDDING)) return null;

  /** Nodes needed to compute `targets`, and the tensors they read from outside. */
  const upstream = (targets: string[], stop: string | null) => {
    const nodes = new Set<number>();
    const leaves = new Set<string>();
    const seen = new Set<string>();
    const todo = [...targets];
    while (todo.length) {
      const name = todo.pop()!;
      if (!name || seen.has(name)) continue;
      seen.add(name);
      const i = producer.get(name);
      if (name === stop || i === undefined) {
        if (!initializers.has(name)) leaves.add(name);
      } else {
        nodes.add(i);
        todo.push(...reads[i]);
      }
    }
    return { nodes, leaves };
  };
  const encoder = upstream([EMBEDDING], null);
  const head = upstream(outputs, EMBEDDING);
  if ([...encoder.leaves].some((name) => !inputs.has(name))) return null;
  if (head.leaves.size !== 1 || !head.leaves.has(EMBEDDING)) return null;
  const declaredEmbedding = graph.find(
    (f) => f.field === GRAPH_VALUE_INFO && nameOf(f.payload!, VALUE_INFO_NAME) === EMBEDDING)?.payload;
  // Undeclared: float [N, D] with both sizes unknown. ValueInfoProto{name, type{tensor_type{
  // elem_type: FLOAT, shape{dim {}, dim {}}}}}
  const embedding = declaredEmbedding ?? concat([
    encode(VALUE_INFO_NAME, new TextEncoder().encode(EMBEDDING)),
    encode(2, encode(1, concat([Uint8Array.of(0x08, 0x01), encode(2, concat([encode(1, new Uint8Array()), encode(1, new Uint8Array())]))]))),
  ]);

  const part = (nodes: Set<number>, newInputs: Uint8Array[], newOutputs: Uint8Array[]) => {
    const used = new Set([...nodes].flatMap((i) => reads[i]));
    const out: Uint8Array[] = [];
    const placed = new Set<number>();
    let index = 0;
    for (const f of graph) {
      if (f.field === GRAPH_NODE) {
        if (nodes.has(index)) out.push(f.raw);
        index++;
      } else if (f.field === GRAPH_INITIALIZER) {
        if (used.has(nameOf(f.payload!, TENSOR_NAME))) out.push(f.raw);
      } else if (f.field === GRAPH_INPUT || f.field === GRAPH_OUTPUT) {
        const name = nameOf(f.payload!, VALUE_INFO_NAME);
        if (initializers.has(name)) { // IR < 4 lists them as inputs too
          if (used.has(name)) out.push(f.raw);
        } else if (!placed.has(f.field)) {
          placed.add(f.field);
          out.push(...(f.field === GRAPH_INPUT ? newInputs : newOutputs).map((v) => encode(f.field, v)));
        }
      } else if (f.field !== GRAPH_VALUE_INFO) { // shapes of tensors that may be gone: onnxruntime infers them
        out.push(f.raw);
      }
    }
    return encode(MODEL_GRAPH, concat(out));
  };

  const header = concat(top.filter((f) => f.field !== MODEL_GRAPH && f.field !== MODEL_METADATA).map((f) => f.raw));
  return {
    encoder: concat([header, part(encoder.nodes, declared(GRAPH_INPUT), [embedding])]),
    head: concat([header, part(head.nodes, [embedding], declared(GRAPH_OUTPUT))]),
  };
}

export function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}
