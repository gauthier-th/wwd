// model.onnx without an ONNX library: scripts/export_web_assets.py ships the encoder composed
// with a marker head (core.export.compose_head, graph-checked) and the byte offsets of its
// Gemm weight and bias. Writing the trained head there gives exactly the file the server
// composes. Metadata is appended as ModelProto field 14 entries: protobuf merges repeated
// fields that come after the others, so every ONNX reader sees them as metadata_props.

import type { Head } from "./finetune";

export interface TemplateInfo {
  weight_offset: number;
  bias_offset: number;
  dim: number;
}

const encoder = new TextEncoder();

function varint(n: number): number[] {
  const out: number[] = [];
  while (n > 0x7f) {
    out.push((n & 0x7f) | 0x80);
    n >>>= 7;
  }
  out.push(n);
  return out;
}

function lengthDelimited(field: number, bytes: Uint8Array | number[]): number[] {
  return [(field << 3) | 2, ...varint(bytes.length), ...bytes];
}

/** Serialized `metadata_props` entries (StringStringEntryProto: key = 1, value = 2). */
export function metadataBytes(metadata: Record<string, string>): Uint8Array {
  const out: number[] = [];
  for (const [key, value] of Object.entries(metadata)) {
    const entry = [...lengthDelimited(1, encoder.encode(key)), ...lengthDelimited(2, encoder.encode(value))];
    out.push(...lengthDelimited(14, entry));
  }
  return Uint8Array.from(out);
}

export function composeModel(template: Uint8Array, info: TemplateInfo, head: Head,
                             metadata: Record<string, string>): Uint8Array {
  if (head.direction.length !== info.dim) throw new Error(`head has ${head.direction.length} dims, template ${info.dim}`);
  const meta = metadataBytes(metadata);
  const out = new Uint8Array(template.length + meta.length);
  out.set(template);
  out.set(meta, template.length);
  const view = new DataView(out.buffer);
  for (let i = 0; i < info.dim; i++) {
    view.setFloat32(info.weight_offset + 4 * i, head.direction[i] * head.scale, true);
  }
  view.setFloat32(info.bias_offset, head.bias, true);
  return out;
}
