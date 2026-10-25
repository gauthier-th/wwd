// Reads ModelProto.metadata_props straight from the ONNX protobuf bytes. onnxruntime-web does
// not expose custom metadata, and this keeps a model file self-describing in every runtime.

const METADATA_PROPS = 14; // ModelProto field number
const KEY = 1;
const VALUE = 2;

export class Reader {
  pos = 0;
  constructor(private readonly buf: Uint8Array) {}

  get done(): boolean {
    return this.pos >= this.buf.length;
  }

  varint(): number {
    let result = 0;
    let scale = 1;
    for (;;) {
      if (this.pos >= this.buf.length) throw new Error("truncated varint");
      const byte = this.buf[this.pos++];
      result += (byte & 0x7f) * scale; // multiply, not shift: lengths can exceed 2^31
      if (!(byte & 0x80)) return result;
      scale *= 128;
    }
  }

  bytes(): Uint8Array {
    const len = this.varint();
    const out = this.buf.subarray(this.pos, this.pos + len);
    if (out.length !== len) throw new Error("truncated field");
    this.pos += len;
    return out;
  }

  skip(wireType: number): void {
    if (wireType === 0) this.varint();
    else if (wireType === 1) this.pos += 8;
    else if (wireType === 2) {
      // Not `this.pos += this.varint()`: that reads pos before varint() advances it.
      const len = this.varint();
      this.pos += len;
    } else if (wireType === 5) this.pos += 4;
    else throw new Error(`unsupported protobuf wire type ${wireType}`);
  }
}

const utf8 = new TextDecoder();

export function readOnnxMetadata(model: Uint8Array): Record<string, string> {
  const meta: Record<string, string> = {};
  const top = new Reader(model);
  while (!top.done) {
    const tag = top.varint();
    if (Math.floor(tag / 8) !== METADATA_PROPS || tag % 8 !== 2) {
      top.skip(tag % 8);
      continue;
    }
    const entry = new Reader(top.bytes());
    let key = "";
    let value = "";
    while (!entry.done) {
      const t = entry.varint();
      const field = Math.floor(t / 8);
      if (field === KEY && t % 8 === 2) key = utf8.decode(entry.bytes());
      else if (field === VALUE && t % 8 === 2) value = utf8.decode(entry.bytes());
      else entry.skip(t % 8);
    }
    meta[key] = value;
  }
  return meta;
}
