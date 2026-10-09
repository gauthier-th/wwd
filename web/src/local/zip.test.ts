import { describe, expect, it } from "vitest";

import { Augmenter } from "./augment";
import { Rng } from "./rng";
import { crc32, zip } from "./zip";

describe("zip", () => {
  it("computes the standard CRC-32", () => {
    expect(crc32(new TextEncoder().encode("123456789"))).toBe(0xcbf43926);
  });

  it("writes entries its central directory finds again", () => {
    const files: [string, Uint8Array][] = [["a/b.txt", new TextEncoder().encode("hello")], ["c.bin", new Uint8Array(70000)]];
    const out = zip(files);
    const view = new DataView(out.buffer);
    const end = out.length - 22;
    expect(view.getUint32(end, true)).toBe(0x06054b50);
    expect(view.getUint16(end + 10, true)).toBe(2);
    let pos = view.getUint32(end + 16, true);
    for (const [name, data] of files) {
      expect(view.getUint32(pos, true)).toBe(0x02014b50);
      expect(view.getUint32(pos + 20, true)).toBe(data.length);
      const nameLength = view.getUint16(pos + 28, true);
      expect(new TextDecoder().decode(out.subarray(pos + 46, pos + 46 + nameLength))).toBe(name);
      const local = view.getUint32(pos + 42, true);
      const start = local + 30 + view.getUint16(local + 26, true);
      expect(out.subarray(start, start + data.length)).toEqual(data);
      pos += 46 + nameLength;
    }
  });
});

describe("Augmenter", () => {
  const n = 24000;
  const speech = Float32Array.from({ length: n }, (_, i) => (i > 8000 && i < 16000 ? 0.3 * Math.sin(i / 7) : 0));
  const banks = {
    noise: Float32Array.from({ length: 160000 }, (_, i) => Math.sin(i * 1.7) * 0.2),
    babble: Float32Array.from({ length: 160000 }, (_, i) => Math.sin(i * 0.37) * 0.1),
    rirs: [Float32Array.from({ length: 8000 }, (_, i) => (i === 0 ? 1 : Math.exp(-i / 800) * Math.sin(i) * 0.3))],
  };

  it("is reproducible from its seed and stays in range", () => {
    const aug = new Augmenter(banks);
    const a = aug.apply(speech, new Rng(5));
    expect(aug.apply(speech, new Rng(5))).toEqual(a);
    expect(aug.apply(speech, new Rng(6))).not.toEqual(a);
    expect(a.length).toBe(n);
    for (const v of a) expect(Math.abs(v)).toBeLessThanOrEqual(1);
  });

  it("sets the speech level within the configured range", () => {
    const aug = new Augmenter(banks, { levelDbfs: [-30, -30], pReverb: 0, pNoise: 0, snrDb: [0, 0], pBabble: 0, pEq: 0,
                                       eqDb: 0, floorDbfs: [-200, -200] });
    const out = aug.apply(speech, new Rng(1));
    let s = 0;
    for (let i = 8001; i < 16000; i++) s += out[i] * out[i];
    expect(20 * Math.log10(Math.sqrt(s / 7999))).toBeCloseTo(-30, 0);
  });
});
