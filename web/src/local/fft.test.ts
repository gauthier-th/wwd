import { describe, expect, it } from "vitest";

import { RealFft } from "./fft";

function dft(x: number[]): { re: number[]; im: number[] } {
  const n = x.length;
  const re: number[] = [];
  const im: number[] = [];
  for (let k = 0; k <= n / 2; k++) {
    let r = 0;
    let i = 0;
    for (let t = 0; t < n; t++) {
      r += x[t] * Math.cos((-2 * Math.PI * k * t) / n);
      i += x[t] * Math.sin((-2 * Math.PI * k * t) / n);
    }
    re.push(r);
    im.push(i);
  }
  return { re, im };
}

describe("RealFft", () => {
  it("matches a direct DFT", () => {
    const x = Array.from({ length: 64 }, (_, i) => Math.sin(i * 0.3) + ((i * 7) % 5) / 5);
    const spec = new RealFft(64).forward(x);
    const ref = dft(x);
    for (let k = 0; k <= 32; k++) {
      expect(spec.re[k]).toBeCloseTo(ref.re[k], 9);
      expect(spec.im[k]).toBeCloseTo(ref.im[k], 9);
    }
  });

  it("inverts, with zero padding", () => {
    const x = Float32Array.from({ length: 1000 }, (_, i) => Math.cos(i / 17) * (i % 13));
    const fft = new RealFft(1024);
    const y = fft.inverse(fft.forward(x));
    for (let i = 0; i < 1024; i++) expect(y[i]).toBeCloseTo(i < 1000 ? x[i] : 0, 9);
  });
});
