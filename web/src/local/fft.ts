// Real FFT for the augmenter (reverb by convolution, microphone EQ): a radix-2 complex FFT
// of half the size, with the usual split for real input. Sizes are powers of two.

class ComplexFft {
  readonly cos: Float64Array;
  readonly sin: Float64Array;
  readonly rev: Uint32Array;

  constructor(readonly n: number) {
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0, x = i; b < bits; b++, x >>= 1) r = (r << 1) | (x & 1);
      this.rev[i] = r;
    }
  }

  /** In place; inverse=true computes the unscaled inverse transform. */
  transform(re: Float64Array, im: Float64Array, inverse: boolean): void {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      const j = this.rev[i];
      if (j > i) {
        let t = re[i];
        re[i] = re[j];
        re[j] = t;
        t = im[i];
        im[i] = im[j];
        im[j] = t;
      }
    }
    const sign = inverse ? 1 : -1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step];
          const wi = sign * this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }
}

/** Spectrum of a real signal: bins 0..n/2 (n/2 + 1 of them). */
export interface Spectrum {
  re: Float64Array;
  im: Float64Array;
}

export class RealFft {
  private readonly half: ComplexFft;
  private readonly zr: Float64Array;
  private readonly zi: Float64Array;
  private readonly twr: Float64Array; // cos(2 pi k / n), k = 0..n/2
  private readonly twi: Float64Array; // sin(2 pi k / n)

  constructor(readonly n: number) {
    if (n < 4 || (n & (n - 1)) !== 0) throw new Error(`FFT size ${n} is not a power of two >= 4`);
    this.half = new ComplexFft(n / 2);
    this.zr = new Float64Array(n / 2);
    this.zi = new Float64Array(n / 2);
    this.twr = new Float64Array(n / 2 + 1);
    this.twi = new Float64Array(n / 2 + 1);
    for (let k = 0; k <= n / 2; k++) {
      this.twr[k] = Math.cos((2 * Math.PI * k) / n);
      this.twi[k] = Math.sin((2 * Math.PI * k) / n);
    }
  }

  /** x (zero-padded to n) -> its n/2 + 1 bins. */
  forward(x: ArrayLike<number>, out?: Spectrum): Spectrum {
    const m = this.n / 2;
    const { zr, zi } = this;
    for (let k = 0; k < m; k++) {
      zr[k] = 2 * k < x.length ? x[2 * k] : 0;
      zi[k] = 2 * k + 1 < x.length ? x[2 * k + 1] : 0;
    }
    this.half.transform(zr, zi, false);
    const re = out?.re ?? new Float64Array(m + 1);
    const im = out?.im ?? new Float64Array(m + 1);
    for (let k = 0; k <= m; k++) {
      const a = k % m;
      const b = (m - k) % m;
      // even part E[k] = (Z[k] + conj Z[m-k]) / 2, odd part O[k] = (Z[k] - conj Z[m-k]) / 2i
      const er = (zr[a] + zr[b]) / 2;
      const ei = (zi[a] - zi[b]) / 2;
      const or = (zi[a] + zi[b]) / 2;
      const oi = -(zr[a] - zr[b]) / 2;
      const wr = this.twr[k];
      const wi = -this.twi[k]; // w^k, w = exp(-2 pi i / n)
      re[k] = er + or * wr - oi * wi;
      im[k] = ei + or * wi + oi * wr;
    }
    return { re, im };
  }

  /** n/2 + 1 bins -> the real signal (scaled, so inverse(forward(x)) == x). */
  inverse(spec: Spectrum, out?: Float64Array): Float64Array {
    const m = this.n / 2;
    const { zr, zi } = this;
    const { re, im } = spec;
    for (let k = 0; k < m; k++) {
      // E[k] = (X[k] + conj X[m-k]) / 2, O[k] = (X[k] - conj X[m-k]) * w^-k / 2; Z = E + iO
      const er = (re[k] + re[m - k]) / 2;
      const ei = (im[k] - im[m - k]) / 2;
      const dr = (re[k] - re[m - k]) / 2;
      const di = (im[k] + im[m - k]) / 2;
      const wr = this.twr[k];
      const wi = this.twi[k]; // w^-k
      const or = dr * wr - di * wi;
      const oi = dr * wi + di * wr;
      zr[k] = er - oi;
      zi[k] = ei + or;
    }
    this.half.transform(zr, zi, true);
    const x = out ?? new Float64Array(this.n);
    for (let k = 0; k < m; k++) {
      x[2 * k] = zr[k] / m;
      x[2 * k + 1] = zi[k] / m;
    }
    return x;
  }
}

const cache = new Map<number, RealFft>();

export function realFft(n: number): RealFft {
  let fft = cache.get(n);
  if (!fft) cache.set(n, (fft = new RealFft(n)));
  return fft;
}

export const nextPow2 = (n: number) => 2 ** Math.ceil(Math.log2(Math.max(4, n)));
