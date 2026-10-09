// The augmenter of core/augment.py, for one signal at a time: level, reverb, microphone EQ,
// noise or babble at a random SNR, and a noise floor. Same probabilities and ranges.
//
// One deliberate difference: when a signal gets both reverb and EQ, they're applied in one
// pass in the frequency domain, and the reverb's loudness match happens after the EQ. The
// level is re-randomized anyway, so the distribution of what the encoder sees is the same.

import { nextPow2, realFft, type Spectrum } from "./fft";
import { Rng } from "./rng";

export interface AugmentConfig {
  levelDbfs: [number, number];
  pReverb: number;
  pNoise: number;
  snrDb: [number, number];
  pBabble: number;
  pEq: number;
  eqDb: number;
  floorDbfs: [number, number];
}

export const AUGMENT: AugmentConfig = {
  levelDbfs: [-40, -15],
  pReverb: 0.5,
  pNoise: 0.8,
  snrDb: [0, 25],
  pBabble: 0.2,
  pEq: 0.5,
  eqDb: 6,
  floorDbfs: [-85, -60],
};

export interface AugmentBanks {
  noise: Float32Array | Int16Array; // int16 is read as v / 32767 (half the memory per worker)
  babble: Float32Array | Int16Array;
  rirs: Float32Array[]; // aligned and peak-normalized (scripts/export_web_assets.py)
}

const rms = (x: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length + 1e-10);
};

const db = (v: number) => 10 ** (v / 20);

export class Augmenter {
  private spec: Spectrum | null = null;

  constructor(readonly banks: AugmentBanks, readonly config: AugmentConfig = AUGMENT) {}

  /** A new, augmented copy of `input` (clean speech in a window, zeros elsewhere). */
  apply(input: Float32Array, rng: Rng): Float32Array {
    const cfg = this.config;
    const n = input.length;
    const x = new Float32Array(input);

    // Level: speech RMS measured over its active region (the window is mostly padding).
    let active = 0;
    for (let i = 0; i < n; i++) if (Math.abs(x[i]) > 1e-4) active++;
    const activeSqrt = Math.sqrt(Math.max(0.1, active / n));
    const gain = db(rng.uniform(...cfg.levelDbfs)) / (rms(x) / activeSqrt);
    for (let i = 0; i < n; i++) x[i] *= gain;

    const reverb = rng.next() < cfg.pReverb && this.banks.rirs.length > 0;
    const eq = rng.next() < cfg.pEq;
    if (reverb || eq) {
      const rir = reverb ? this.banks.rirs[rng.int(this.banks.rirs.length)] : null;
      const size = nextPow2(n + (rir?.length ?? 0));
      const fft = realFft(size);
      const spec = fft.forward(x);
      if (rir) {
        const h = fft.forward(rir, this.scratch(size));
        for (let k = 0; k < spec.re.length; k++) {
          const r = spec.re[k] * h.re[k] - spec.im[k] * h.im[k];
          spec.im[k] = spec.re[k] * h.im[k] + spec.im[k] * h.re[k];
          spec.re[k] = r;
        }
      }
      if (eq) { // smooth random bumps: 8 knots over 0..Nyquist, linear in between
        const knots = Array.from({ length: 8 }, () => rng.uniform(-cfg.eqDb, cfg.eqDb));
        const bins = spec.re.length;
        for (let k = 0; k < bins; k++) {
          const pos = (k / (bins - 1)) * 7;
          const i = Math.min(6, Math.floor(pos));
          const g = db(knots[i] + (knots[i + 1] - knots[i]) * (pos - i));
          spec.re[k] *= g;
          spec.im[k] *= g;
        }
      }
      const before = rms(x);
      const y = fft.inverse(spec);
      for (let i = 0; i < n; i++) x[i] = y[i];
      if (rir) {
        const match = before / rms(x);
        for (let i = 0; i < n; i++) x[i] *= match;
      }
    }

    const speechRms = rms(x) / activeSqrt;
    if (rng.next() < cfg.pNoise) {
      const bank = rng.next() < cfg.pBabble && this.banks.babble.length > n ? this.banks.babble : this.banks.noise;
      if (bank.length > n) {
        const start = rng.int(bank.length - n);
        const noise = bank.subarray(start, start + n); // its unit cancels in the SNR scaling
        const scale = speechRms / rms(noise) / db(rng.uniform(...cfg.snrDb));
        for (let i = 0; i < n; i++) x[i] += noise[i] * scale;
      }
    }

    const floor = db(rng.uniform(...cfg.floorDbfs));
    for (let i = 0; i < n; i++) x[i] = Math.max(-1, Math.min(1, x[i] + rng.normal() * floor));
    return x;
  }

  private scratch(size: number): Spectrum {
    if (!this.spec || this.spec.re.length !== size / 2 + 1) {
      this.spec = { re: new Float64Array(size / 2 + 1), im: new Float64Array(size / 2 + 1) };
    }
    return this.spec;
  }
}
