// Upload checks, ported from server/audio.py: cut takes with the runtime VAD, normalize
// loudness, reject unusable clips, flag outliers. Same thresholds and messages.

import { SileroVad, VAD_FRAME } from "@wwd/runtime";

import { speechRegions } from "../audio/segments";

export const SR = 16000;
const WINDOW_S = 1.5;
const MAX_WORD_S = 1.3; // speech longer than this can't sit inside the 1.5 s window with margins
const MIN_WORD_S = 0.2;
const PAD_S = 0.2; // kept around the VAD's speech region (its onset lags by about a frame)
const FADE_S = 0.01;
const TARGET_DBFS = -20;
const PEAK_LIMIT = 10 ** (-1 / 20);
const CLIP_LEVEL = 0.999;
const CLIP_FRACTION = 1e-3;
const QUIET_DBFS = -55;
const SILENT_DBFS = -70;
const MIN_OTHER_SPEECH_S = 2;
const OUTLIER_FLOOR = 0.5;
const OUTLIER_DROP = 0.2;
export const MAX_UPLOAD_S = 180;

export type Reason = "too_long" | "too_short" | "clipped" | "too_quiet";

/** The whole upload is unusable (the server answers 422). */
export class UploadError extends Error {}

export interface Clip {
  audio: Float32Array;
  startS: number;
  speechS: number;
  accepted: boolean;
  reason: Reason | null;
}

export const dbfs = (v: number) => 20 * Math.log10(Math.max(v, 1e-10));

export function rms(x: Float32Array): number {
  if (!x.length) return 0;
  let s = 0;
  for (let i = 0; i < x.length; i++) s += x[i] * x[i];
  return Math.sqrt(s / x.length);
}

export function peak(x: Float32Array): number {
  let m = 0;
  for (let i = 0; i < x.length; i++) m = Math.max(m, Math.abs(x[i]));
  return m;
}

function fade(x: Float32Array): Float32Array {
  const n = Math.min(Math.floor(FADE_S * SR), Math.floor(x.length / 2));
  const out = new Float32Array(x);
  for (let i = 0; i < n; i++) {
    const ramp = 0.5 - 0.5 * Math.cos((Math.PI * i) / Math.max(1, n - 1));
    out[i] *= ramp;
    out[x.length - 1 - i] *= ramp;
  }
  return out;
}

function normalize(x: Float32Array, speech: Float32Array): Float32Array {
  let gain = 10 ** (TARGET_DBFS / 20) / Math.max(rms(speech), 1e-6);
  gain = Math.min(gain, PEAK_LIMIT / Math.max(peak(x), 1e-6));
  return x.map((v) => v * gain);
}

export async function vadProbs(audio: Float32Array, vad: SileroVad): Promise<Float32Array> {
  vad.reset();
  const n = Math.floor(audio.length / VAD_FRAME);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = await vad.process(audio.subarray(i * VAD_FRAME, (i + 1) * VAD_FRAME));
  return out;
}

/** One take holding several utterances of the wake word -> one clip per utterance. */
export async function splitPositives(audio: Float32Array, vad: SileroVad): Promise<Clip[]> {
  if (dbfs(peak(audio)) < SILENT_DBFS) throw new UploadError("the recording is silent");
  const regions = speechRegions(await vadProbs(audio, vad));
  if (!regions.length) throw new UploadError("no speech detected");
  return regions.map(([a, b], k) => {
    const speechS = (b - a) / SR;
    const pad = Math.floor(Math.max(0.05, Math.min(PAD_S, (WINDOW_S - speechS) / 2)) * SR);
    const lo = Math.max(a - pad, k ? Math.floor((regions[k - 1][1] + a) / 2) : 0);
    const hi = Math.min(b + pad, k + 1 < regions.length ? Math.floor((b + regions[k + 1][0]) / 2) : audio.length);
    let x: Float32Array = audio.slice(lo, hi);
    const speech = audio.subarray(a, b);
    let clipped = 0;
    for (let i = 0; i < x.length; i++) if (Math.abs(x[i]) >= CLIP_LEVEL) clipped++;
    const reason: Reason | null = speechS > MAX_WORD_S ? "too_long"
      : speechS < MIN_WORD_S ? "too_short"
      : clipped / x.length >= CLIP_FRACTION ? "clipped"
      : dbfs(rms(speech)) < QUIET_DBFS ? "too_quiet" : null;
    if (reason === null) x = fade(normalize(x, speech));
    return { audio: x, startS: lo / SR, speechS, accepted: reason === null, reason };
  });
}

/** Other speech (a paragraph read aloud) or room ambience: kept whole, as negatives. */
export async function prepareOther(audio: Float32Array, kind: "other_speech" | "ambience",
                                   vad: SileroVad): Promise<Clip> {
  const p = dbfs(peak(audio));
  if (kind === "ambience") {
    if (p < -90) throw new UploadError("the recording is digital silence");
    return { audio, startS: 0, speechS: 0, accepted: true, reason: null };
  }
  if (p < SILENT_DBFS) throw new UploadError("the recording is silent");
  const regions = speechRegions(await vadProbs(audio, vad));
  const speechS = regions.reduce((s, [a, b]) => s + b - a, 0) / SR;
  if (speechS < MIN_OTHER_SPEECH_S) throw new UploadError(`only ${speechS.toFixed(1)} s of speech detected`);
  const speech = new Float32Array(regions.reduce((s, [a, b]) => s + b - a, 0));
  let off = 0;
  for (const [a, b] of regions) {
    speech.set(audio.subarray(a, b), off);
    off += b - a;
  }
  return { audio: normalize(audio, speech), startS: 0, speechS, accepted: true, reason: null };
}

/** Leave-one-out cosine of each recording to the others' prototype, and outlier flags. */
export function outliers(embeddings: Float32Array[]): { sims: number[]; flags: boolean[] } {
  const e = embeddings.map((v) => {
    const n = Math.hypot(...v);
    return v.map((x) => x / n);
  });
  if (e.length < 2) return { sims: e.map(() => 1), flags: e.map(() => false) };
  const dim = e[0].length;
  const sum = new Float64Array(dim);
  for (const v of e) for (let i = 0; i < dim; i++) sum[i] += v[i];
  const sims = e.map((v) => {
    const others = Array.from(sum, (s, i) => s - v[i]);
    const norm = Math.hypot(...others);
    return others.reduce((acc, o, i) => acc + v[i] * o, 0) / norm;
  });
  const sorted = [...sims].sort((a, b) => a - b);
  const mid = sorted.length / 2;
  const median = sorted.length % 2 ? sorted[Math.floor(mid)] : (sorted[mid - 1] + sorted[mid]) / 2;
  const flags = sims.map((s) => s < OUTLIER_FLOOR || (e.length >= 4 && s < median - OUTLIER_DROP));
  return { sims, flags };
}

/** server/audio.py `center_window`. */
export function centerWindow(clip: Float32Array, window = WINDOW_S * SR): Float32Array {
  const c = clip.subarray(0, window);
  const out = new Float32Array(window);
  out.set(c, Math.floor((window - c.length) / 2));
  return out;
}
