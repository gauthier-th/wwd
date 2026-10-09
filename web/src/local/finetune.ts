// Quick mode's maths, ported from core/finetune.py, core/calibrate.py and server/quick.py:
// training windows, negative-bank selection, the pushed-prototype head, threshold calibration
// on long-form ticks, and the leave-2-out recall estimate. Each function names its original.

import { Rng } from "./rng";

export const WINDOW = 24000;
export const HOP = 1536;
export const SHIFTS = 6; // consecutive ticks during which a 1 s word sits fully in the window

export interface FinetuneConfig {
  augmentPerClip: number;
  partialPerClip: number;
  speed: [number, number];
  partialFraction: [number, number];
  hardNegatives: number;
  randomNegatives: number;
  sameLanguageShare: number;
  excludeSimilarity: number;
  userHopS: number;
  rocchio: number;
  scoreScale: number;
  seed: number;
}

export const FINETUNE: FinetuneConfig = {
  augmentPerClip: 300,
  partialPerClip: 40,
  speed: [0.9, 1.1],
  partialFraction: [0.4, 0.7],
  hardNegatives: 3000,
  randomNegatives: 30000,
  sameLanguageShare: 0.7,
  excludeSimilarity: 0.8,
  userHopS: 0.5,
  rocchio: 0.5,
  scoreScale: 20,
  seed: 0,
};

/** Detector settings the calibration replays (runtime/SPEC.md §3). */
export interface TriggerConfig {
  k: number;
  n: number;
  refractoryTicks: number;
}

// ------------------------------------------------------------------ embeddings as rows

/** Row-major [rows, dim] float32. */
export interface Rows {
  data: Float32Array;
  rows: number;
  dim: number;
}

/** int8 rows with one scale each (scripts/export_web_assets.py `quantize`). */
export interface QuantizedRows {
  q: Int8Array;
  scale: Float32Array;
  rows: number;
  dim: number;
}

export const row = (m: Rows, i: number) => m.data.subarray(i * m.dim, (i + 1) * m.dim);

export function rowsOf(list: Float32Array[], dim: number): Rows {
  const data = new Float32Array(list.length * dim);
  list.forEach((r, i) => data.set(r, i * dim));
  return { data, rows: list.length, dim };
}

export function concatRows(parts: Rows[]): Rows {
  const dim = parts[0].dim;
  const data = new Float32Array(parts.reduce((n, p) => n + p.data.length, 0));
  let off = 0;
  for (const p of parts) {
    data.set(p.data, off);
    off += p.data.length;
  }
  return { data, rows: data.length / dim, dim };
}

export function unit(v: Float32Array): Float32Array {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  const inv = 1 / Math.sqrt(s);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] * inv;
  return out;
}

/** Each row's dot product with v. Rows from the encoder are unit vectors: cosines. */
export function dots(m: Rows, v: Float32Array): Float64Array {
  const out = new Float64Array(m.rows);
  for (let r = 0, o = 0; r < m.rows; r++, o += m.dim) {
    let s = 0;
    for (let i = 0; i < m.dim; i++) s += m.data[o + i] * v[i];
    out[r] = s;
  }
  return out;
}

export function quantizedDots(m: QuantizedRows, v: Float32Array, pick?: ArrayLike<number>): Float64Array {
  const n = pick ? pick.length : m.rows;
  const out = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const r = pick ? pick[j] : j;
    const o = r * m.dim;
    let s = 0;
    for (let i = 0; i < m.dim; i++) s += m.q[o + i] * v[i];
    out[j] = s * m.scale[r];
  }
  return out;
}

export function dequantize(m: QuantizedRows, pick: ArrayLike<number>): Rows {
  const data = new Float32Array(pick.length * m.dim);
  for (let j = 0; j < pick.length; j++) {
    const r = pick[j];
    const s = m.scale[r];
    for (let i = 0; i < m.dim; i++) data[j * m.dim + i] = m.q[r * m.dim + i] * s;
  }
  return { data, rows: pick.length, dim: m.dim };
}

function normalizeRows(m: Rows): Rows {
  const data = new Float32Array(m.data.length);
  for (let r = 0; r < m.rows; r++) data.set(unit(row(m, r)), r * m.dim);
  return { data, rows: m.rows, dim: m.dim };
}

function meanRow(m: Rows, pick?: ArrayLike<number>): Float32Array {
  const out = new Float32Array(m.dim);
  const n = pick ? pick.length : m.rows;
  for (let j = 0; j < n; j++) {
    const o = (pick ? pick[j] : j) * m.dim;
    for (let i = 0; i < m.dim; i++) out[i] += m.data[o + i];
  }
  for (let i = 0; i < m.dim; i++) out[i] /= n;
  return out;
}

/** numpy.quantile with linear interpolation. */
export function quantile(values: ArrayLike<number>, q: number): number {
  const sorted = Float64Array.from(values).sort();
  const pos = q * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(sorted.length - 1, lo + 1);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** Indices of the k largest values, largest first. */
export function topK(values: ArrayLike<number>, k: number, among?: ArrayLike<number>): number[] {
  const idx = among ? Array.from(among) : Array.from({ length: values.length }, (_, i) => i);
  idx.sort((a, b) => values[b] - values[a]);
  return idx.slice(0, k);
}

// ------------------------------------------------------------------ windows

/** core/finetune.py `_speed`: linear interpolation, align_corners=True. */
export function speed(clip: Float32Array, factor: number): Float32Array {
  const n = Math.max(1, Math.round(clip.length / factor));
  const out = new Float32Array(n);
  if (n === 1 || clip.length === 1) {
    out[0] = clip[0];
    return out;
  }
  const ratio = (clip.length - 1) / (n - 1);
  for (let i = 0; i < n; i++) {
    const pos = i * ratio;
    const j = Math.min(clip.length - 2, Math.floor(pos));
    const f = pos - j;
    out[i] = clip[j] * (1 - f) + clip[j + 1] * f;
  }
  return out;
}

/** core/finetune.py `_place`: into a WINDOW of zeros, centered or at a random position. */
export function place(clip: Float32Array, rng: Rng | null): Float32Array {
  const c = clip.subarray(0, WINDOW);
  const out = new Float32Array(WINDOW);
  const start = rng ? rng.int(WINDOW - c.length + 1) : Math.floor((WINDOW - c.length) / 2);
  out.set(c, start);
  return out;
}

/** Positive window i: recording floor(i / n), at a random speed and position. */
export function positiveWindow(clips: Float32Array[], n: number, cfg: FinetuneConfig, seed: number) {
  return (i: number): Float32Array => {
    const rng = Rng.derive(seed, i);
    return place(speed(clips[Math.floor(i / n)], rng.uniform(...cfg.speed)), rng);
  };
}

/** Partial window i: the first or last 40-70% of recording floor(i / n), which must not
 * trigger (no early firing). */
export function partialWindow(clips: Float32Array[], n: number, cfg: FinetuneConfig, seed: number) {
  return (i: number): Float32Array => {
    const rng = Rng.derive(seed, i);
    const c = clips[Math.floor(i / n)];
    const keep = Math.max(1, Math.floor(c.length * rng.uniform(...cfg.partialFraction)));
    const part = rng.next() < 0.5 ? c.subarray(0, keep) : c.subarray(c.length - keep);
    return place(speed(part, rng.uniform(...cfg.speed)), rng);
  };
}

/** core/finetune.py `tick_windows`, as streams: each clip ends where the first of SHIFTS
 * consecutive tick windows ends. Cut them with `streamWindows` (after augmenting, if any). */
export function tickStreams(clips: Float32Array[], copies: number): Float32Array[] {
  const length = WINDOW + (SHIFTS - 1) * HOP;
  const out: Float32Array[] = [];
  for (const clip of clips) {
    const c = clip.subarray(Math.max(0, clip.length - WINDOW));
    for (let i = 0; i < copies; i++) {
      const s = new Float32Array(length);
      s.set(c, WINDOW - c.length);
      out.push(s);
    }
  }
  return out;
}

/** Windows of a signal every `hop` samples (one window for a WINDOW-long signal). */
export function streamWindows(signal: Float32Array, hop: number): Float32Array[] {
  const out: Float32Array[] = [];
  for (let s = 0; s + WINDOW <= signal.length; s += hop) out.push(signal.subarray(s, s + WINDOW));
  return out;
}

/** core/finetune.py `sliding_windows` (user negatives). */
export function slidingWindows(audio: Float32Array[], hopS: number): Float32Array[] {
  const out: Float32Array[] = [];
  for (const a of audio) {
    let x = a;
    if (x.length < WINDOW) {
      x = new Float32Array(WINDOW);
      x.set(a);
    }
    for (const w of streamWindows(x, Math.round(hopS * 16000))) out.push(new Float32Array(w));
  }
  return out;
}

// ------------------------------------------------------------------ bank and head

export interface Bank {
  rows: QuantizedRows;
  word: Uint32Array; // index into `words`
  lang: Uint8Array; // index into `languages`
  words: string[]; // normalized (NFC, casefolded)
  languages: string[];
}

/** core/finetune.py `normalize_word`: NFC + casefold (lowercase, and ß -> ss as Python does). */
export function normalizeWord(word: string): string {
  return word.normalize("NFC").toLowerCase().replaceAll("ß", "ss").trim();
}

export interface Negatives {
  bank: number[]; // bank rows: the hardest first, then random ones
  info: Record<string, number | null>;
}

/** core/finetune.py `select_bank`: bank negatives without the wake word itself. */
export function selectBank(bank: Bank, prototype: Float32Array, wakeWord: string, language: string,
                           cfg: FinetuneConfig, rng: Rng): Negatives {
  const sims = quantizedDots(bank.rows, prototype);
  const target = bank.words.indexOf(normalizeWord(wakeWord));
  const lang = bank.languages.indexOf(language);
  const keep: number[] = [];
  let sameWord = 0;
  let tooSimilar = 0;
  for (let r = 0; r < bank.rows.rows; r++) {
    if (bank.word[r] === target) sameWord++;
    else if (sims[r] >= cfg.excludeSimilarity) tooSimilar++;
    else keep.push(r);
  }
  const hard = topK(sims, cfg.hardNegatives, keep);
  const sameLang = keep.filter((r) => bank.lang[r] === lang);
  const otherLang = keep.filter((r) => bank.lang[r] !== lang);
  const nSame = Math.min(sameLang.length, Math.floor(cfg.randomNegatives * cfg.sameLanguageShare));
  const nOther = Math.min(otherLang.length, cfg.randomNegatives - nSame);
  const hardSet = new Set(hard);
  const rand = [...rng.sample(sameLang, nSame), ...rng.sample(otherLang, nOther)].filter((r) => !hardSet.has(r));
  return {
    bank: [...hard, ...rand],
    info: { bank_same_word: sameWord, bank_too_similar: tooSimilar, bank_hard: hard.length,
            bank_random: rand.length, hardest_similarity: hard.length ? sims[hard[0]] : null },
  };
}

/** logit = scale * (direction . embedding) + bias: what PrototypeHead exports as a Gemm. */
export interface Head {
  direction: Float32Array; // unit
  scale: number;
  bias: number;
}

export const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** core/finetune.py `prototype_head`: the prototype pushed away from the hardest negatives;
 * the bias puts the sigmoid midpoint between typical positives and the hardest negatives. */
export function prototypeHead(proto: Float32Array, pos: Rows, neg: Rows, cfg: FinetuneConfig): Head {
  const n = normalizeRows(neg);
  const hard = topK(dots(n, proto), cfg.hardNegatives);
  const pushed = unit(meanRow(n, hard));
  const d = new Float32Array(proto.length);
  for (let i = 0; i < d.length; i++) d[i] = proto[i] - cfg.rocchio * pushed[i];
  const direction = unit(d);
  const posCos = quantile(dots(normalizeRows(pos), direction), 0.5);
  const negCos = quantile(dots(n, direction), 0.999);
  const midpoint = (posCos + negCos) / 2;
  return { direction, scale: cfg.scoreScale, bias: -cfg.scoreScale * midpoint };
}

export interface EnrollmentEmbeddings {
  clean: Rows; // [C, D] each recording centered, unaugmented
  positives: Rows; // [C * A, D] augmented
  partial: Rows; // [C * P, D]
  user: Rows; // [U, D] the user's other speech and ambience
}

export interface FitResult {
  head: Head;
  prototype: Float32Array;
  info: Record<string, number | string | null>;
}

/** core/finetune.py `fit_head` on `recordings` (default: all). */
export function fitHead(emb: EnrollmentEmbeddings, bank: Bank, wakeWord: string, language: string,
                        cfg: FinetuneConfig, recordings?: number[]): FitResult {
  const rng = new Rng(cfg.seed + 1);
  const n = emb.clean.rows;
  const use = recordings ?? Array.from({ length: n }, (_, i) => i);
  const prototype = unit(meanRow(emb.clean, use));
  const per = emb.positives.rows / n;
  const perPartial = emb.partial.rows / n;
  const pick = (m: Rows, size: number): Rows => {
    const idx = use.flatMap((c) => Array.from({ length: size }, (_, i) => c * size + i));
    const data = new Float32Array(idx.length * m.dim);
    idx.forEach((r, j) => data.set(row(m, r), j * m.dim));
    return { data, rows: idx.length, dim: m.dim };
  };
  const pos = pick(emb.positives, per);
  const partial = pick(emb.partial, perPartial);
  const neg = selectBank(bank, prototype, wakeWord, language, cfg, rng);
  const negatives = concatRows([dequantize(bank.rows, neg.bank), partial, emb.user].filter((m) => m.rows > 0));
  const head = prototypeHead(prototype, pos, negatives, cfg);
  return { head, prototype, info: { ...neg.info, scorer: "prototype", positives: pos.rows,
                                    partial_negatives: partial.rows, user_negatives: emb.user.rows } };
}

/** Scores (probabilities, as the exported model computes them) of quantized rows. */
export function headScores(head: Head, rows: QuantizedRows | Rows): Float32Array {
  const cos = "q" in rows ? quantizedDots(rows, head.direction) : dots(rows, head.direction);
  const out = new Float32Array(cos.length);
  for (let i = 0; i < cos.length; i++) out[i] = sigmoid(head.scale * cos[i] + head.bias);
  return out;
}

// ------------------------------------------------------------------ calibration (core/calibrate.py)

/** Triggers the runtime would fire, each stream starting fresh (`count_triggers`). */
export function countTriggers(scores: Float32Array, streamStarts: ArrayLike<number>, threshold: number,
                              cfg: TriggerConfig): number {
  let total = 0;
  for (let s = 0; s + 1 < streamStarts.length; s++) {
    const a = streamStarts[s];
    const b = streamStarts[s + 1];
    let nextAllowed = 0;
    for (let t = 0; t < b - a; t++) {
      let hits = 0;
      for (let j = Math.max(0, t - cfg.n + 1); j <= t; j++) if (scores[a + j] >= threshold) hits++;
      if (hits >= cfg.k && t >= nextAllowed) {
        total++;
        nextAllowed = t + cfg.refractoryTicks + 1;
      }
    }
  }
  return total;
}

/** Lowest threshold whose false-accept rate is at most the target -> [threshold, FA/h]. */
export function calibrate(scores: Float32Array, streamStarts: ArrayLike<number>, hours: number,
                          target: number, cfg: TriggerConfig, ignore?: Uint8Array): [number, number] {
  let s = scores;
  if (ignore) {
    s = new Float32Array(scores);
    for (let i = 0; i < s.length; i++) if (ignore[i]) s[i] = 0;
  }
  const fa = (thr: number) => countTriggers(s, streamStarts, thr, cfg) / hours;
  let lo = 0;
  let hi = 1;
  if (fa(hi - 1e-6) > target) return [1, fa(1)];
  for (let i = 0; i < 30; i++) {
    const mid = (lo + hi) / 2;
    if (fa(mid) <= target) hi = mid;
    else lo = mid;
  }
  return [hi, fa(hi)];
}

/** `stream_recall`: clips scored at SHIFTS consecutive ticks ([P * shifts]); a clip is
 * detected when some run of n consecutive ticks holds k hits. */
export function streamRecall(scores: Float32Array, shifts: number, threshold: number, cfg: TriggerConfig): number {
  const clips = scores.length / shifts;
  let detected = 0;
  for (let c = 0; c < clips; c++) {
    let best = 0;
    for (let i = 0; i + cfg.n <= shifts; i++) {
      let hits = 0;
      for (let j = i; j < i + cfg.n; j++) if (scores[c * shifts + j] >= threshold) hits++;
      best = Math.max(best, hits);
    }
    if (best >= cfg.k) detected++;
  }
  return detected / clips;
}

export interface Cache {
  rows: QuantizedRows;
  streamStarts: Uint32Array;
  occWord: Uint32Array; // index into occWords
  occTicks: Uint32Array; // [O, 2] half-open tick ranges
  occWords: string[]; // normalized
  hours: number;
}

/** `LongformCache.mask_word`: ticks whose window holds the wake word (not false accepts). */
export function maskWord(cache: Cache, word: string): Uint8Array {
  const mask = new Uint8Array(cache.rows.rows);
  const target = cache.occWords.indexOf(normalizeWord(word));
  if (target < 0) return mask;
  for (let o = 0; o < cache.occWord.length; o++) {
    if (cache.occWord[o] === target) mask.fill(1, cache.occTicks[2 * o], cache.occTicks[2 * o + 1]);
  }
  return mask;
}

/** server/quick.py `folds_of`: leave-2-out, or leave-1-out below 5 recordings. */
export function foldsOf(n: number): number[][] {
  const size = n < 5 ? 1 : 2;
  const out: number[][] = [];
  for (let i = 0; i < n; i += size) out.push(Array.from({ length: Math.min(size, n - i) }, (_, j) => i + j));
  return out;
}

export interface CrossValidation {
  folds: number;
  recall_noisy: number;
  recall_clean: number;
  per_recording: { sample_id: string; recall_noisy: number; recall_clean: number; weak: boolean }[];
}

/** server/quick.py `cross_validate`. noisy: [C * copies * SHIFTS] rows, clean: [C * SHIFTS]. */
export function crossValidate(emb: EnrollmentEmbeddings, noisy: Rows, clean: Rows, ids: string[], bank: Bank,
                              cache: Cache, wakeWord: string, language: string, target: number,
                              ignore: Uint8Array, cfg: FinetuneConfig, trigger: TriggerConfig): CrossValidation {
  const n = emb.clean.rows;
  const perNoisy = noisy.rows / n;
  const perClean = clean.rows / n;
  const recallNoisy = new Float64Array(n);
  const recallClean = new Float64Array(n);
  const folds = foldsOf(n);
  const slice = (m: Rows, i: number, per: number): Rows =>
    ({ data: m.data.subarray(i * per * m.dim, (i + 1) * per * m.dim), rows: per, dim: m.dim });
  for (const held of folds) {
    const train = Array.from({ length: n }, (_, i) => i).filter((i) => !held.includes(i));
    const res = fitHead(emb, bank, wakeWord, language, cfg, train);
    const [thr] = calibrate(headScores(res.head, cache.rows), cache.streamStarts, cache.hours, target, trigger, ignore);
    for (const i of held) {
      recallNoisy[i] = streamRecall(headScores(res.head, slice(noisy, i, perNoisy)), SHIFTS, thr, trigger);
      recallClean[i] = streamRecall(headScores(res.head, slice(clean, i, perClean)), SHIFTS, thr, trigger);
    }
  }
  const median = quantile(recallNoisy, 0.5);
  return {
    folds: folds.length,
    recall_noisy: recallNoisy.reduce((a, b) => a + b, 0) / n,
    recall_clean: recallClean.reduce((a, b) => a + b, 0) / n,
    per_recording: ids.map((sample_id, i) => ({
      sample_id, recall_noisy: recallNoisy[i], recall_clean: recallClean[i],
      weak: recallNoisy[i] < 0.5 && recallNoisy[i] < median - 0.25,
    })),
  };
}
