// The base model's static files (scripts/export_web_assets.py), parsed into what quick mode
// needs. Fetching is abstracted, so the same code runs in the browser (Cache Storage keeps
// the ~200 MB across visits) and under Node (tests, the quality run).

import type { AugmentBanks } from "./augment";
import type { TemplateInfo } from "./compose";
import { FINETUNE, type Bank, type Cache, type FinetuneConfig, type QuantizedRows } from "./finetune";

export interface ArraySpec {
  file: string;
  dtype: string;
  shape: number[];
}

export interface LanguageInfo {
  name: string;
  native_name: string;
  rtl: boolean;
  needs_review: boolean;
  paragraph: string;
}

export interface Manifest {
  format: number;
  version: string;
  created_at: string;
  base_created_at: string;
  embedding_dim: number;
  finetune: Record<string, number>; // core.finetune.FinetuneConfig overrides (snake_case), for tests
  template: TemplateInfo;
  languages: string[];
  bank: { embeddings: ArraySpec; scales: ArraySpec; word: ArraySpec; lang: ArraySpec; words: string };
  cache: { embeddings: ArraySpec; scales: ArraySpec; stream_starts: ArraySpec; occ_word: ArraySpec;
           occ_ticks: ArraySpec; occ_words: string; hours: number; hop_samples: number };
  verify: { embeddings: ArraySpec; scales: ArraySpec; stream_starts: ArraySpec; vad: ArraySpec };
  augment: { noise: ArraySpec; babble: ArraySpec; rirs: ArraySpec };
  base: { version: string; created_at: string; teacher: { model: string } | null;
          evaluation?: Record<string, unknown> };
  languages_detail: Record<string, LanguageInfo>;
  files: Record<string, { bytes: number; sha256: string }>;
}

/** The fine-tuning settings, with the manifest's overrides. */
export function finetuneConfig(m: Manifest): FinetuneConfig {
  const camel = (k: string) => k.replace(/_(\w)/g, (_, c: string) => c.toUpperCase());
  const overrides = Object.fromEntries(Object.entries(m.finetune ?? {}).map(([k, v]) => [camel(k), v]));
  for (const k of Object.keys(overrides)) if (!(k in FINETUNE)) throw new Error(`unknown fine-tuning setting ${k}`);
  return { ...FINETUNE, ...overrides };
}

export interface Verify {
  rows: QuantizedRows;
  streamStarts: Uint32Array;
  vad: Uint8Array; // per VAD frame, probability * 255
  hours: number;
}

/** Bytes of one file of the version directory; `onBytes` reports download progress. */
export type Fetcher = (name: string, onBytes?: (received: number) => void) => Promise<Uint8Array>;

function typed(spec: ArraySpec, bytes: Uint8Array) {
  const n = spec.shape.reduce((a, b) => a * b, 1);
  const { buffer, byteOffset } = bytes.byteOffset % 8 === 0 ? bytes : bytes.slice();
  switch (spec.dtype) {
    case "int8": return new Int8Array(buffer, byteOffset, n);
    case "uint8": return new Uint8Array(buffer, byteOffset, n);
    case "int16": return new Int16Array(buffer, byteOffset, n);
    case "uint32": return new Uint32Array(buffer, byteOffset, n);
    case "float32": return new Float32Array(buffer, byteOffset, n);
    default: throw new Error(`unsupported dtype ${spec.dtype} in ${spec.file}`);
  }
}

async function load<T>(fetcher: Fetcher, spec: ArraySpec, progress?: (name: string, n: number) => void): Promise<T> {
  return typed(spec, await fetcher(spec.file, (n) => progress?.(spec.file, n))) as T;
}

async function json<T>(fetcher: Fetcher, name: string): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await fetcher(name))) as T;
}

const audio = (pcm: Int16Array) => Float32Array.from(pcm, (v) => v / 32767);

export async function loadAugment(fetcher: Fetcher, m: Manifest): Promise<AugmentBanks> {
  const [noise, babble, rirs] = await Promise.all([
    load<Int16Array>(fetcher, m.augment.noise), load<Int16Array>(fetcher, m.augment.babble),
    load<Int16Array>(fetcher, m.augment.rirs),
  ]);
  const [count, length] = m.augment.rirs.shape;
  const all = audio(rirs);
  return { noise, babble, // int16: noise is scaled to its SNR, so its unit doesn't matter
           rirs: Array.from({ length: count }, (_, i) => all.subarray(i * length, (i + 1) * length)) };
}

export interface TrainAssets {
  manifest: Manifest;
  bank: Bank;
  cache: Cache;
  verify: Verify;
  template: Uint8Array;
  vad: Uint8Array;
  vadLicense: Uint8Array;
}

/** What each embedding worker loads (fetched once by the compute worker first, so the
 * workers find it cached instead of racing to download it). */
export const EMBED_FILES = (m: Manifest) =>
  ["encoder.onnx", m.augment.noise.file, m.augment.babble.file, m.augment.rirs.file];

/** Bytes of the large files, for a progress bar before the first training. */
export const TRAIN_FILES = (m: Manifest) => [
  m.bank.embeddings, m.bank.scales, m.bank.word, m.bank.lang, m.cache.embeddings, m.cache.scales,
  m.cache.stream_starts, m.cache.occ_word, m.cache.occ_ticks, m.verify.embeddings, m.verify.scales,
  m.verify.stream_starts, m.verify.vad,
].map((s) => s.file).concat([m.bank.words, m.cache.occ_words, "model_template.onnx", "vad.onnx", "LICENSE-silero-vad",
                             ...EMBED_FILES(m)]);

export async function loadTrainAssets(fetcher: Fetcher, m: Manifest,
                                      progress?: (name: string, n: number) => void): Promise<TrainAssets> {
  const dim = m.embedding_dim;
  const quantized = async (e: ArraySpec, s: ArraySpec): Promise<QuantizedRows> => ({
    q: await load<Int8Array>(fetcher, e, progress), scale: await load<Float32Array>(fetcher, s, progress),
    rows: e.shape[0], dim,
  });
  const [bankRows, cacheRows, verifyRows] = await Promise.all([
    quantized(m.bank.embeddings, m.bank.scales), quantized(m.cache.embeddings, m.cache.scales),
    quantized(m.verify.embeddings, m.verify.scales),
  ]);
  const bank: Bank = {
    rows: bankRows, word: await load(fetcher, m.bank.word, progress), lang: await load(fetcher, m.bank.lang, progress),
    words: await json<string[]>(fetcher, m.bank.words), languages: m.languages,
  };
  const cache: Cache = {
    rows: cacheRows, streamStarts: await load(fetcher, m.cache.stream_starts, progress),
    occWord: await load(fetcher, m.cache.occ_word, progress), occTicks: await load(fetcher, m.cache.occ_ticks, progress),
    occWords: await json<string[]>(fetcher, m.cache.occ_words), hours: m.cache.hours,
  };
  const verify: Verify = {
    rows: verifyRows, streamStarts: await load(fetcher, m.verify.stream_starts, progress),
    vad: await load(fetcher, m.verify.vad, progress),
    hours: (verifyRows.rows * m.cache.hop_samples) / 16000 / 3600,
  };
  const [template, vad, vadLicense] = await Promise.all(
    ["model_template.onnx", "vad.onnx", "LICENSE-silero-vad"].map((f) => fetcher(f, (n) => progress?.(f, n))));
  return { manifest: m, bank, cache, verify, template, vad, vadLicense };
}
