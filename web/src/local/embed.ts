// Embedding windows with the base encoder, augmented or not. Signals are made lazily, one per
// index, each with its own seed: memory stays bounded (3000 augmented windows would be
// 288 MB at once) and the result doesn't depend on which worker embeds which chunk.

import { configureOrt, ort } from "@wwd/runtime";

import { Augmenter, type AugmentBanks } from "./augment";
import { HOP, streamWindows } from "./finetune";
import { Rng } from "./rng";

export interface EmbedJob {
  count: number;
  make: (i: number) => Float32Array; // a window, or a stream cut every `hop` samples
  augment: boolean;
  seed: number;
  hop?: number;
}

export interface Embedder {
  readonly dim: number;
  /** [windows, dim] in signal order; `onDone` gets the number of signals embedded so far. */
  embed(job: EmbedJob, onDone?: (signals: number) => void): Promise<Float32Array>;
  close(): void;
}

/** Augment (optionally) and cut signals into windows: what an embedding worker does. */
export function prepare(signals: Float32Array[], first: number, augmenter: Augmenter | null, augment: boolean,
                        seed: number, hop: number): Float32Array[] {
  return signals.flatMap((s, j) => {
    const x = augment && augmenter ? augmenter.apply(s, Rng.derive(seed, first + j)) : s;
    return streamWindows(x, hop);
  });
}

export class EncoderSession {
  private constructor(private readonly session: ort.InferenceSession, readonly dim: number) {}

  static async create(encoder: Uint8Array, dim: number): Promise<EncoderSession> {
    configureOrt();
    return new EncoderSession(await ort.InferenceSession.create(encoder, { executionProviders: ["wasm"] }), dim);
  }

  /** Windows -> [n, dim], run in batches. */
  async run(windows: Float32Array[], batch = 8): Promise<Float32Array> {
    const out = new Float32Array(windows.length * this.dim);
    const input = this.session.inputNames[0];
    for (let i = 0; i < windows.length; i += batch) {
      const n = Math.min(batch, windows.length - i);
      const data = new Float32Array(n * windows[i].length);
      for (let j = 0; j < n; j++) data.set(windows[i + j], j * windows[i].length);
      const res = await this.session.run({ [input]: new ort.Tensor("float32", data, [n, windows[i].length]) });
      out.set(res[this.session.outputNames[0]].data as Float32Array, i * this.dim);
    }
    return out;
  }

  release(): void {
    void this.session.release();
  }
}

/** In-process embedder: one session, one thread (Node tests, small jobs). */
export class SessionEmbedder implements Embedder {
  private readonly augmenter: Augmenter | null;

  constructor(private readonly session: EncoderSession, banks: AugmentBanks | null, readonly chunk = 32) {
    this.augmenter = banks ? new Augmenter(banks) : null;
  }

  get dim(): number {
    return this.session.dim;
  }

  async embed(job: EmbedJob, onDone?: (signals: number) => void): Promise<Float32Array> {
    const parts: Float32Array[] = [];
    for (let i = 0; i < job.count; i += this.chunk) {
      const signals = Array.from({ length: Math.min(this.chunk, job.count - i) }, (_, j) => job.make(i + j));
      parts.push(await this.session.run(prepare(signals, i, this.augmenter, job.augment, job.seed, job.hop ?? HOP)));
      onDone?.(Math.min(job.count, i + this.chunk));
    }
    const out = new Float32Array(parts.reduce((n, p) => n + p.length, 0));
    let off = 0;
    for (const p of parts) {
      out.set(p, off);
      off += p.length;
    }
    return out;
  }

  close(): void {
    this.session.release();
  }
}
