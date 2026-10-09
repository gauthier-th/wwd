// Several embedding workers, each with its own single-threaded onnxruntime session: parallel
// without SharedArrayBuffer, so the app needs no cross-origin-isolation headers and stays
// servable by any static host.

import type { Manifest } from "./assets";
import type { EmbedJob, Embedder } from "./embed";
import { HOP } from "./finetune";
import type { FromEmbedder, ToEmbedder } from "./protocol";

const CHUNK = 24; // signals per message: big enough to amortize, small enough to balance

// Each worker holds an onnxruntime instance (~50 MB): four is most of the speed-up on a desktop
// and keeps phones within memory.
export const poolSize = () => Math.max(1, Math.min(4, (navigator.hardwareConcurrency || 2) - 1));

class EmbedWorker {
  private readonly worker = new Worker(new URL("./embed.worker.ts", import.meta.url), { type: "module" });
  private pending = new Map<number, { resolve: (e: Float32Array) => void; reject: (e: Error) => void }>();
  private nextId = 0;
  readonly ready: Promise<void>;

  constructor(root: URL, manifest: Manifest) {
    this.ready = new Promise((resolve, reject) => {
      this.worker.onmessage = (e: MessageEvent<FromEmbedder>) => {
        const msg = e.data;
        if (msg.type === "ready") resolve();
        else if (msg.type === "result") this.settle(msg.id)?.resolve(msg.embeddings);
        else if (msg.id === null) reject(new Error(msg.message));
        else this.settle(msg.id)?.reject(new Error(msg.message));
      };
    });
    this.send({ type: "init", root: root.href, manifest });
  }

  private settle(id: number) {
    const p = this.pending.get(id);
    this.pending.delete(id);
    return p;
  }

  private send(msg: ToEmbedder, transfer: Transferable[] = []) {
    this.worker.postMessage(msg, transfer);
  }

  embed(signals: Float32Array[], first: number, job: EmbedJob): Promise<Float32Array> {
    const id = this.nextId++;
    // Own buffers only: a view into a larger buffer would transfer (and detach) all of it.
    const own = signals.map((s) => (s.byteOffset === 0 && s.byteLength === s.buffer.byteLength ? s : s.slice()));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.send({ type: "embed", id, signals: own, first, augment: job.augment, seed: job.seed, hop: job.hop ?? HOP },
                own.map((s) => s.buffer as ArrayBuffer));
    });
  }

  terminate(): void {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Error("embedding worker stopped"));
  }
}

export class PoolEmbedder implements Embedder {
  private readonly workers: EmbedWorker[];
  readonly ready: Promise<void>;

  constructor(root: URL, readonly manifest: Manifest, size = poolSize()) {
    this.workers = Array.from({ length: size }, () => new EmbedWorker(root, manifest));
    this.ready = Promise.all(this.workers.map((w) => w.ready)).then(() => undefined);
  }

  get dim(): number {
    return this.manifest.embedding_dim;
  }

  async embed(job: EmbedJob, onDone?: (signals: number) => void): Promise<Float32Array> {
    await this.ready;
    const chunks = Math.ceil(job.count / CHUNK);
    const results: Float32Array[] = new Array(chunks);
    let next = 0;
    let done = 0;
    await Promise.all(this.workers.map(async (w) => {
      while (next < chunks) {
        const c = next++;
        const first = c * CHUNK;
        const signals = Array.from({ length: Math.min(CHUNK, job.count - first) }, (_, j) => job.make(first + j));
        results[c] = await w.embed(signals, first, job);
        done += signals.length;
        onDone?.(done);
      }
    }));
    const out = new Float32Array(results.reduce((n, r) => n + r.length, 0));
    let off = 0;
    for (const r of results) {
      out.set(r, off);
      off += r.length;
    }
    return out;
  }

  close(): void {
    for (const w of this.workers) w.terminate();
  }
}
