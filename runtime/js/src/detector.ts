// Streaming wake-word detector (runtime/SPEC.md §3, §5) on onnxruntime-web. Mirrors runtime/detector.py.

import { readOnnxMetadata } from "./onnxMeta.js";
import { sameBytes, splitModel } from "./onnxSplit.js";
import { createSession, loadBytes, ort, type ModelSource } from "./ort.js";
import {
  configFromMetadata,
  framesPerTick,
  gateFrames,
  makeConfig,
  refractoryTicks,
  VAD_FRAME,
  type DetectorConfig,
} from "./spec.js";
import { SileroVad } from "./vad.js";

export interface WakeEvent {
  tick: number;
  sample: number; // stream position (samples) at the end of the triggering window
  timeS: number;
  score: number;
  word: number; // which wake word fired: its index in the models given to MultiDetector.create
}

export interface Tick {
  index: number;
  vadMax: number; // max VAD probability over the gate span; NaN when no VAD is used
  gateOpen: boolean;
  score: number; // 0 when the gate was closed
}

/** Speech gate: open while any VAD frame in the last `gate_frames` reached the threshold. */
export class Gate {
  private recent: number[] = [];
  private readonly size: number;

  constructor(readonly config: DetectorConfig) {
    this.size = gateFrames(config);
  }

  reset(): void {
    this.recent = [];
  }

  push(vadProb: number): void {
    this.recent.push(vadProb);
    if (this.recent.length > this.size) this.recent.shift();
  }

  get vadMax(): number {
    return this.recent.length ? Math.max(...this.recent) : 0;
  }

  get open(): boolean {
    return this.vadMax >= this.config.vad_threshold;
  }
}

/** k-of-n smoothing + refractory period, fed one tick score at a time. */
export class Trigger {
  private recent: boolean[] = [];
  private nextAllowed = 0;

  constructor(readonly config: DetectorConfig) {}

  reset(): void {
    this.recent = [];
    this.nextAllowed = 0;
  }

  update(tick: number, score: number): boolean {
    this.recent.push(score >= this.config.threshold);
    if (this.recent.length > this.config.smoothing_n) this.recent.shift();
    const hits = this.recent.filter(Boolean).length;
    if (tick < this.nextAllowed || hits < this.config.smoothing_k) return false;
    this.nextAllowed = tick + refractoryTicks(this.config) + 1;
    return true;
  }
}

/** One wake word of a MultiDetector: its file's metadata and settings, and its ticks. */
export interface Word {
  readonly index: number;
  readonly name: string; // the file's `wake_word`, or "word <index + 1>"
  readonly metadata: Record<string, string>;
  readonly config: DetectorConfig;
  ticks: Tick[];
}

/** A session the audio windows go through: one whole file, or an encoder that several files
 * share, followed by each of their heads. */
class Pass {
  constructor(
    readonly words: number[],
    private readonly session: ort.InferenceSession,
    private readonly heads: ort.InferenceSession[] | null = null,
  ) {}

  /** [n, window] audio -> one column of n scores per word. */
  async run(windows: Float32Array, n: number, window: number): Promise<Float32Array[]> {
    const res = await this.session.run({ [this.session.inputNames[0]]: new ort.Tensor("float32", windows, [n, window]) });
    const out = res[this.session.outputNames[0]];
    if (!this.heads) return [out.data as Float32Array];
    const columns: Float32Array[] = [];
    for (const head of this.heads) {
      const scores = await head.run({ [head.inputNames[0]]: out });
      columns.push(scores[head.outputNames[0]].data as Float32Array);
    }
    return columns;
  }
}

/** Files whose encoders are byte-identical (made from the same base model and precision) share
 * one encoder pass; any other file runs whole, as it would alone. */
async function createPasses(models: Uint8Array[], shareEncoder: boolean): Promise<Pass[]> {
  const splits = models.map((m) => (shareEncoder && models.length > 1 ? splitModel(m) : null));
  const groups: number[][] = [];
  splits.forEach((s, i) => {
    const group = s && groups.find((g) => splits[g[0]] && sameBytes(splits[g[0]]!.encoder, s.encoder));
    if (group) group.push(i);
    else groups.push([i]);
  });
  return Promise.all(groups.map(async (words) =>
    words.length === 1
      ? new Pass(words, await createSession(models[words[0]]))
      : new Pass(words, await createSession(splits[words[0]]!.encoder),
                 await Promise.all(words.map((i) => createSession(splits[i]!.head))))));
}

export interface MultiDetectorOptions {
  gate?: boolean;
  /** One set for every word, or one per word (undefined keeps a file's settings). */
  overrides?: Partial<DetectorConfig> | (Partial<DetectorConfig> | undefined)[];
  batchSize?: number;
  recordTicks?: boolean;
  shareEncoder?: boolean;
}

/**
 * Listen for several wake words at once. Feed 16 kHz float32 audio of any chunk size to
 * `process`; get trigger events back, in tick order, then word order.
 *
 * Each word keeps its file's settings, its own gate and trigger, and gets exactly the ticks and
 * events a Detector on its file would. The words share the audio, the VAD and the ticks, and
 * files made from the same base model run their common encoder once per window: a second wake
 * word then costs a head of a few hundred multiply-adds. `groups` lists the words of each pass
 * (one group: everything shared).
 *
 * Long chunks run the models batched, with results identical to small-chunk streaming.
 */
export class MultiDetector {
  readonly gate: boolean;
  readonly batchSize: number;
  readonly recordTicks: boolean;
  readonly groups: number[][];

  private readonly config: DetectorConfig; // the stream settings, common to every word
  private readonly gates: Gate[];
  private readonly triggers: Trigger[];
  private history!: Float32Array;
  private pending!: Float32Array;
  private frames = 0;
  private tickCount = 0;

  private constructor(
    readonly words: Word[],
    private readonly passes: Pass[],
    private readonly vad: SileroVad | null,
    options: MultiDetectorOptions,
  ) {
    this.config = words[0].config;
    this.gate = (options.gate ?? true) && vad !== null;
    this.batchSize = options.batchSize ?? 64;
    this.recordTicks = options.recordTicks ?? false;
    this.groups = passes.map((p) => p.words);
    this.gates = words.map((w) => new Gate(w.config));
    this.triggers = words.map((w) => new Trigger(w.config));
    this.reset();
  }

  static async create(models: ModelSource[], vad: ModelSource | null = null,
                      options: MultiDetectorOptions = {}): Promise<MultiDetector> {
    if (!models.length) throw new Error("give at least one wake-word model");
    const data = await Promise.all(models.map(loadBytes));
    const given = options.overrides;
    const overrides = Array.isArray(given) ? given : data.map(() => given);
    if (overrides.length !== data.length) throw new Error(`${overrides.length} overrides for ${data.length} models`);
    const words: Word[] = data.map((bytes, index) => {
      const metadata = readOnnxMetadata(bytes);
      const config = makeConfig(overrides[index] ?? {}, configFromMetadata(metadata));
      return { index, name: metadata.wake_word || `word ${index + 1}`, metadata, config, ticks: [] };
    });
    const stream = new Set(words.map((w) => `${w.config.window_samples}/${w.config.hop_samples}`));
    if (stream.size > 1) {
      throw new Error(`the wake words need the same window_samples and hop_samples, got ${[...stream].join(", ")}`);
    }
    const [passes, vadModel] = await Promise.all([
      createPasses(data, options.shareEncoder ?? true),
      vad === null ? Promise.resolve(null) : SileroVad.create(vad),
    ]);
    return new MultiDetector(words, passes, vadModel, options);
  }

  reset(): void {
    this.history = new Float32Array(this.config.window_samples);
    this.pending = new Float32Array(0);
    this.frames = 0;
    this.tickCount = 0;
    this.words.forEach((w, i) => {
      w.ticks = [];
      this.gates[i].reset();
      this.triggers[i].reset();
    });
    this.vad?.reset();
  }

  async process(samples: Float32Array): Promise<WakeEvent[]> {
    const cfg = this.config;
    const audio = new Float32Array(this.pending.length + samples.length);
    audio.set(this.pending, 0);
    audio.set(samples, this.pending.length);
    const nFrames = Math.floor(audio.length / VAD_FRAME);
    this.pending = audio.slice(nFrames * VAD_FRAME);
    if (nFrames === 0) return [];

    const window = cfg.window_samples;
    const buf = new Float32Array(window + nFrames * VAD_FRAME);
    buf.set(this.history, 0);
    buf.set(audio.subarray(0, nFrames * VAD_FRAME), window);

    const perTick = framesPerTick(cfg);
    const ends: number[] = []; // per due tick; per word for the next two
    const vadMax: number[][] = [];
    const gateOpen: boolean[][] = [];
    for (let i = 0; i < nFrames; i++) {
      const end = window + (i + 1) * VAD_FRAME;
      if (this.vad) {
        const prob = await this.vad.process(buf.slice(end - VAD_FRAME, end));
        for (const gate of this.gates) gate.push(prob);
      }
      this.frames += 1;
      if (this.frames % perTick) continue;
      ends.push(end);
      vadMax.push(this.gates.map((g) => (this.vad ? g.vadMax : Number.NaN)));
      gateOpen.push(this.gates.map((g) => !this.gate || g.open));
    }

    const scores = await this.score(ends, gateOpen, buf);
    const n = this.words.length;
    const events: WakeEvent[] = [];
    for (let j = 0; j < ends.length; j++) {
      const tick = this.tickCount + j;
      for (const word of this.words) {
        const k = word.index;
        const score = scores[j * n + k];
        if (this.recordTicks) word.ticks.push({ index: tick, vadMax: vadMax[j][k], gateOpen: gateOpen[j][k], score });
        if (this.triggers[k].update(tick, score)) {
          const sample = (tick + 1) * cfg.hop_samples;
          events.push({ tick, sample, timeS: sample / cfg.sample_rate, score, word: k });
        }
      }
    }
    this.tickCount += ends.length;
    this.history = buf.slice(buf.length - window);
    return events;
  }

  /** [ticks x words] scores, row-major; 0 where a word's gate is closed. A pass runs on the
   * ticks where any of its words' gates is open. */
  private async score(ends: number[], gateOpen: boolean[][], buf: Float32Array): Promise<Float32Array> {
    const w = this.config.window_samples;
    const n = this.words.length;
    const out = new Float32Array(ends.length * n);
    for (const pass of this.passes) {
      const rows = ends.flatMap((_, j) => (pass.words.some((k) => gateOpen[j][k]) ? [j] : []));
      for (let i = 0; i < rows.length; i += this.batchSize) {
        const batch = rows.slice(i, i + this.batchSize);
        const input = new Float32Array(batch.length * w);
        batch.forEach((j, b) => input.set(buf.subarray(ends[j] - w, ends[j]), b * w));
        const columns = await pass.run(input, batch.length, w);
        batch.forEach((j, b) => pass.words.forEach((k, c) => {
          if (gateOpen[j][k]) out[j * n + k] = columns[c][b];
        }));
      }
    }
    return out;
  }
}

export interface DetectorOptions {
  gate?: boolean;
  config?: DetectorConfig; // replaces the settings in the file's metadata; `overrides` then apply
  overrides?: Partial<DetectorConfig>;
  batchSize?: number;
  recordTicks?: boolean;
}

/**
 * One wake word. Feed 16 kHz float32 audio of any chunk size to `process`; get trigger events
 * back. Long chunks run the wake model batched, with results identical to small-chunk streaming.
 */
export class Detector {
  private constructor(private readonly multi: MultiDetector) {}

  static async create(model: ModelSource, vad: ModelSource | null = null, options: DetectorOptions = {}) {
    const { config, overrides, ...rest } = options;
    const defined = Object.fromEntries(Object.entries(overrides ?? {}).filter(([, v]) => v !== undefined));
    return new Detector(await MultiDetector.create([model], vad, { ...rest, overrides: { ...config, ...defined } }));
  }

  get config(): DetectorConfig {
    return this.multi.words[0].config;
  }

  get metadata(): Record<string, string> {
    return this.multi.words[0].metadata;
  }

  get gate(): boolean {
    return this.multi.gate;
  }

  get batchSize(): number {
    return this.multi.batchSize;
  }

  get recordTicks(): boolean {
    return this.multi.recordTicks;
  }

  get ticks(): Tick[] {
    return this.multi.words[0].ticks;
  }

  set ticks(ticks: Tick[]) {
    this.multi.words[0].ticks = ticks;
  }

  reset(): void {
    this.multi.reset();
  }

  process(samples: Float32Array): Promise<WakeEvent[]> {
    return this.multi.process(samples);
  }
}
