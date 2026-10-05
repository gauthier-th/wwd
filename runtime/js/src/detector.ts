// Streaming wake-word detector (runtime/SPEC.md §3) on onnxruntime-web. Mirrors runtime/detector.py.

import { readOnnxMetadata } from "./onnxMeta.js";
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

export interface DetectorOptions {
  gate?: boolean;
  config?: DetectorConfig;
  overrides?: Partial<DetectorConfig>;
  batchSize?: number;
  recordTicks?: boolean;
}

/**
 * Feed 16 kHz float32 audio of any chunk size to `process`; get trigger events back.
 * Long chunks run the wake model batched, with results identical to small-chunk streaming.
 */
export class Detector {
  readonly config: DetectorConfig;
  readonly gate: boolean;
  readonly batchSize: number;
  readonly recordTicks: boolean;
  ticks: Tick[] = [];

  private readonly gateState: Gate;
  private readonly trigger: Trigger;
  private readonly inputName: string;
  private readonly outputName: string;
  private history!: Float32Array;
  private pending!: Float32Array;
  private frames = 0;
  private tickCount = 0;

  private constructor(
    private readonly session: ort.InferenceSession,
    readonly metadata: Record<string, string>,
    private readonly vad: SileroVad | null,
    options: DetectorOptions,
  ) {
    const base = options.config ?? configFromMetadata(metadata);
    this.config = makeConfig(options.overrides ?? {}, base);
    this.gate = (options.gate ?? true) && vad !== null;
    this.batchSize = options.batchSize ?? 64;
    this.recordTicks = options.recordTicks ?? false;
    this.inputName = session.inputNames[0];
    this.outputName = session.outputNames[0];
    this.gateState = new Gate(this.config);
    this.trigger = new Trigger(this.config);
    this.reset();
  }

  static async create(model: ModelSource, vad: ModelSource | null = null, options: DetectorOptions = {}) {
    const bytes = await loadBytes(model);
    const [session, vadModel] = await Promise.all([
      createSession(bytes),
      vad === null ? Promise.resolve(null) : SileroVad.create(vad),
    ]);
    return new Detector(session, readOnnxMetadata(bytes), vadModel, options);
  }

  reset(): void {
    this.history = new Float32Array(this.config.window_samples);
    this.pending = new Float32Array(0);
    this.frames = 0;
    this.tickCount = 0;
    this.gateState.reset();
    this.trigger.reset();
    this.vad?.reset();
    this.ticks = [];
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
    const due: { tick: number; vadMax: number; gateOpen: boolean; end: number }[] = [];
    for (let i = 0; i < nFrames; i++) {
      const end = window + (i + 1) * VAD_FRAME;
      if (this.vad) this.gateState.push(await this.vad.process(buf.slice(end - VAD_FRAME, end)));
      this.frames += 1;
      if (this.frames % perTick) continue;
      const vadMax = this.vad ? this.gateState.vadMax : Number.NaN;
      const gateOpen = !this.gate || this.gateState.open;
      due.push({ tick: this.tickCount, vadMax, gateOpen, end });
      this.tickCount += 1;
    }

    const scores = await this.score(due.filter((d) => d.gateOpen).map((d) => d.end), buf);
    const events: WakeEvent[] = [];
    let next = 0;
    for (const { tick, vadMax, gateOpen } of due) {
      const score = gateOpen ? scores[next++] : 0;
      if (this.recordTicks) this.ticks.push({ index: tick, vadMax, gateOpen, score });
      if (this.trigger.update(tick, score)) {
        const sample = (tick + 1) * cfg.hop_samples;
        events.push({ tick, sample, timeS: sample / cfg.sample_rate, score });
      }
    }

    this.history = buf.slice(buf.length - window);
    return events;
  }

  private async score(ends: number[], buf: Float32Array): Promise<Float32Array> {
    const w = this.config.window_samples;
    const out = new Float32Array(ends.length);
    for (let i = 0; i < ends.length; i += this.batchSize) {
      const batch = ends.slice(i, i + this.batchSize);
      const input = new Float32Array(batch.length * w);
      batch.forEach((end, j) => input.set(buf.subarray(end - w, end), j * w));
      const res = await this.session.run({ [this.inputName]: new ort.Tensor("float32", input, [batch.length, w]) });
      out.set(res[this.outputName].data as Float32Array, i);
    }
    return out;
  }
}
