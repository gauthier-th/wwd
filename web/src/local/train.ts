// Quick mode in the browser: server/quick.py `run`, stage for stage. It embeds the
// recordings, fits the pushed-prototype head, calibrates it on the long-form ticks, composes
// model.onnx, and verifies it: false accepts on held-out ticks (through the runtime's own Gate
// and Trigger) and every recording through the runtime Detector running the composed file.

import { DEFAULT_CONFIG, Detector, FORMAT_VERSION, Gate, makeConfig, ort, refractoryTicks, Trigger,
  type DetectorConfig } from "@wwd/runtime";

import type { Report } from "../api";
import type { TrainAssets } from "./assets";
import { composeModel } from "./compose";
import type { Embedder } from "./embed";
import {
  calibrate, crossValidate, FINETUNE, fitHead, headScores, HOP, maskWord, partialWindow, place, positiveWindow,
  SHIFTS, slidingWindows, tickStreams, WINDOW, type EnrollmentEmbeddings, type FinetuneConfig, type Head,
  type Rows, type TriggerConfig,
} from "./finetune";
import { Rng } from "./rng";

export const OPERATING_POINTS = [0.1, 0.25, 0.5, 1, 2, 5];
const RECALL_COPIES = 10;

export interface TrainJob {
  wakeWord: string;
  language: string;
  targetFaPerHour: number;
  positives: Float32Array[]; // processed recordings, 16 kHz
  positiveIds: string[];
  negatives: Float32Array[]; // the user's other speech and ambience
  negativeSeconds: Record<string, number>;
}

export type Progress = (stage: string, progress: number, message: string) => void;

export class Cancelled extends Error {
  constructor() {
    super("cancelled");
  }
}

export interface TrainResult {
  report: Report;
  head: Head;
  model: Uint8Array; // model.onnx at the target operating point
}

export const triggerConfig = (c: DetectorConfig): TriggerConfig =>
  ({ k: c.smoothing_k, n: c.smoothing_n, refractoryTicks: refractoryTicks(c) });

/** model.onnx metadata at one threshold (as server/quick.py writes it). */
export function modelMetadata(job: { wakeWord: string; language: string }, version: string, threshold: number,
                              createdAt: string): Record<string, string> {
  const cfg = makeConfig({ threshold });
  return {
    format_version: FORMAT_VERSION,
    ...Object.fromEntries(Object.entries(cfg).map(([k, v]) => [k, String(v)])),
    wake_word: job.wakeWord,
    language: job.language,
    base_model_version: version,
    created_at: createdAt,
    vad_file: "vad.onnx",
  };
}

const rows = (data: Float32Array, dim: number): Rows => ({ data, rows: data.length / dim, dim });

export async function train(job: TrainJob, assets: TrainAssets, embedder: Embedder, progress: Progress,
                            cancelled: () => boolean = () => false,
                            cfg: FinetuneConfig = FINETUNE): Promise<TrainResult> {
  const det = DEFAULT_CONFIG;
  const trigger = triggerConfig(det);
  const { bank, cache, verify, manifest } = assets;
  const n = job.positives.length;
  const timings: Record<string, number> = {};
  const started = performance.now();
  let last = started;
  let current = "loading";
  const step = (stage: string, frac: number, message: string) => {
    if (cancelled()) throw new Cancelled();
    timings[current] = Math.round(performance.now() - last) / 1000;
    last = performance.now();
    current = stage;
    progress(stage, frac, message);
  };

  // ---------------------------------------------------------------- embedding
  step("embedding", 0.05, `Embedding ${n} recordings x ${cfg.augmentPerClip} augmentations`);
  const user = slidingWindows(job.negatives, cfg.userHopS);
  const quiet = new Rng(cfg.seed + 7); // eval windows never hold digital silence (-70 dBFS floor)
  const cleanStreams = tickStreams(job.positives, 1).map((s) => s.map((v) => v + quiet.normal() * 10 ** (-70 / 20)));
  const noisyStreams = tickStreams(job.positives, RECALL_COPIES);
  const jobs = [
    { count: n, make: (i: number) => place(job.positives[i], null), augment: false, seed: 1 },
    { count: n * cfg.augmentPerClip, make: positiveWindow(job.positives, cfg.augmentPerClip, cfg, 2), augment: true, seed: 3 },
    { count: n * cfg.partialPerClip, make: partialWindow(job.positives, cfg.partialPerClip, cfg, 4), augment: true, seed: 5 },
    { count: user.length, make: (i: number) => user[i], augment: true, seed: 6 },
    { count: noisyStreams.length, make: (i: number) => noisyStreams[i], augment: true, seed: 8, hop: HOP },
    { count: cleanStreams.length, make: (i: number) => cleanStreams[i], augment: false, seed: 9, hop: HOP },
  ];
  const total = jobs.reduce((a, j) => a + j.count, 0);
  let done = 0;
  const out: Float32Array[] = [];
  for (const j of jobs) {
    const before = done;
    out.push(j.count ? await embedder.embed(j, (k) => {
      if (cancelled()) throw new Cancelled();
      done = before + k;
      progress("embedding", 0.05 + (0.3 * done) / total, `Embedded ${done} of ${total} windows`);
    }) : new Float32Array(0));
  }
  const dim = embedder.dim;
  const emb: EnrollmentEmbeddings = {
    clean: rows(out[0], dim), positives: rows(out[1], dim), partial: rows(out[2], dim), user: rows(out[3], dim),
  };
  const noisy = rows(out[4], dim);
  const clean = rows(out[5], dim);
  if (noisy.rows !== n * RECALL_COPIES * SHIFTS || clean.rows !== n * SHIFTS) throw new Error("tick windows miscounted");

  // ---------------------------------------------------------------- cross-validation
  step("cross_validating", 0.35, `Leave-2-out folds over ${n} recordings`);
  const ignore = maskWord(cache, job.wakeWord);
  const folds = n >= 3
    ? crossValidate(emb, noisy, clean, job.positiveIds, bank, cache, job.wakeWord, job.language,
                    job.targetFaPerHour, ignore, cfg, trigger)
    : null;

  // ---------------------------------------------------------------- calibration
  step("calibrating", 0.5, `Calibrating to ${job.targetFaPerHour} false accepts/h`);
  const fit = fitHead(emb, bank, job.wakeWord, job.language, cfg);
  const scores = headScores(fit.head, cache.rows);
  const targets = [...new Set([...OPERATING_POINTS, job.targetFaPerHour])].sort((a, b) => a - b);
  const points = targets.map((fa) => {
    const [threshold, measured] = calibrate(scores, cache.streamStarts, cache.hours, fa, trigger, ignore);
    return { fa_per_hour: fa, fp32: { threshold, calibration_fa_per_hour: measured } };
  });
  const main = points.find((p) => p.fa_per_hour === job.targetFaPerHour)!.fp32;
  const calibration = {
    hours: cache.hours, target_fa_per_hour: job.targetFaPerHour,
    fp32: { threshold: main.threshold, fa_per_hour: main.calibration_fa_per_hour,
            target_met: main.calibration_fa_per_hour <= job.targetFaPerHour, own_cache: true },
  };

  // ---------------------------------------------------------------- compose
  step("composing", 0.55, "Composing model.onnx");
  const created = new Date().toISOString().replace(/\.\d+Z$/, "Z");
  const metadata = modelMetadata(job, manifest.version, main.threshold, created);
  const model = composeModel(assets.template, manifest.template, fit.head, metadata);

  // ---------------------------------------------------------------- verify
  step("verifying", 0.6, "Running model.onnx through the runtime");
  const runCfg = makeConfig({ threshold: main.threshold });
  const vScores = headScores(fit.head, verify.rows);
  const counts = { gate: 0, no_gate: 0 };
  for (let s = 0; s + 1 < verify.streamStarts.length; s++) {
    for (const gateOn of [false, true]) {
      const g = new Gate(runCfg);
      const t = new Trigger(runCfg);
      const a = verify.streamStarts[s];
      for (let tick = 0; tick < verify.streamStarts[s + 1] - a; tick++) {
        if (gateOn) for (let f = 0; f < 3; f++) g.push(verify.vad[3 * (a + tick) + f] / 255);
        if (t.update(tick, !gateOn || g.open ? vScores[a + tick] : 0)) counts[gateOn ? "gate" : "no_gate"]++;
      }
    }
  }
  const detector = await Detector.create(model, assets.vad, { gate: true });
  let detected = 0;
  const noise = new Rng(cfg.seed + 11);
  for (const clip of job.positives) {
    detector.reset();
    const stream = new Float32Array(16000 + clip.length + 24000);
    stream.set(clip, 16000);
    for (let i = 0; i < stream.length; i++) stream[i] += noise.normal() * 10 ** (-60 / 20);
    if ((await detector.process(stream)).length) detected++;
  }
  // The composed file against the head's own maths on the clean windows.
  const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  const windows = new Float32Array(n * WINDOW);
  job.positives.forEach((c, i) => windows.set(place(c, null), i * WINDOW));
  const res = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", windows, [n, WINDOW]) });
  void session.release();
  const fileScores = res[session.outputNames[0]].data as Float32Array;
  const mathScores = headScores(fit.head, emb.clean);
  const parity = Math.max(...Array.from(fileScores, (v, i) => Math.abs(v - mathScores[i])));
  const verification = {
    fp32: {
      hours: verify.hours,
      no_gate: { false_accepts: counts.no_gate, per_hour: counts.no_gate / verify.hours },
      gate: { false_accepts: counts.gate, per_hour: counts.gate / verify.hours },
      enrollment_detected: detected, enrollment_total: n,
    },
    parity: { ticks: n, file_vs_embedding_max: parity },
  };

  step("bundling", 0.97, "Saving the model");
  timings[current] = Math.round(performance.now() - last) / 1000;
  timings.total = Math.round(performance.now() - started) / 1000;
  const report = {
    wake_word: job.wakeWord, language: job.language, base_model_version: manifest.version,
    base_teacher: manifest.base.teacher, mode: "quick", trained_in: "browser", recordings: n,
    negative_seconds: job.negativeSeconds, threshold: main.threshold, target_fa_per_hour: job.targetFaPerHour,
    calibration, operating_points: points, cross_validation: folds, verification,
    finetune: fit.info, timings_s: timings,
  } as unknown as Report;
  return { report, head: fit.head, model };
}
