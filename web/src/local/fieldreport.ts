// runtime/fieldreport.py `summarize`, for field tests logged in the browser: hours, triggers,
// false accepts per hour, busy share, latency, battery, and a threshold sweep that replays the
// runtime's trigger over the logged tick scores.

import { configFromMetadata, refractoryTicks } from "@wwd/runtime";

import type { FieldReport } from "../api";

export interface FieldEvent {
  time: number;
  tick: number;
  sample?: number;
  score: number;
  clip?: string;
}

export interface FieldStats {
  time: number;
  busy_pct: number;
  cpu_pct?: number | null;
  latency_ms: { p50: number | null; p95: number | null; max: number | null };
  overruns?: number;
  battery?: number | null;
}

export interface FieldRun {
  meta: { model: string; metadata: Record<string, string>; config: Record<string, string>; gate: boolean;
          host: Record<string, unknown>; input: string };
  events: FieldEvent[];
  stats: FieldStats[];
  scores: Float32Array;
  vadMax: Float32Array;
}

/** `trigger_ticks`: ticks where the runtime's Trigger fires over one continuous stream. */
export function triggerTicks(scores: Float32Array, threshold: number, k: number, n: number, refractory: number): number[] {
  const fired: number[] = [];
  let next = 0;
  for (let t = 0; t < scores.length; t++) {
    let hits = 0;
    for (let j = Math.max(0, t - n + 1); j <= t; j++) if (scores[j] >= threshold) hits++;
    if (hits >= k && t >= next) {
      fired.push(t);
      next = t + refractory + 1;
    }
  }
  return fired;
}

/** numpy.percentile (linear), rounded to 2 decimals; null without values. */
function pct(values: (number | null | undefined)[], q: number): number | null {
  const v = values.filter((x): x is number => x !== null && x !== undefined).sort((a, b) => a - b);
  if (!v.length) return null;
  const pos = (q / 100) * (v.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(v.length - 1, lo + 1);
  return Math.round((v[lo] + (v[hi] - v[lo]) * (pos - lo)) * 100) / 100;
}

const round = (x: number, d: number) => Math.round(x * 10 ** d) / 10 ** d;

export function summarize(run: FieldRun, trueEvents: Set<number> = new Set()): FieldReport {
  const cfg = configFromMetadata(run.meta.config);
  const tickS = cfg.hop_samples / cfg.sample_rate;
  const hours = (run.scores.length * tickS) / 3600;
  const gate = run.meta.gate && Array.from(run.vadMax).some((v) => !Number.isNaN(v));
  const speechHours = gate ? (Array.from(run.vadMax).filter((v) => v >= cfg.vad_threshold).length * tickS) / 3600 : hours;
  const falseAccepts = run.events.filter((_, i) => !trueEvents.has(i + 1));
  const stats = run.stats;
  const sweep = [0.5, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0].flatMap((factor) => {
    const threshold = cfg.threshold * factor;
    if (!(threshold > 0 && threshold < 1)) return [];
    const triggers = triggerTicks(run.scores, threshold, cfg.smoothing_k, cfg.smoothing_n, refractoryTicks(cfg)).length;
    return [{ factor, threshold: round(threshold, 4), triggers, per_hour: hours ? round(triggers / hours, 3) : null }];
  });
  const levels = stats.filter((s) => s.battery !== null && s.battery !== undefined);
  const battery = levels.length < 2 ? null : {
    start: levels[0].battery!, end: levels.at(-1)!.battery!,
    per_hour: round((levels.at(-1)!.battery! - levels[0].battery!) /
                    Math.max((levels.at(-1)!.time - levels[0].time) / 3600, 1e-9), 3),
  };
  return {
    wake_word: run.meta.metadata.wake_word ?? null,
    model: run.meta.model,
    host: run.meta.host,
    gate,
    threshold: cfg.threshold,
    hours: round(hours, 3),
    speech_hours: round(speechHours, 3),
    triggers: run.events.length,
    false_accepts: falseAccepts.length,
    false_accepts_per_hour: hours ? round(falseAccepts.length / hours, 3) : null,
    busy_pct: { mean: stats.length ? round(stats.reduce((a, s) => a + s.busy_pct, 0) / stats.length, 2) : null,
                max: stats.length ? Math.max(...stats.map((s) => s.busy_pct)) : null },
    latency_ms: { p50: pct(stats.map((s) => s.latency_ms.p50), 50), p95: pct(stats.map((s) => s.latency_ms.p95), 50),
                  max: stats.length ? Math.max(...stats.map((s) => s.latency_ms.max ?? 0)) : null },
    battery,
    sweep,
    events: run.events.map((e, i) => ({ n: i + 1, time: e.time, score: e.score, clip: e.clip,
                                        false_accept: !trueEvents.has(i + 1) })),
  };
}
