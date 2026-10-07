// Benchmark (PLAN.md M4): stream audio through the detector in this worker, one tick (1536
// samples) per call as when listening, and time each call. The same kind of worker the live test
// runs in, so the numbers are what a field test will see.

import { Detector } from "@wwd/runtime";

export interface BenchRequest {
  model: ArrayBuffer;
  vad: ArrayBuffer;
  audio: Float32Array; // 16 kHz
  gate: boolean;
}

export interface BenchResult {
  gate: boolean;
  ticks: number;
  audioS: number;
  gateOpen: number; // share of ticks
  events: number;
  latencyMs: { p50: number; p95: number; max: number; mean: number };
  busyPct: number; // detector time over audio time: the share of one core it needs live
}

const quantile = (sorted: number[], q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))];

self.onmessage = async (e: MessageEvent<BenchRequest>) => {
  try {
    const { model, vad, audio, gate } = e.data;
    const det = await Detector.create(model, vad, { gate, batchSize: 1, recordTicks: true });
    const hop = det.config.hop_samples;
    await det.process(audio.subarray(0, 2 * 16000)); // warm up the sessions
    det.reset();
    const times: number[] = [];
    let open = 0;
    let events = 0;
    for (let i = 0; i + hop <= audio.length; i += hop) {
      const started = performance.now();
      events += (await det.process(audio.slice(i, i + hop))).length;
      times.push(performance.now() - started);
      open += det.ticks.filter((t) => t.gateOpen).length;
      det.ticks = [];
    }
    const sorted = [...times].sort((a, b) => a - b);
    const total = times.reduce((a, b) => a + b, 0);
    const audioS = (times.length * hop) / 16000;
    const r = (x: number) => Math.round(x * 1000) / 1000;
    const result: BenchResult = {
      gate,
      ticks: times.length,
      audioS,
      gateOpen: open / times.length,
      events,
      latencyMs: { p50: r(quantile(sorted, 0.5)), p95: r(quantile(sorted, 0.95)), max: r(sorted[sorted.length - 1]),
                   mean: r(total / times.length) },
      busyPct: Math.round((total / 1000 / audioS) * 10000) / 100,
    };
    self.postMessage({ ok: true, result });
  } catch (err) {
    self.postMessage({ ok: false, error: String(err) });
  }
};
