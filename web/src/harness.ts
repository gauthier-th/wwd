// M0 test harness (driven by e2e/runtime.spec.ts, also usable by hand):
//   /harness.html?mode=golden          golden parity through the real Worker + browser wasm engine
//   /harness.html?mode=mic&seconds=6   microphone -> AudioWorklet -> Worker pipeline smoke test

import { decodeWav, type Tick, type WakeEvent } from "@wwd/runtime";

import vadUrl from "../../runtime/assets/silero_vad_16k_op15.onnx?url";
import goldenUrl from "../../runtime/testdata/golden.json?url";
import speechUrl from "../../runtime/testdata/audio/speech.wav?url";
import int8Url from "../../runtime/testdata/models/model.int8.onnx?url";
import fp32Url from "../../runtime/testdata/models/model.onnx?url";
import { startDetector, startMic, type DetectorHandle } from "./audio/pipeline";

type HarnessState = { status: "running" | "done" | "error"; [key: string]: unknown };
declare global {
  interface Window {
    harness: HarnessState;
  }
}

const out = document.getElementById("out")!;
const show = (state: HarnessState) => {
  window.harness = state;
  out.textContent = JSON.stringify(state, null, 2);
};
const fetchBytes = async (url: string) => (await fetch(url)).arrayBuffer();

function collect(det: DetectorHandle) {
  const ticks: Tick[] = [];
  const events: WakeEvent[] = [];
  const errors: string[] = [];
  det.onMessage((msg) => {
    if (msg.type === "ticks") {
      ticks.push(...msg.ticks);
      events.push(...msg.events);
    } else if (msg.type === "error") errors.push(msg.message);
  });
  return { ticks, events, errors };
}

async function golden(): Promise<HarnessState> {
  const [golden, wav, vad, fp32, int8] = await Promise.all([
    fetch(goldenUrl).then((r) => r.json()),
    fetchBytes(speechUrl),
    fetchBytes(vadUrl),
    fetchBytes(fp32Url),
    fetchBytes(int8Url),
  ]);
  const audio = decodeWav(new Uint8Array(wav)).samples;
  const expected = golden.files["speech.wav"];
  const models = { fp32, int8 } as const;
  const runs = [];
  for (const kind of ["fp32", "int8"] as const) {
    for (const gate of [true, false]) {
      const started = performance.now();
      const det = await startDetector(models[kind].slice(0), vad.slice(0), { gate });
      const got = collect(det);
      for (let i = 0; i < audio.length; i += 1536) {
        det.send({ type: "audio", samples: audio.slice(i, i + 1536) }); // live-sized chunks
      }
      const n = expected.scores[kind].length;
      while (got.ticks.length < n && !got.errors.length) await new Promise((r) => setTimeout(r, 20));
      det.terminate();

      const gateOpen: boolean[] = gate ? expected.gate_open : expected.gate_open.map(() => true);
      const want = expected.scores[kind].map((s: number, i: number) => (gateOpen[i] ? s : 0));
      const diffs = got.ticks.map((t, i) => Math.abs(t.score - want[i]));
      const maxDiff = Math.max(...diffs);
      const meanDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
      const vadDiff = Math.max(...got.ticks.map((t, i) => Math.abs(t.vadMax - expected.vad_max[i])));
      const eventsGot = got.events.map((e) => e.tick);
      const eventsWant: number[] = expected.events[kind][gate ? "gate" : "no_gate"];
      const checks = {
        ticks: got.ticks.length === n,
        gate: got.ticks.every((t, i) => t.gateOpen === gateOpen[i]),
        scores: maxDiff <= golden.tolerance[kind] && (kind !== "int8" || meanDiff <= golden.tolerance.int8_mean),
        vad: vadDiff <= golden.tolerance.vad,
        events: JSON.stringify(eventsGot) === JSON.stringify(eventsWant),
        noErrors: got.errors.length === 0,
      };
      runs.push({
        name: `${kind} ${gate ? "gate" : "no_gate"}`,
        ok: Object.values(checks).every(Boolean),
        checks,
        maxDiff,
        meanDiff,
        vadDiff,
        events: eventsGot,
        errors: got.errors,
        ms: Math.round(performance.now() - started),
      });
      show({ status: "running", runs });
    }
  }
  return { status: "done", runs, ok: runs.every((r) => r.ok) };
}

async function mic(seconds: number): Promise<HarnessState> {
  const [vad, fp32] = await Promise.all([fetchBytes(vadUrl), fetchBytes(fp32Url)]);
  const det = await startDetector(fp32, vad, { gate: true });
  const got = collect(det);
  const capture = await startMic(det);
  const inputRate = capture.context.sampleRate;
  await new Promise((r) => setTimeout(r, seconds * 1000));
  await capture.stop();
  det.terminate();
  return {
    status: "done",
    inputRate,
    ticks: got.ticks.length,
    vadMax: Math.max(0, ...got.ticks.map((t) => t.vadMax)),
    gateOpenTicks: got.ticks.filter((t) => t.gateOpen).length,
    events: got.events.length,
    errors: got.errors,
  };
}

const params = new URLSearchParams(location.search);
show({ status: "running" });
(params.get("mode") === "mic" ? mic(Number(params.get("seconds") ?? 6)) : golden())
  .then(show)
  .catch((err) => show({ status: "error", message: String(err), stack: (err as Error).stack }));
