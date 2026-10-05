// JS side of the dual-runtime contract; mirrors runtime/tests/test_runtime.py and runs the
// real onnxruntime-web wasm build under Node.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import {
  configFromMetadata,
  decodeWav,
  Detector,
  encodeWav,
  framesPerTick,
  Gate,
  makeConfig,
  ort,
  readOnnxMetadata,
  SileroVad,
  Trigger,
  type WakeEvent,
} from "../src/index.js";

const RUNTIME = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DATA = join(RUNTIME, "testdata");
const VAD = new Uint8Array(readFileSync(join(RUNTIME, "assets", "silero_vad_16k_op15.onnx")));
const GOLDEN = JSON.parse(readFileSync(join(DATA, "golden.json"), "utf8"));
const CASES = JSON.parse(readFileSync(join(DATA, "logic_cases.json"), "utf8"));
const TOL = GOLDEN.tolerance;
const FILES = Object.keys(GOLDEN.files).sort();
const MODELS = {
  fp32: new Uint8Array(readFileSync(join(DATA, "models", "model.onnx"))),
  int8: new Uint8Array(readFileSync(join(DATA, "models", "model.int8.onnx"))),
};

const audio = (name: string) => {
  const { samples, sampleRate } = decodeWav(new Uint8Array(readFileSync(join(DATA, "audio", name))));
  expect(sampleRate).toBe(16000);
  return samples;
};

function expectClose(got: ArrayLike<number>, want: ArrayLike<number>, atol: number) {
  expect(got.length).toBe(want.length);
  let worst = 0;
  let at = -1;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    if (d > worst) [worst, at] = [d, i];
  }
  expect(worst, `max |diff| ${worst} at index ${at}`).toBeLessThanOrEqual(atol);
}

describe("logic cases", () => {
  for (const c of CASES.trigger) {
    it(`trigger: ${c.name}`, () => {
      const trigger = new Trigger(makeConfig(c.config));
      const fired = c.scores.flatMap((s: number, t: number) => (trigger.update(t, s) ? [t] : []));
      expect(fired).toEqual(c.expected);
    });
  }
  for (const c of CASES.gate) {
    it(`gate: ${c.name}`, () => {
      const cfg = makeConfig(c.config);
      const gate = new Gate(cfg);
      const opened: boolean[] = [];
      const maxes: number[] = [];
      c.vad.forEach((p: number, i: number) => {
        gate.push(p);
        if ((i + 1) % framesPerTick(cfg) === 0) {
          opened.push(gate.open);
          maxes.push(gate.vadMax);
        }
      });
      expect(opened).toEqual(c.expected_open);
      expectClose(maxes, c.expected_vad_max, 1e-12);
    });
  }
  it("config round-trips through metadata", () => {
    const meta = { ...GOLDEN.config, wake_word: "x" };
    expect(configFromMetadata(meta)).toEqual(makeConfig(Object.fromEntries(
      Object.entries(GOLDEN.config).map(([k, v]) => [k, Number(v)]))));
  });
});

describe("model files", () => {
  it("reads ONNX metadata_props without onnxruntime", () => {
    const meta = readOnnxMetadata(MODELS.fp32);
    expect(meta.wake_word).toBe("golden-stub");
    for (const [k, v] of Object.entries(GOLDEN.config)) expect(meta[k]).toBe(v);
    expect(readOnnxMetadata(MODELS.int8)).toEqual(meta);
  });

  it("WAV encode/decode round-trips 16-bit samples exactly", () => {
    const a = audio("speech.wav").subarray(0, 4000);
    expect(decodeWav(encodeWav(a)).samples).toEqual(a);
  });
});

describe("golden parity (onnxruntime-web wasm)", () => {
  for (const name of FILES) {
    it(`VAD matches official wrapper: ${name}`, async () => {
      const vad = await SileroVad.create(VAD);
      const a = audio(name);
      const got: number[] = [];
      for (let i = 0; i + 512 <= a.length; i += 512) got.push(await vad.process(a.slice(i, i + 512)));
      expectClose(got, GOLDEN.files[name].vad, TOL.vad);
    });
  }

  const runs: [keyof typeof MODELS, string, boolean, number | null][] = [];
  for (const kind of ["fp32", "int8"] as const)
    for (const name of FILES) for (const gate of [true, false]) runs.push([kind, name, gate, null]);
  runs.push(["fp32", "speech.wav", true, 1000]); // streaming in odd-sized chunks

  for (const [kind, name, gate, chunk] of runs) {
    it(`detector ${kind} ${name} ${gate ? "gate" : "no_gate"} ${chunk ? `chunk=${chunk}` : "whole"}`, async () => {
      const expected = GOLDEN.files[name];
      const det = await Detector.create(MODELS[kind], VAD, { gate, recordTicks: true });
      expect(det.config).toEqual(configFromMetadata(GOLDEN.config));
      const a = audio(name);
      const step = chunk ?? a.length;
      const events: WakeEvent[] = [];
      for (let i = 0; i < a.length; i += step) events.push(...(await det.process(a.slice(i, i + step))));

      const scores: number[] = expected.scores[kind];
      const gateOpen: boolean[] = gate ? expected.gate_open : scores.map(() => true);
      expect(det.ticks.map((t) => t.gateOpen)).toEqual(gateOpen);
      expectClose(det.ticks.map((t) => t.vadMax), expected.vad_max, TOL.vad);
      const got = det.ticks.map((t) => t.score);
      const want = scores.map((s, i) => (gateOpen[i] ? s : 0));
      expectClose(got, want, TOL[kind]);
      if (kind === "int8") {
        // rare one-step requantization flips are allowed, systematic drift is not
        const mean = got.reduce((acc, g, i) => acc + Math.abs(g - want[i]), 0) / got.length;
        expect(mean).toBeLessThanOrEqual(TOL.int8_mean);
      }
      expect(events.map((e) => e.tick)).toEqual(expected.events[kind][gate ? "gate" : "no_gate"]);
    });
  }

  it("encoder embeddings", async () => {
    const enc = GOLDEN.encoder;
    const a = audio(enc.file);
    const padded = new Float32Array(24000 + a.length);
    padded.set(a, 24000);
    const input = new Float32Array(enc.ticks.length * 24000);
    enc.ticks.forEach((t: number, i: number) =>
      input.set(padded.subarray((t + 1) * 1536, (t + 1) * 1536 + 24000), i * 24000));
    const session = await ort.InferenceSession.create(
      new Uint8Array(readFileSync(join(DATA, "models", "encoder.onnx"))), { executionProviders: ["wasm"] });
    const out = await session.run({ audio: new ort.Tensor("float32", input, [enc.ticks.length, 24000]) });
    expectClose(out.embedding.data as Float32Array, enc.embeddings.flat(), TOL.embedding);
  });
});
