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
  MultiDetector,
  ort,
  readOnnxMetadata,
  SileroVad,
  Trigger,
  type Tick,
  type WakeEvent,
} from "../src/index.js";
import { sameBytes, splitModel } from "../src/onnxSplit.js";

const RUNTIME = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const DATA = join(RUNTIME, "testdata");
const VAD = new Uint8Array(readFileSync(join(RUNTIME, "assets", "silero_vad_16k_op15.onnx")));
const GOLDEN = JSON.parse(readFileSync(join(DATA, "golden.json"), "utf8"));
const CASES = JSON.parse(readFileSync(join(DATA, "logic_cases.json"), "utf8"));
const TOL = GOLDEN.tolerance;
const FILES = Object.keys(GOLDEN.files).sort();
const model = (name: string) => new Uint8Array(readFileSync(join(DATA, "models", name)));
const MODELS = { fp32: model("model.onnx"), int8: model("model.int8.onnx") };
// Composed like shipped files, on the same encoder (runtime/SPEC.md §5).
const WORDS: Record<string, Uint8Array> = Object.fromEntries(Object.keys(GOLDEN.words).map((n) => [n, model(n)]));
const file = (m: string) => (m in MODELS ? MODELS[m as keyof typeof MODELS] : WORDS[m]);

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

/** Expected settings, ticks and events of one model file (a MODELS kind or a WORDS file). */
function golden(m: string, name: string) {
  if (m in WORDS) return { config: GOLDEN.words[m].config, ...GOLDEN.words[m].files[name] };
  const f = GOLDEN.files[name];
  return { config: GOLDEN.config, vad_max: f.vad_max, gate_open: f.gate_open, scores: f.scores[m], events: f.events[m] };
}

function checkGolden(m: string, name: string, gate: boolean, ticks: Tick[], events: WakeEvent[]) {
  const want = golden(m, name);
  const scores: number[] = want.scores;
  const gateOpen: boolean[] = gate ? want.gate_open : scores.map(() => true);
  expect(ticks.map((t) => t.gateOpen)).toEqual(gateOpen);
  expectClose(ticks.map((t) => t.vadMax), want.vad_max, TOL.vad);
  const got = ticks.map((t) => t.score);
  const expected = scores.map((s, i) => (gateOpen[i] ? s : 0));
  expectClose(got, expected, m === "int8" ? TOL.int8 : TOL.fp32);
  if (m === "int8") {
    // rare one-step requantization flips are allowed, systematic drift is not
    const mean = got.reduce((acc, g, i) => acc + Math.abs(g - expected[i]), 0) / got.length;
    expect(mean).toBeLessThanOrEqual(TOL.int8_mean);
  }
  expect(events.map((e) => e.tick)).toEqual(want.events[gate ? "gate" : "no_gate"]);
}

async function stream(det: { process(s: Float32Array): Promise<WakeEvent[]> }, a: Float32Array, chunk: number | null) {
  const step = chunk ?? a.length;
  const events: WakeEvent[] = [];
  for (let i = 0; i < a.length; i += step) events.push(...(await det.process(a.slice(i, i + step))));
  return events;
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

  const runs: [string, string, boolean, number | null][] = [];
  for (const m of ["fp32", "int8", ...Object.keys(WORDS)])
    for (const name of FILES) for (const gate of [true, false]) runs.push([m, name, gate, null]);
  runs.push(["fp32", "speech.wav", true, 1000]); // streaming in odd-sized chunks

  for (const [m, name, gate, chunk] of runs) {
    it(`detector ${m} ${name} ${gate ? "gate" : "no_gate"} ${chunk ? `chunk=${chunk}` : "whole"}`, async () => {
      const det = await Detector.create(file(m), VAD, { gate, recordTicks: true });
      expect(det.config).toEqual(configFromMetadata(golden(m, name).config));
      const events = await stream(det, audio(name), chunk);
      checkGolden(m, name, gate, det.ticks, events);
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

describe("several wake words (runtime/SPEC.md §5)", () => {
  it("splits composed files into a shared encoder and their heads", () => {
    const [a, b] = Object.values(WORDS).map(splitModel);
    expect(a && b && sameBytes(a.encoder, b.encoder) && !sameBytes(a.head, b.head)).toBe(true);
    for (const bytes of [MODELS.fp32, MODELS.int8, model("encoder.onnx"), VAD]) {
      expect(splitModel(bytes)).toBeNull(); // no `embedding` tensor between two parts
    }
    expect(splitModel(Uint8Array.of(0x3a, 0xff))).toBeNull(); // truncated
  });

  it("the encoder part gives the golden embeddings", async () => {
    const enc = GOLDEN.encoder;
    const a = audio(enc.file);
    const padded = new Float32Array(24000 + a.length);
    padded.set(a, 24000);
    const input = new Float32Array(enc.ticks.length * 24000);
    enc.ticks.forEach((t: number, i: number) =>
      input.set(padded.subarray((t + 1) * 1536, (t + 1) * 1536 + 24000), i * 24000));
    const session = await ort.InferenceSession.create(splitModel(WORDS["word_a.onnx"])!.encoder,
                                                      { executionProviders: ["wasm"] });
    const out = await session.run({ audio: new ort.Tensor("float32", input, [enc.ticks.length, 24000]) });
    expectClose(out.embedding.data as Float32Array, enc.embeddings.flat(), TOL.embedding);
  });

  const runs: [string, boolean, number | null][] = [];
  for (const name of FILES) for (const gate of [true, false]) runs.push([name, gate, null]);
  runs.push(["speech.wav", true, 1000]);
  for (const [name, gate, chunk] of runs) {
    it(`two words sharing an encoder + the stub: ${name} ${gate ? "gate" : "no_gate"} ${chunk ? `chunk=${chunk}` : "whole"}`, async () => {
      const models = ["word_a.onnx", "fp32", "word_b.onnx"];
      const det = await MultiDetector.create(models.map(file), VAD, { gate, recordTicks: true });
      expect(det.groups).toEqual([[0, 2], [1]]);
      expect(det.words.map((w) => w.name)).toEqual(["golden-a", "golden-stub", "golden-b"]);
      const events = await stream(det, audio(name), chunk);
      expect(events).toEqual([...events].sort((x, y) => x.tick - y.tick || x.word - y.word));
      det.words.forEach((word, k) => {
        expect(word.config).toEqual(configFromMetadata(golden(models[k], name).config));
        checkGolden(models[k], name, gate, word.ticks, events.filter((e) => e.word === k));
      });
    });
  }

  it("without sharing, the same results", async () => {
    const a = audio("speech.wav");
    const runs = await Promise.all([true, false].map((shareEncoder) =>
      MultiDetector.create(Object.values(WORDS), VAD, { recordTicks: true, shareEncoder })));
    expect(runs.map((r) => r.groups)).toEqual([[[0, 1]], [[0], [1]]]);
    const events = [await stream(runs[0], a, 1536), await stream(runs[1], a, 1536)]; // one wasm instance: one run at a time
    expect(events[0].map((e) => [e.tick, e.word])).toEqual(events[1].map((e) => [e.tick, e.word]));
    runs[0].words.forEach((w, k) => expectClose(w.ticks.map((t) => t.score), runs[1].words[k].ticks.map((t) => t.score), TOL.fp32));
  });

  it("settings: shared or per word, one tick for all", async () => {
    const paths = Object.values(WORDS);
    let det = await MultiDetector.create(paths, null, { overrides: { threshold: 0.9 } });
    expect(det.words.map((w) => w.config.threshold)).toEqual([0.9, 0.9]);
    det = await MultiDetector.create(paths, null, { overrides: [undefined, { threshold: 0.9 }] });
    expect(det.words.map((w) => w.config.threshold)).toEqual([Number(GOLDEN.words["word_a.onnx"].config.threshold), 0.9]);
    expect((await MultiDetector.create(paths.slice(0, 1))).groups).toEqual([[0]]); // one file runs whole
    await expect(MultiDetector.create(paths, null, { overrides: [undefined, { hop_samples: 3072 }] }))
      .rejects.toThrow("same window_samples and hop_samples");
    await expect(MultiDetector.create(paths, null, { overrides: [undefined] })).rejects.toThrow("overrides");
    await expect(MultiDetector.create([])).rejects.toThrow("at least one");
  });

  it("Detector: `config` replaces the metadata, then overrides apply", async () => {
    const det = await Detector.create(MODELS.fp32, null, {
      config: makeConfig({ threshold: 0.9, smoothing_k: 1 }),
      overrides: { smoothing_n: 4, refractory_ms: undefined },
    });
    expect(det.config).toEqual(makeConfig({ threshold: 0.9, smoothing_k: 1, smoothing_n: 4 }));
  });
});
