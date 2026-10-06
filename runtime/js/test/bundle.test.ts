// Parity of a server-produced bundle (the M2 smoke run, scripts/m2_smoke.sh): this runtime on
// onnxruntime-web wasm against the Python runtime's per-tick values, written by
// scripts/check_bundle.py. Skipped unless WWD_BUNDLE points at such an unzipped bundle.

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { configFromMetadata, decodeWav, Detector, readOnnxMetadata, type WakeEvent } from "../src/index.js";

const BUNDLE = process.env.WWD_BUNDLE;
const read = (name: string) => new Uint8Array(readFileSync(join(BUNDLE!, name)));
const PARITY = BUNDLE ? JSON.parse(readFileSync(join(BUNDLE, "parity", "parity.json"), "utf8")) : null;

function maxDiff(got: number[], want: number[]): number {
  expect(got.length).toBe(want.length);
  return got.reduce((worst, g, i) => Math.max(worst, Math.abs(g - want[i])), 0);
}

describe.skipIf(!BUNDLE)("server bundle parity (onnxruntime-web wasm)", () => {
  const models = { fp32: "model.onnx", int8: "model.int8.onnx" } as const;

  it("both model files carry their calibrated settings", () => {
    for (const [kind, file] of Object.entries(models)) {
      const meta = readOnnxMetadata(read(file));
      expect(meta).toEqual(PARITY.config[kind]);
      expect(configFromMetadata(meta).threshold).toBe(Number(PARITY.config[kind].threshold));
    }
  });

  for (const name of Object.keys(PARITY?.files ?? {})) {
    for (const kind of ["fp32", "int8"] as const) {
      for (const gate of [true, false]) {
        it(`${name} ${kind} ${gate ? "gate" : "no_gate"}`, async () => {
          const want = PARITY.files[name];
          const tol = PARITY.tolerance;
          const { samples } = decodeWav(read(join("parity", name)));
          const det = await Detector.create(read(models[kind]), read("vad.onnx"), { gate, recordTicks: true });
          const events: WakeEvent[] = await det.process(samples);

          const gateOpen: boolean[] = gate ? want.gate_open : want.gate_open.map(() => true);
          expect(det.ticks.map((t) => t.gateOpen)).toEqual(gateOpen);
          expect(maxDiff(det.ticks.map((t) => t.vadMax), want.vad_max)).toBeLessThanOrEqual(tol.vad);
          const got = det.ticks.map((t) => t.score);
          const expected = (want.scores[kind] as number[]).map((s, i) => (gateOpen[i] ? s : 0));
          const diffs = got.map((g, i) => Math.abs(g - expected[i]));
          const worst = Math.max(...diffs);
          if (kind === "fp32") {
            expect(worst, `max |score diff| ${worst}`).toBeLessThanOrEqual(tol.fp32);
          } else {
            const mean = diffs.reduce((a, d) => a + d, 0) / diffs.length;
            const flips = diffs.filter((d) => d > tol.int8_flip).length;
            expect(mean, "mean |score diff|").toBeLessThanOrEqual(tol.int8_mean);
            expect(flips / diffs.length, `${flips} ticks off by > ${tol.int8_flip}`).toBeLessThanOrEqual(
              tol.int8_flip_share);
            expect(worst, `max |score diff| ${worst}`).toBeLessThanOrEqual(tol.int8_max);
          }
          // Ticks the two runtimes put on opposite sides of the threshold: the only way events
          // can differ, given the logic itself is proven identical by the golden tests.
          const threshold = Number(PARITY.config[kind].threshold);
          const straddle = got.flatMap((g, i) => ((g >= threshold) !== (expected[i] >= threshold) ? [i] : []));
          if (straddle.length === 0) {
            expect(events.map((e) => e.tick)).toEqual(want.events[kind][gate ? "gate" : "no_gate"]);
          } else {
            console.warn(`${name} ${kind}: ticks ${straddle} straddle the threshold; events not compared`);
          }
        });
      }
    }
  }
});
