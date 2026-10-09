// The ported maths against Python's own results (scripts/make_web_fixtures.py).

import { readOnnxMetadata, ort } from "@wwd/runtime";
import { describe, expect, it } from "vitest";

import { composeModel } from "./compose";
import { calibrate, countTriggers, foldsOf, normalizeWord, prototypeHead, quantile, speed, streamRecall, type Rows,
  FINETUNE } from "./finetune";
import { diskManifest, E2E, golden, HAVE_FIXTURES, readFixture } from "./testing";

const rows = (list: number[][]): Rows => ({ data: Float32Array.from(list.flat()), rows: list.length, dim: list[0].length });

function testWindow(i: number, n = 24000): Float32Array {
  return Float32Array.from({ length: n }, (_, j) => {
    const t = j / 16000;
    return 0.1 * Math.sin(2 * Math.PI * (200 + 150 * i) * t) + 0.02 * (t - 0.75);
  });
}

describe.skipIf(!HAVE_FIXTURES)("ported maths match Python", () => {
  const g = HAVE_FIXTURES ? golden() : null;

  it("counts triggers like core.calibrate.count_triggers", () => {
    const t = g.triggers;
    const scores = Float32Array.from(t.scores);
    const cfg = { k: t.k, n: t.n, refractoryTicks: t.refractory_ticks };
    for (const [thr, count] of Object.entries(t.counts)) expect(countTriggers(scores, t.starts, Number(thr), cfg)).toBe(count);
    for (const [fa, [thr, measured]] of Object.entries(t.calibrate) as [string, number[]][]) {
      const [jsThr, jsFa] = calibrate(scores, t.starts, t.hours, Number(fa), cfg);
      expect(jsThr).toBeCloseTo(thr, 12);
      expect(jsFa).toBeCloseTo(measured, 9);
    }
    const ignore = new Uint8Array(scores.length).fill(1, t.ignore[0], t.ignore[1]);
    const [thr, fa] = calibrate(scores, t.starts, t.hours, 100, cfg, ignore);
    expect(thr).toBeCloseTo(t.calibrate_ignore[0], 12);
    expect(fa).toBeCloseTo(t.calibrate_ignore[1], 9);
  });

  it("scores stream recall, quantiles, folds and words the same way", () => {
    const cfg = { k: g.triggers.k, n: g.triggers.n, refractoryTicks: g.triggers.refractory_ticks };
    const shifted = Float32Array.from(g.recall.scores.flat());
    for (const [thr, r] of Object.entries(g.recall.results)) expect(streamRecall(shifted, 6, Number(thr), cfg)).toBeCloseTo(r as number, 12);
    for (const [q, v] of Object.entries(g.quantile.results)) expect(quantile(g.quantile.values, Number(q))).toBeCloseTo(v as number, 12);
    for (const [n, folds] of Object.entries(g.folds)) expect(foldsOf(Number(n))).toEqual(folds);
    for (const [w, norm] of Object.entries(g.normalize)) expect(normalizeWord(w)).toBe(norm);
    const clip = Float32Array.from(g.speed.clip);
    for (const [f, out] of Object.entries(g.speed.results) as [string, number[]][]) {
      const js = speed(clip, Number(f));
      expect(js.length).toBe(out.length);
      // torch computes sample positions in float32: its error grows with the index
      out.forEach((v, i) => expect(Math.abs(js[i] - v)).toBeLessThan(1e-3));
    }
  });

  it("fits the pushed-prototype head like core.finetune.prototype_head", () => {
    const h = g.head;
    const head = prototypeHead(Float32Array.from(h.proto), rows(h.pos), rows(h.neg),
      { ...FINETUNE, hardNegatives: h.hard_negatives, rocchio: h.rocchio, scoreScale: h.score_scale });
    h.weight.forEach((w: number, i: number) => expect(head.direction[i] * head.scale).toBeCloseTo(w, 4));
    expect(head.bias).toBeCloseTo(h.bias, 4);
  });

  it("composes the same model.onnx as core.export.compose_head", async () => {
    const dir = `${E2E}public/base/tiny/`;
    const manifest = diskManifest(dir);
    const c = g.compose;
    const model = composeModel(readFixture("public", "base", "tiny", "model_template.onnx"), manifest.template,
      { direction: Float32Array.from(c.direction), scale: c.scale, bias: c.bias }, c.metadata);
    expect(readOnnxMetadata(model)).toEqual(c.metadata);
    const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
    const windows = new Float32Array(3 * 24000);
    for (let i = 0; i < 3; i++) windows.set(testWindow(i), i * 24000);
    const res = await session.run({ [session.inputNames[0]]: new ort.Tensor("float32", windows, [3, 24000]) });
    const scores = res[session.outputNames[0]].data as Float32Array;
    c.scores.forEach((s: number, i: number) => expect(Math.abs(scores[i] - s)).toBeLessThan(1e-5));
  });
});
