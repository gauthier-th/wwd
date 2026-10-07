import { describe, expect, it } from "vitest";

import { SpeechSegmenter, speechRegions } from "./segments";

const run = (...parts: [number, number][]) => parts.flatMap(([p, n]) => Array<number>(n).fill(p));

describe("speechRegions (mirrors server/audio.py speech_regions)", () => {
  it("bridges short dips and drops short blips, like the server test", () => {
    // server/tests/test_audio.py::test_speech_regions_hysteresis
    const probs = run([0, 10], [0.9, 10], [0.2, 3], [0.9, 5], [0, 20], [0.9, 2], [0, 15]);
    expect(speechRegions(probs)).toEqual([[10 * 512, 28 * 512]]);
  });

  it("splits on 11 quiet frames (0.35 s) but not on 10", () => {
    expect(speechRegions(run([0.9, 8], [0, 10], [0.9, 8], [0, 12]))).toEqual([[0, 26 * 512]]);
    expect(speechRegions(run([0.9, 8], [0, 11], [0.9, 8], [0, 12]))).toEqual([
      [0, 8 * 512],
      [19 * 512, 27 * 512],
    ]);
  });

  it("probabilities between the two thresholds keep speech going", () => {
    expect(speechRegions(run([0.9, 6], [0.4, 30], [0, 12]))).toEqual([[0, 36 * 512]]);
  });

  it("closes a region still open at the end", () => {
    expect(speechRegions(run([0, 3], [0.9, 10]))).toEqual([[3 * 512, 13 * 512]]);
    expect(speechRegions(run([0, 3], [0.9, 10], [0, 4]))).toEqual([[3 * 512, 13 * 512]]);
  });
});

describe("SpeechSegmenter (streaming)", () => {
  it("reports regions as they close and whether speech is ongoing", () => {
    const seg = new SpeechSegmenter();
    const closed = run([0, 5], [0.9, 10], [0, 11]).map((p) => seg.push(p));
    expect(closed.filter(Boolean)).toEqual([{ start: 5, end: 15 }]);
    expect(seg.speaking).toBe(false);
    seg.push(0.9);
    expect(seg.speaking).toBe(true);
    expect(seg.regions).toHaveLength(1);
  });
});
