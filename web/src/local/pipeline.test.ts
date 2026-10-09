// The whole browser training on the tiny base model, in Node: upload checks on the e2e takes,
// quick mode, the composed file through the runtime, and the bundle.

import { decodeWav, Detector, readOnnxMetadata, SileroVad } from "@wwd/runtime";
import { describe, expect, it } from "vitest";

import { finetuneConfig, loadAugment, loadTrainAssets } from "./assets";
import { bundleFiles } from "./bundle";
import { prepareOther, splitPositives } from "./checks";
import { EncoderSession, SessionEmbedder } from "./embed";
import { E2E, HAVE_FIXTURES, diskFetcher, diskManifest, readFixture } from "./testing";
import { train } from "./train";

describe.skipIf(!HAVE_FIXTURES)("browser quick mode on the tiny base", () => {
  it("trains, calibrates, composes and verifies", async () => {
    const dir = `${E2E}public/base/tiny/`;
    const manifest = diskManifest(dir);
    const fetcher = diskFetcher(dir);
    const assets = await loadTrainAssets(fetcher, manifest);
    const embedder = new SessionEmbedder(
      await EncoderSession.create(await fetcher("encoder.onnx"), manifest.embedding_dim), await loadAugment(fetcher, manifest));
    const vad = await SileroVad.create(assets.vad);
    const wav = (name: string) => decodeWav(readFixture("fixtures", name)).samples;
    const clips = (await splitPositives(wav("take.wav"), vad)).filter((c) => c.accepted);
    expect(clips.length).toBe(6); // as the server cuts the same take (server/tests/test_api.py)
    const other = await prepareOther(wav("other_speech.wav"), "other_speech", vad);
    const ambience = await prepareOther(wav("ambience.wav"), "ambience", vad);

    const stages: string[] = [];
    const { report, model } = await train({
      wakeWord: "abad", language: "fr", targetFaPerHour: 2, positives: clips.map((c) => c.audio),
      positiveIds: clips.map((_, i) => `s${i}`), negatives: [other.audio, ambience.audio],
      negativeSeconds: { other_speech: other.audio.length / 16000, ambience: ambience.audio.length / 16000 },
    }, assets, embedder, (stage) => stages.push(stage), () => false, finetuneConfig(manifest));
    embedder.close();

    expect([...new Set(stages)]).toEqual(["embedding", "cross_validating", "calibrating", "composing", "verifying", "bundling"]);
    const thresholds = report.operating_points.map((p) => p.fp32.threshold);
    expect(thresholds).toEqual([...thresholds].sort((a, b) => b - a)); // more false accepts allowed, lower threshold
    expect(report.cross_validation?.folds).toBe(3);
    const v = report.verification.fp32;
    expect(v.enrollment_total).toBe(6);
    expect(v.gate.false_accepts).toBeLessThanOrEqual(v.no_gate.false_accepts);
    expect((report.verification as unknown as { parity: { file_vs_embedding_max: number } }).parity.file_vs_embedding_max)
      .toBeLessThan(1e-4);

    const meta = readOnnxMetadata(model);
    expect(Number(meta.threshold)).toBe(report.threshold);
    expect(meta.wake_word).toBe("abad");
    expect(meta.base_model_version).toBe("tiny");
    const det = await Detector.create(model, assets.vad);
    await det.process(new Float32Array(48000));

    const files = await bundleFiles({ model, metadata: meta, vad: assets.vad, vadLicense: assets.vadLicense, report,
                                      target: 2, teacher: null });
    expect(files.map(([name]) => name)).toEqual(["model.onnx", "model.json", "vad.onnx", "LICENSE-silero-vad",
                                                 "ATTRIBUTION.md", "README.md"]);
    const modelJson = JSON.parse(new TextDecoder().decode(files[1][1]));
    expect(modelJson.threshold).toBe(meta.threshold);
    console.log(`tiny training: ${JSON.stringify(report.timings_s)}`);
  });
});
