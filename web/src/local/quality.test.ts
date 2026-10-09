// Browser training on real held-out words (scripts/web_quality.py prepares them and scores
// the results). Runs only when WWD_QUALITY points at the prepared directory; the web assets
// come from web/public/base (scripts/export_web_assets.py) unless WWD_WEB_BASE says otherwise.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, it } from "vitest";

import { finetuneConfig, loadAugment, loadTrainAssets, type Manifest } from "./assets";
import { EncoderSession, SessionEmbedder } from "./embed";
import { diskFetcher } from "./testing";
import { train } from "./train";

const DIR = process.env.WWD_QUALITY;
const BASE = process.env.WWD_WEB_BASE ?? fileURLToPath(new URL("../../public/base/", import.meta.url));

describe.skipIf(!DIR)("browser training on held-out words", () => {
  it("trains every prepared word", { timeout: 24 * 3600_000 }, async () => {
    const index = JSON.parse(readFileSync(join(BASE, "index.json"), "utf8"));
    const dir = join(BASE, index.default);
    const manifest: Manifest = JSON.parse(readFileSync(join(dir, "manifest.json"), "utf8"));
    const fetcher = diskFetcher(dir);
    const assets = await loadTrainAssets(fetcher, manifest);
    const embedder = new SessionEmbedder(
      await EncoderSession.create(await fetcher("encoder.onnx"), manifest.embedding_dim), await loadAugment(fetcher, manifest));
    for (const name of JSON.parse(readFileSync(join(DIR!, "words.json"), "utf8")) as string[]) {
      const d = join(DIR!, name);
      if (existsSync(join(d, "report.json"))) continue; // resumable
      const meta = JSON.parse(readFileSync(join(d, "word.json"), "utf8"));
      const all = new Float32Array(new Uint8Array(readFileSync(join(d, "enroll.f32"))).buffer);
      let off = 0;
      const positives = (meta.enroll_lengths as number[]).map((n) => all.slice(off, (off += n)));
      const { report, model } = await train({
        wakeWord: meta.word, language: meta.lang, targetFaPerHour: 0.5, positives,
        positiveIds: positives.map((_, i) => String(i)), negatives: [], negativeSeconds: {},
      }, assets, embedder, () => {}, () => false, finetuneConfig(manifest));
      writeFileSync(join(d, "model.onnx"), model);
      writeFileSync(join(d, "report.json"), JSON.stringify(report, null, 2) + "\n");
      console.log(`${name} ${meta.word}: threshold ${report.threshold.toFixed(4)}, ${report.timings_s.total}s`);
    }
    embedder.close();
  });
});
