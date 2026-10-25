// The download bundle, as server/bundle.py writes it, minus the int8 file: the browser builds
// model.onnx only (the one to deploy everywhere, PLAN.md §12.4).

import type { Report } from "../api";
import { zip } from "./zip";

export const BUNDLE_DIR = "wake-word";

const ATTRIBUTION = (version: string, teacher: string) => `# Attribution

These wake-word models were built on base model \`${version}\`, trained with the data below.

## Data

- **Multilingual Spoken Words Corpus** (MLCommons), [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  Mazumder et al., "Multilingual Spoken Words Corpus", NeurIPS 2021 Datasets and Benchmarks.
  <https://mlcommons.org/datasets/multilingual-spoken-words/>
  Used to train the base model, as fine-tuning negatives and as calibration audio.
- **MUSAN** (OpenSLR 17), noise, music and babble for augmentation and calibration.
  Snyder, Chen and Povey, "MUSAN: A Music, Speech, and Noise Corpus", 2015.
  <https://www.openslr.org/17/>
- **RIRS_NOISES** (OpenSLR 28), room impulse responses for augmentation, Apache-2.0.
  <https://www.openslr.org/28/>
- **Multilingual LibriSpeech** (dev and test sets of de, fr, es, it, pl), CC-BY 4.0, and
  **LibriSpeech** (test-clean, test-other; OpenSLR 12), CC-BY 4.0: long-form speech used to
  calibrate and verify the false-accept rate.

${teacher}## Software

- **Silero VAD** v6.2.3 (\`vad.onnx\`), MIT license: see \`LICENSE-silero-vad\`.
  <https://github.com/snakers4/silero-vad>
`;

const TEACHER = (model: string) => `## Models

- **XLS-R 300M** (\`${model}\`, Meta AI), Apache-2.0. The base model's encoder was distilled
  from it; no XLS-R weights are in these files. Babu et al., "XLS-R: Self-supervised
  Cross-lingual Speech Representation Learning at Scale", 2021.
  <https://huggingface.co/${model}>

`;

const REPO = "https://github.com/gauthier-th/wwd";

const README = (p: { wakeWord: string; language: string; target: number; threshold: number; hours: number;
                     verified: string }) => `# Wake word "${p.wakeWord}" (${p.language})

| File | What it is |
|---|---|
| \`model.onnx\` | Detector, float32. Input \`audio\` [N, 24000] (1.5 s at 16 kHz), output \`score\` [N]. |
| \`vad.onnx\` | Silero VAD: the speech gate. |
| \`model.json\` | Detector settings (also stored in the ONNX file's metadata) and evaluation. |

Trained in the browser. The threshold, calibrated for ${p.target} false accepts per hour on
${p.hours.toFixed(1)} h of long-form audio, is **${p.threshold.toFixed(4)}**.${p.verified}

## Run it

The runtime is in ${REPO}: \`runtime/\` for Python (a Raspberry Pi or any computer),
\`runtime/js/\` for browsers and Node. From a clone of it:

    pip install -r runtime/requirements.txt
    python -m runtime.listen path/to/model.onnx --vad path/to/vad.onnx

To listen for several wake words at once, give the \`model.onnx\` of each. Wake words trained on
the same base model share most of the work, so each extra one costs almost no CPU. Its README
shows how to use the detector in your own code.
`;

async function sha256(data: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", data as Uint8Array<ArrayBuffer>);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

export interface BundleInputs {
  model: Uint8Array; // model.onnx with the metadata of the chosen operating point
  metadata: Record<string, string>;
  vad: Uint8Array;
  vadLicense: Uint8Array;
  report: Report;
  target: number; // false accepts per hour the metadata's threshold was calibrated for
  teacher: string | null;
}

export async function bundleFiles(b: BundleInputs): Promise<[string, Uint8Array][]> {
  const { report, metadata: meta, target } = b;
  const fp32 = report.verification.fp32;
  const verified = target === report.target_fa_per_hour
    ? ` On ${fp32.hours.toFixed(1)} h of held-out audio it measured ${fp32.no_gate.per_hour.toFixed(2)} false ` +
      `accepts per hour with the gate off and ${fp32.gate.per_hour.toFixed(2)} with the gate on.`
    : "";
  const modelJson = {
    ...meta,
    files: {
      "model.onnx": { sha256: await sha256(b.model), threshold: meta.threshold },
      "vad.onnx": { sha256: await sha256(b.vad) },
    },
    evaluation: {
      target_fa_per_hour: target,
      trained_target_fa_per_hour: report.target_fa_per_hour,
      calibration_fa_per_hour: { fp32: report.calibration.fp32.fa_per_hour },
      verification: { fp32: { hours: fp32.hours, fa_per_hour_gate_off: fp32.no_gate.per_hour,
                              fa_per_hour_gate_on: fp32.gate.per_hour } },
      cross_validation_recall: report.cross_validation?.recall_noisy ?? null,
    },
  };
  const text = (s: string) => new TextEncoder().encode(s);
  return [
    ["model.onnx", b.model],
    ["model.json", text(JSON.stringify(modelJson, null, 2) + "\n")],
    ["vad.onnx", b.vad],
    ["LICENSE-silero-vad", b.vadLicense],
    ["ATTRIBUTION.md", text(ATTRIBUTION(meta.base_model_version, b.teacher ? TEACHER(b.teacher) : ""))],
    ["README.md", text(README({ wakeWord: meta.wake_word, language: meta.language, target,
                                threshold: Number(meta.threshold), hours: report.calibration.hours, verified }))],
  ];
}

export async function bundleZip(b: BundleInputs): Promise<Uint8Array> {
  return zip((await bundleFiles(b)).map(([name, data]) => [`${BUNDLE_DIR}/${name}`, data]));
}
