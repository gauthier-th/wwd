"""The download bundle (PLAN.md §4.1): model files, VAD, settings, licenses, a zip of it all."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from server.basemodel import VAD_FILE, VAD_LICENSE, sha256

BUNDLE_FILES = ("model.onnx", "model.int8.onnx", "model.json", "vad.onnx", "LICENSE-silero-vad",
                "ATTRIBUTION.md", "README.md")
ARTIFACTS = (*BUNDLE_FILES, "report.json", "bundle.zip")

ATTRIBUTION = """\
# Attribution

These wake-word models were built on base model `{version}`, trained with the data below.

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

## Software

- **Silero VAD** v6.2.3 (`vad.onnx`), MIT license: see `LICENSE-silero-vad`.
  <https://github.com/snakers4/silero-vad>
"""

README = """\
# Wake word "{wake_word}" ({language})

| File | What it is |
|---|---|
| `model.onnx` | Detector, float32. Input `audio` [N, 24000] (1.5 s at 16 kHz), output `score` [N]. |
| `model.int8.onnx` | Same detector with an int8 encoder: smaller, same interface. |
| `vad.onnx` | Silero VAD: the speech gate. |
| `model.json` | Detector settings (also stored in each ONNX file's metadata) and evaluation. |

Each file carries its own threshold, calibrated for {target} false accepts per hour on
{calib_hours:.1f} h of long-form audio: **{threshold:.4f}** for `model.onnx`, **{threshold_int8:.4f}**
for `model.int8.onnx`. On {verify_hours:.1f} h of held-out audio the float32 file measured
{fa_off:.2f} false accepts per hour with the gate off and {fa_on:.2f} with the gate on.

## Python (Raspberry Pi, any computer)

Copy the `runtime/` folder of the wwd repository, then:

    pip install -r runtime/requirements.txt
    python -m runtime.listen model.onnx --vad vad.onnx

## Browser

`@wwd/runtime` runs the same files on onnxruntime-web (wasm):

    const detector = await Detector.create(modelBytes, vadBytes);
    const events = await detector.process(samples16k);

The detector logic (ticks, speech gate, smoothing, refractory period) is specified in
`runtime/SPEC.md`.
"""


def write_bundle(out: Path, metadata: dict[str, dict[str, str]], report: dict) -> None:
    """`metadata` per model kind (fp32, int8): the files differ only in their threshold."""
    shutil.copyfile(VAD_FILE, out / "vad.onnx")
    shutil.copyfile(VAD_LICENSE, out / "LICENSE-silero-vad")
    meta = metadata["fp32"]
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(version=meta["base_model_version"]))
    fp32 = report["verification"]["fp32"]
    (out / "README.md").write_text(README.format(
        wake_word=meta["wake_word"], language=meta["language"],
        threshold=float(meta["threshold"]), threshold_int8=float(metadata["int8"]["threshold"]),
        target=report["target_fa_per_hour"],
        calib_hours=report["calibration"]["hours"], verify_hours=fp32["hours"],
        fa_off=fp32["no_gate"]["per_hour"], fa_on=fp32["gate"]["per_hour"]))
    model_json = {
        **meta,  # the float32 file's metadata; each file's own threshold is under "files"
        "files": {
            "model.onnx": {"sha256": sha256(out / "model.onnx"), "threshold": meta["threshold"]},
            "model.int8.onnx": {"sha256": sha256(out / "model.int8.onnx"),
                                "threshold": metadata["int8"]["threshold"]},
            "vad.onnx": {"sha256": sha256(out / "vad.onnx")},
        },
        "evaluation": {
            "target_fa_per_hour": report["target_fa_per_hour"],
            "calibration_fa_per_hour": {k: report["calibration"][k]["fa_per_hour"]
                                        for k in ("fp32", "int8")},
            "verification": {k: {"hours": v["hours"], "fa_per_hour_gate_off": v["no_gate"]["per_hour"],
                                 "fa_per_hour_gate_on": v["gate"]["per_hour"]}
                             for k, v in report["verification"].items() if k != "parity"},
            "cross_validation_recall": (report["cross_validation"] or {}).get("recall_noisy"),
        },
    }
    (out / "model.json").write_text(json.dumps(model_json, indent=2, ensure_ascii=False) + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    tmp = out / "bundle.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in BUNDLE_FILES:
            zf.write(out / name, f"wake-word/{name}")
    tmp.replace(out / "bundle.zip")
