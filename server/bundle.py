"""The download bundle (PLAN.md §4.1): model files, VAD, settings, licenses, a zip of it all."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import onnx

from core.export import set_metadata
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
| `model.int8.onnx` | Same detector with an int8 encoder: a smaller file, same interface. Not faster on a Raspberry Pi 4 or in a browser, and about 2 points less recall: prefer `model.onnx` unless size matters. |
| `vad.onnx` | Silero VAD: the speech gate. |
| `model.json` | Detector settings (also stored in each ONNX file's metadata) and evaluation. |

Each file carries its own threshold, calibrated for {target} false accepts per hour on
{calib_hours:.1f} h of long-form audio: **{threshold:.4f}** for `model.onnx`, **{threshold_int8:.4f}**
for `model.int8.onnx`.{verified}

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


MODELS = {"fp32": "model.onnx", "int8": "model.int8.onnx"}


def _texts(metadata: dict[str, dict[str, str]], report: dict, hashes: dict[str, str],
           target: float) -> dict[str, bytes]:
    """README.md, ATTRIBUTION.md and model.json for model files with this metadata, whose
    thresholds were calibrated for `target` false accepts per hour."""
    meta = metadata["fp32"]
    verified = ""
    if target == report["target_fa_per_hour"]:  # verification ran at the trained target only
        fp32 = report["verification"]["fp32"]
        verified = (f" On {fp32['hours']:.1f} h of held-out audio the float32 file measured "
                    f"{fp32['no_gate']['per_hour']:.2f} false accepts per hour with the gate off "
                    f"and {fp32['gate']['per_hour']:.2f} with the gate on.")
    readme = README.format(
        wake_word=meta["wake_word"], language=meta["language"], target=target,
        threshold=float(meta["threshold"]), threshold_int8=float(metadata["int8"]["threshold"]),
        calib_hours=report["calibration"]["hours"], verified=verified)
    model_json = {
        **meta,  # the float32 file's metadata; each file's own threshold is under "files"
        "files": {
            "model.onnx": {"sha256": hashes["model.onnx"], "threshold": meta["threshold"]},
            "model.int8.onnx": {"sha256": hashes["model.int8.onnx"],
                                "threshold": metadata["int8"]["threshold"]},
            "vad.onnx": {"sha256": hashes["vad.onnx"]},
        },
        "evaluation": {
            "target_fa_per_hour": target,
            "trained_target_fa_per_hour": report["target_fa_per_hour"],
            "calibration_fa_per_hour": {k: report["calibration"][k]["fa_per_hour"]
                                        for k in ("fp32", "int8")},
            "verification": {k: {"hours": v["hours"], "fa_per_hour_gate_off": v["no_gate"]["per_hour"],
                                 "fa_per_hour_gate_on": v["gate"]["per_hour"]}
                             for k, v in report["verification"].items() if k != "parity"},
            "cross_validation_recall": (report["cross_validation"] or {}).get("recall_noisy"),
        },
    }
    return {
        "README.md": readme.encode(),
        "ATTRIBUTION.md": ATTRIBUTION.format(version=meta["base_model_version"]).encode(),
        "model.json": (json.dumps(model_json, indent=2, ensure_ascii=False) + "\n").encode(),
    }


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in BUNDLE_FILES:
            zf.writestr(f"wake-word/{name}", files[name])
    return buf.getvalue()


def write_bundle(out: Path, metadata: dict[str, dict[str, str]], report: dict) -> None:
    """`metadata` per model kind (fp32, int8): the files differ only in their threshold."""
    shutil.copyfile(VAD_FILE, out / "vad.onnx")
    shutil.copyfile(VAD_LICENSE, out / "LICENSE-silero-vad")
    hashes = {name: sha256(out / name) for name in ("model.onnx", "model.int8.onnx", "vad.onnx")}
    for name, data in _texts(metadata, report, hashes, report["target_fa_per_hour"]).items():
        (out / name).write_bytes(data)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    files = {name: (out / name).read_bytes() for name in BUNDLE_FILES}
    tmp = out / "bundle.zip.tmp"
    tmp.write_bytes(_zip(files))
    tmp.replace(out / "bundle.zip")


def retargeted_bundle(out: Path, report: dict, fa_per_hour: float) -> bytes:
    """The bundle at another of the training's operating points: only the threshold in each
    model file's metadata (and the texts describing it) changes."""
    point = next((p for p in report["operating_points"]
                  if abs(p["fa_per_hour"] - fa_per_hour) < 1e-9), None)
    if point is None:
        raise ValueError(f"no operating point at {fa_per_hour} false accepts/h")
    files = {name: (out / name).read_bytes()
             for name in ("vad.onnx", "LICENSE-silero-vad")}
    metadata = {}
    for kind, name in MODELS.items():
        model = onnx.load_from_string((out / name).read_bytes())
        meta = {p.key: p.value for p in model.metadata_props}
        meta["threshold"] = str(point[kind]["threshold"])
        set_metadata(model, meta)
        files[name] = model.SerializeToString()
        metadata[kind] = meta
    report = copy.deepcopy(report)
    for kind in MODELS:
        report["calibration"][kind]["fa_per_hour"] = point[kind]["calibration_fa_per_hour"]
    hashes = {name: hashlib.sha256(files[name]).hexdigest()
              for name in ("model.onnx", "model.int8.onnx", "vad.onnx")}
    return _zip(files | _texts(metadata, report, hashes, point["fa_per_hour"]))
