"""Run a downloaded bundle through the Python runtime and write the per-tick values that the
JS runtime must reproduce (runtime/js/test/bundle.test.ts, onnxruntime-web wasm).

    python -m scripts.check_bundle <bundle-dir> a.wav b.wav [--labels meta.json]

Writes <bundle-dir>/parity/: the WAVs and parity.json, in the golden-vector format of
runtime/testdata/golden.json. Uses the runtime only (onnxruntime + numpy), like a device would.
With --labels, also reports detection of the labelled wake-word clips.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from runtime import Detector
from runtime.wav import read_wav

# Cross-runtime contract for real models on long recordings (PLAN.md §3). Looser than the
# golden vectors' in two places, both measured in the M2 smoke run:
#  - VAD: its LSTM state accumulates float differences over minutes (1.2e-5 after 2 min);
#  - int8: rare one-step requantization flips reach 0.035 on a real encoder (2 ticks of
#    1572), while 99% of ticks agree within 1e-6. So: bound the mean, the share of large
#    flips and the worst flip, and require identical events unless a tick's scores straddle
#    the threshold across runtimes (the only way int8 drift can change an event).
TOLERANCE = {"vad": 1e-4, "fp32": 1e-4, "int8_mean": 1e-3, "int8_flip": 2e-2,
             "int8_flip_share": 0.01, "int8_max": 0.1}
MODELS = {"fp32": "model.onnx", "int8": "model.int8.onnx"}


def run(bundle: Path, kind: str, audio: np.ndarray, gate: bool) -> tuple[Detector, list]:
    det = Detector(bundle / MODELS[kind], bundle / "vad.onnx", gate=gate, record_ticks=True)
    return det, det.process(audio)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("bundle", type=Path)
    p.add_argument("wavs", type=Path, nargs="+")
    p.add_argument("--labels", type=Path, help="meta.json with test_spans per file")
    args = p.parse_args(argv)
    out = args.bundle / "parity"
    out.mkdir(exist_ok=True)
    spans = json.loads(args.labels.read_text())["test_spans"] if args.labels else {}

    files, report, config = {}, {}, {}
    for wav in args.wavs:
        audio, rate = read_wav(wav)
        assert rate == 16000, wav
        shutil.copyfile(wav, out / wav.name)
        entry: dict = {"scores": {}, "events": {}}
        for kind in MODELS:
            (off, off_events), (on, on_events) = (run(args.bundle, kind, audio, False),
                                                  run(args.bundle, kind, audio, True))
            config[kind] = dict(off.metadata)  # each file carries its own threshold
            scores = np.array([t.score for t in off.ticks])
            gate_open = [t.gate_open for t in on.ticks]
            on_scores = np.array([t.score for t in on.ticks])
            assert np.array_equal(np.where(gate_open, scores, 0.0), on_scores)  # stateless model
            entry["vad_max"] = [t.vad_max for t in on.ticks]
            entry["gate_open"] = gate_open
            entry["scores"][kind] = scores.tolist()
            entry["events"][kind] = {"gate": [e.tick for e in on_events],
                                     "no_gate": [e.tick for e in off_events]}
            if wav.name in spans:
                window = off.config.window_samples
                for gate, events in (("gate", on_events), ("no_gate", off_events)):
                    # an event counts for a clip if its window, ending at e.sample, overlaps it
                    hit = [any(a < e.sample <= b + window for e in events) for a, b in spans[wav.name]]
                    inside = sum(any(a < e.sample <= b + window for a, b in spans[wav.name])
                                 for e in events)
                    report[f"{wav.name} {kind} {gate}"] = {
                        "detected": f"{sum(hit)}/{len(hit)}", "recall": round(sum(hit) / len(hit), 3),
                        "other_events": len(events) - inside}
        entry["vad"] = []  # per-frame VAD is covered by vad_max; keep the file small
        files[wav.name] = entry
        print(f"{wav.name}: {len(entry['gate_open'])} ticks, events "
              + ", ".join(f"{k} {len(v['gate'])} gate / {len(v['no_gate'])} no gate"
                          for k, v in entry["events"].items()))

    (out / "parity.json").write_text(json.dumps(
        {"tolerance": TOLERANCE, "config": config, "files": files}, ensure_ascii=False) + "\n")
    for key, value in report.items():
        print(f"{key}: {value}")
    print(f"wrote {out / 'parity.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
