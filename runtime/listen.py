"""Listen on the microphone and print wake-word events, or record a take for the app.

    python -m runtime.listen model.onnx --vad vad.onnx [--no-gate] [--threshold 0.6] [--show]
    python -m runtime.listen --record take.wav [--seconds 30]

Recording on the device itself (a Pi and its own microphone) gives the closest training data:
upload the file with the app's "Import a file instead".
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from runtime.detector import Detector
from runtime.spec import SAMPLE_RATE
from runtime.wav import write_wav


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model", nargs="?", help="model.onnx from the downloaded bundle")
    p.add_argument("--vad", help="Silero VAD ONNX file; enables the speech gate")
    p.add_argument("--no-gate", action="store_true", help="score every tick even without speech")
    p.add_argument("--threshold", type=float, help="override the calibrated threshold")
    p.add_argument("--device", help="sounddevice input device name or index")
    p.add_argument("--show", action="store_true", help="print every tick score")
    p.add_argument("--record", metavar="WAV", help="record 16 kHz audio to this file instead")
    p.add_argument("--seconds", type=float, default=30.0, help="length of a --record take")
    args = p.parse_args(argv)
    if not args.model and not args.record:
        p.error("give a model to listen with, or --record FILE")

    import sounddevice as sd  # only the mic CLI needs it

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    if args.record:
        return record(sd, args.record, args.seconds, device)

    det = Detector(
        args.model,
        args.vad,
        gate=not args.no_gate,
        overrides={"threshold": args.threshold},
        record_ticks=args.show,
    )
    hop = det.config.hop_samples
    chunks: queue.Queue[np.ndarray] = queue.Queue()

    def on_audio(indata, frames, t, status):  # PortAudio thread: hand off, never block
        if status:
            print(status, file=sys.stderr)
        chunks.put(indata[:, 0].copy())

    name = det.metadata.get("wake_word", "?")
    print(f"listening for '{name}' (threshold {det.config.threshold:.3f}); Ctrl+C to stop")
    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=hop,
        device=device, callback=on_audio,
    ):
        try:
            while True:
                for ev in det.process(chunks.get()):
                    print(f"{time.strftime('%H:%M:%S')}  WAKE  score={ev.score:.3f}", flush=True)
                if args.show:
                    for tick in det.ticks:
                        gate = "open" if tick.gate_open else "shut"
                        print(f"  tick {tick.index:6d} vad={tick.vad_max:.2f} {gate} "
                              f"score={tick.score:.3f}")
                    det.ticks.clear()
        except KeyboardInterrupt:
            return 0


def record(sd, path: str, seconds: float, device) -> int:
    print(f"recording {seconds:g} s to {path}; Ctrl+C to stop early")
    frames: list[np.ndarray] = []
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device) as stream:
            while sum(len(f) for f in frames) < seconds * SAMPLE_RATE:
                block, _ = stream.read(SAMPLE_RATE // 10)
                frames.append(block[:, 0].copy())
    except KeyboardInterrupt:
        pass
    audio = np.concatenate(frames) if frames else np.zeros(0, np.float32)
    write_wav(path, audio[: int(seconds * SAMPLE_RATE)])
    print(f"wrote {len(audio) / SAMPLE_RATE:.1f} s; import it in the app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
