"""Listen on the microphone and print wake-word events.

    python -m runtime.listen model.onnx --vad vad.onnx [--no-gate] [--threshold 0.6] [--show]
"""

from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np

from runtime.detector import Detector
from runtime.spec import SAMPLE_RATE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("model")
    p.add_argument("--vad", help="Silero VAD ONNX file; enables the speech gate")
    p.add_argument("--no-gate", action="store_true", help="score every tick even without speech")
    p.add_argument("--threshold", type=float, help="override the calibrated threshold")
    p.add_argument("--device", help="sounddevice input device name or index")
    p.add_argument("--show", action="store_true", help="print every tick score")
    args = p.parse_args(argv)

    import sounddevice as sd  # only the mic CLI needs it

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
    device = int(args.device) if args.device and args.device.isdigit() else args.device
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


if __name__ == "__main__":
    sys.exit(main())
