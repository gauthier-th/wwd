# wwd runtime on a device

Runs a downloaded wake-word bundle with onnxruntime and numpy only, on a Raspberry Pi or any
Python 3.10+ machine. `scripts/pack_runtime.sh` builds this kit: the `runtime/` package with its
golden tests, plus the bundle if you give it one.

## Install (Raspberry Pi OS, 64-bit)

onnxruntime ships wheels for 64-bit ARM (`aarch64`) only: use the 64-bit Raspberry Pi OS. On a Pi 4
or 5, `uname -m` must print `aarch64`.

    sudo apt install python3-venv libportaudio2
    tar xzf wwd-runtime.tar.gz && cd wwd-runtime
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r runtime/requirements.txt pytest

## 1. Parity: same results as everywhere else

    python -m pytest -q runtime/tests

These are the golden vectors every runtime must reproduce (runtime/SPEC.md): the Silero VAD against
its official wrapper, then fp32 and int8 scores and trigger events on recorded speech. On ARM, int8
runs other kernels than on x86 or in a browser; the test bounds how far they may drift.

## 2. Microphone

    python -m sounddevice                                   # list the input devices
    python -m runtime.listen --record mic-check.wav --seconds 5 [--device N]

A take recorded here is also the best training data for this device: import it in the app
("Import a file instead").

## 3. Benchmark: CPU and latency

    python -m runtime.bench bundle/model.onnx bundle/model.int8.onnx --vad bundle/vad.onnx --json bench.json

Streams 60 s of speech one tick (96 ms) at a time, as live, through each file with the speech gate
on and off. "CPU % (real time)" is the share of one core the detector needs to keep up.

## 4. Field test: 24 h in a real room

    python -m runtime.listen bundle/model.onnx --vad bundle/vad.onnx --log fieldtest [--device N]

Leave it running for 24 hours where the device will live: TV on for part of the day, people
talking, normal life. Don't say the wake word, or write down when you did. Keep it running when you
log out with `tmux`, or as a user service:

    systemd-run --user --unit wwd-field --working-directory "$PWD" \
      .venv/bin/python -m runtime.listen bundle/model.onnx --vad bundle/vad.onnx --log fieldtest
    journalctl --user -fu wwd-field      # watch; `systemctl --user stop wwd-field` ends it cleanly

Then:

    python -m runtime.fieldreport fieldtest [--true 4,9]

Every trigger has a clip in `fieldtest/clips/` (3 s before, 1 s after). Listen to them; the
numbers of real wake words go to `--true`, and every other trigger is a false accept. The report also
replays the logged scores at other thresholds, so one run shows the whole trade-off. Copy the
`fieldtest/` directory back to analyze it elsewhere.

## 5. Recall in the room (optional, 5 minutes)

With the TV on at a normal level, say the wake word 20 times from about 1 m, pausing a few seconds
between each, then 20 times from about 3 m:

    python -m runtime.listen bundle/model.onnx --vad bundle/vad.onnx --log recall-1m   # Ctrl+C after 20
    python -m runtime.fieldreport recall-1m     # triggers / 20 = recall at 1 m
