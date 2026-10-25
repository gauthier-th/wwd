# wwd runtime on a device

Runs a downloaded wake-word model with onnxruntime and numpy only, on a Raspberry Pi or any
Python 3.10+ machine. The steps below test the device with your model: put its files in
`bundle/` (`unzip -j model.zip -d bundle`).

## Install

On a Raspberry Pi, use the 64-bit Raspberry Pi OS: onnxruntime has no wheels for 32-bit ARM.

    sudo apt install git python3-venv libportaudio2
    git clone --depth 1 https://github.com/gauthier-th/wwd && cd wwd
    python3 -m venv .venv && . .venv/bin/activate
    pip install -r runtime/requirements.txt pytest

No git on the device? On another machine, `scripts/pack_runtime.sh model.zip` builds
`dist/wwd-runtime.tar.gz` with the runtime and the model; copy and extract it instead.

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

    python -m runtime.bench bundle/model.onnx --vad bundle/vad.onnx --json bench.json

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

## Several wake words

Give the `model.onnx` of each wake word. Any bundle's `vad.onnx` works: it's the same file in all
of them.

    python -m runtime.listen jarvis/model.onnx lights/model.onnx --vad jarvis/vad.onnx
    python -m runtime.bench jarvis/model.onnx lights/model.onnx --vad jarvis/vad.onnx --together

Models trained on the same base model share their encoder, so the detector runs it once per tick
for all of them. `bench --together` then shows "1 pass" and about the CPU of a single wake word.
These run side by side instead, each with its own pass:

- models trained on different base models;
- a `model.int8.onnx` listening alongside a `model.onnx`.

With `--log fieldtest`, each wake word gets its own log inside `fieldtest/`. The report covers
each word, then all of them together: that's how often the device wakes up by mistake. Mark the
real wake words per word:

    python -m runtime.fieldreport fieldtest                          # every word, and the total
    python -m runtime.fieldreport fieldtest/1-hey-jarvis --true 4,9  # one word

## In your own project

Copy `runtime/` into your project under any name (`js/` can go): its modules import each other
relatively. With the folder copied as `wakeword/`:

    from wakeword import Detector
    python -m wakeword.listen model.onnx --vad vad.onnx
    python -m pytest -q wakeword/tests
