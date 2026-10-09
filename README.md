# Wake Word Trainer

Make your own wake word from two minutes of recordings, in 12 languages. Training runs in your
browser and gives you a small ONNX model that listens on a Raspberry Pi, a computer or a web page.

**→ [Open the app](https://gauthier-th.github.io/wwd/)**

No install, no account, no server: your recordings and models stay in your browser.

## Make a wake word

1. Enter the wake word and its language. Two words ("hey jarvis") work better than one short word.
2. Record with the microphone the wake word will be used with: the wake word 10 times, a
   paragraph read aloud, and 15 seconds of room sound.
3. Train: under a minute, about 15 seconds on a recent desktop. The first time also downloads the
   base model (230 MB, kept for later visits). Keep the tab open while it trains.
4. Try it live, adjust the sensitivity if needed, and download the zip.

Languages: English, German, French, Catalan, Spanish, Persian, Italian, Russian, Polish, Basque,
Welsh and Kinyarwanda (much less accurate than the others).

## Use the model

The zip holds `model.onnx` (the detector, its settings included) and `vad.onnx` (a voice
detector, so the model only runs while someone speaks).

### Python: Raspberry Pi or any computer

```sh
git clone --depth 1 https://github.com/gauthier-th/wwd && cd wwd
python3 -m venv .venv && . .venv/bin/activate
pip install -r runtime/requirements.txt
python -m runtime.listen path/to/model.onnx --vad path/to/vad.onnx   # prints each detection
```

On a Raspberry Pi, use the 64-bit OS and `sudo apt install libportaudio2` first. In your own code:

```python
import sounddevice as sd
from runtime.detector import Detector

detector = Detector("model.onnx", "vad.onnx")
with sd.InputStream(samplerate=16000, channels=1, dtype="float32") as mic:
    while True:
        audio, _ = mic.read(1536)
        for event in detector.process(audio[:, 0]):  # 16 kHz mono, any chunk size
            print(f"Wake word! (score {event.score:.2f})")
```

[runtime/README.md](runtime/README.md) shows how to measure CPU use and false alarms on a device.

### JavaScript: browser or Node

`runtime/js/src/` runs the same files with onnxruntime-web. Copy it into your project and add
`onnxruntime-web@1.30.0`:

```ts
import { Detector } from "./runtime";

const detector = await Detector.create("model.onnx", "vad.onnx"); // URLs or the files' bytes
const events = await detector.process(samples); // 16 kHz mono Float32Array, any length
```

To port the detector to another language, follow [runtime/SPEC.md](runtime/SPEC.md).

## How it works

A multilingual speech encoder, trained once on the Multilingual Spoken Words Corpus and distilled
from XLS-R, turns 1.5 s of audio into a vector. Training fits a small classifier on your
recordings, with added noise and reverb, against 200,000 other words. It then sets the threshold
on 14 hours of speech and noise, so false alarms stay near the rate you chose. Tested on 240 words
it never saw, each enrolled from 10 recordings, it detects 82% of utterances by other speakers at
about 0.7 false alarms per hour.

To run the app yourself, run the tests or train a base model, see [DEVELOPMENT.md](DEVELOPMENT.md).

## License

The code is under the [MIT license](LICENSE). The base model is trained on the Multilingual Spoken
Words Corpus (MLCommons, CC-BY 4.0), with MUSAN, RIRS_NOISES, Multilingual LibriSpeech and
LibriSpeech, and distilled from XLS-R (Apache-2.0). The voice detector is Silero VAD (MIT). Every
downloaded model includes an `ATTRIBUTION.md` with the details.
