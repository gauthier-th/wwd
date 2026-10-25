# Wake-word runtime spec (format version 1)

Every runtime must implement this exactly: `runtime/detector.py` (onnxruntime) and `runtime/js/src/detector.ts` (onnxruntime-web, wasm). Both are checked against `testdata/golden.json` and `testdata/logic_cases.json`.

## 1. Files

| File | Input | Output |
|---|---|---|
| `model.onnx`, `model.int8.onnx` | `audio`: float32 `[N, window_samples]`, 16 kHz mono in [-1, 1] | `score`: float32 `[N]`, a probability |
| `encoder.onnx` | same as above | `embedding`: float32 `[N, D]`, L2-normalized |
| VAD (`assets/silero_vad_16k_op15.onnx`, Silero v6.2.3) | `input` float32 `[1, 576]`, `state` float32 `[2, 1, 128]`, `sr` int64 scalar `16000` | `output` float32 `[1, 1]`, `stateN` float32 `[2, 1, 128]` |

**Model metadata.** A model carries its detector settings in ONNX `metadata_props`. `model.json` holds the same keys. All values are strings.

| Key | Default | Meaning |
|---|---|---|
| `format_version` | `1` | This spec's version |
| `sample_rate` | `16000` | Must be 16000 |
| `window_samples` | `24000` | Wake-model window (1.5 s) |
| `hop_samples` | `1536` | Tick period, a multiple of 512 (96 ms) |
| `threshold` | per model | Calibrated score threshold |
| `smoothing_k`, `smoothing_n` | `2`, `3` | Trigger when k of the last n ticks reach the threshold |
| `refractory_ms` | `1500` | Minimum time between triggers |
| `vad_threshold` | `0.5` | VAD probability that opens the gate |
| `vad_hangover_ms` | `1500` | How long one VAD frame keeps the gate open |
| `wake_word`, `language`, `base_model_version`, `created_at`, `vad_file` | | Informational |

Runtimes ignore unknown keys. Callers may override any detector setting.

**Derived integers.** Compute these with the same operation order in every language, so they round identically:

- `frames_per_tick = hop_samples / 512`
- `gate_frames = ceil(vad_hangover_ms * sample_rate / 1000 / 512)`
- `refractory_ticks = ceil(refractory_ms * sample_rate / 1000 / hop_samples)`

## 2. VAD

- The stream is cut into consecutive 512-sample frames. Frame `f` covers samples `[512 f, 512 (f + 1))`.
- Each frame is one VAD call. Its input is the **last 64 samples of the previous call's input**, followed by the 512 new samples.
- `state` carries over from call to call.
- Context and state both start as zeros.

## 3. Detector

**Audio history.** The stream is preceded by `window_samples` zeros, so every window is full-length from the first tick on.

**Ticks.** Tick `j` happens once frame `(j + 1) * frames_per_tick - 1` has been processed. Each tick:

1. **VAD maximum.** `vad_max` is the maximum VAD probability over the last `gate_frames` frames, including the current one. Fewer frames are used at stream start, and it's 0 if there are none.
2. **Gate.** The gate is open when `vad_max >= vad_threshold`. With gating disabled, or with no VAD, it's always open.
3. **Score.** If the gate is open, `score` is the model's output on the window `audio[e - window_samples, e)`, where `e = (j + 1) * hop_samples`. Otherwise `score = 0` and the model doesn't run.
4. **Trigger.** Append `score >= threshold` to a history of the last `smoothing_n` ticks.
   - **Fires when** the history holds at least `smoothing_k` true values and `j >= next_allowed`.
   - **On firing**, set `next_allowed = j + refractory_ticks + 1`.
   - **During refractory**, scores still enter the history.
5. **Event.** A trigger emits `{tick: j, sample: e, time_s: e / sample_rate, score}`.

**Comparisons.** All comparisons are made in float64, on the float32 values the models return.

**Chunking.** Callers may feed audio in chunks of any size. Results must not depend on chunking. Offline callers may batch the model calls.

## 4. Conformance

| Quantity | Max abs diff vs golden | Also |
|---|---|---|
| VAD probability, `vad_max` | 1e-5 | |
| fp32 `score`, embedding values | 1e-4 | |
| int8 `score` | 2e-2 | Mean ≤ 1e-3 |
| Gate decisions, event ticks | Identical | |

- **int8 tolerance.** int8 kernels differ between platforms (wasm SIMD, x86 VNNI, ARM dot-product). A rare one-step requantization difference moves one score by about 1e-2. Golden scores sit at least 0.025 from the threshold, so this can't flip an event. On x86 CPUs without VNNI, onnxruntime's default uint8 × int8 kernel can overflow and shift every score by up to about 0.08: set the session option `session.x64quantprecision` to `1`.
- **Golden values** come from independent oracles: the official Silero wrapper, PyTorch, and a plain-loop reference implementation of §3. They never come from the runtimes under test. To regenerate them, run `uv run --extra training python -m scripts.make_golden`.

## 5. Several wake words

A runtime may listen for several wake words on one stream, one file each. Every word keeps its own settings, gate and trigger, and gets exactly the ticks, scores and events §3 gives for its file alone. The words share work, never results.

- **One stream.** All files must have the same `sample_rate`, `window_samples` and `hop_samples`. The audio history, VAD frames and tick numbers are shared.
- **One VAD.** It runs once per frame. Each word takes its own `vad_max` and gate from the same probabilities, with its own `vad_hangover_ms` and `vad_threshold`. A word's score is 0 while its own gate is closed.
- **Events** carry the word's index, in the order the files were given. Within one call they come out by tick, then by word.

**Sharing the encoder (optional).** A wake-word file is an encoder (`audio` → a tensor named `embedding`) followed by a small head (`embedding` → `score`). Files made from the same base model have identical encoders. A runtime may split each file at `embedding`, run one encoder per window for all the files that share it, then each file's head on the embeddings:

- **Encoder part:** the nodes needed to compute `embedding` from the graph's inputs, the initializers they read, and `embedding` as the output.
- **Head part:** the nodes needed to compute the graph's outputs from `embedding`, the initializers they read, and `embedding` as the input.
- **Files that run whole:** no node outputs `embedding`; the head also reads `audio` or another encoder tensor; or a node holds a subgraph.
- **What is shared:** an encoder part, among files whose encoder parts are byte-identical (nodes, initializers, opsets). A float32 file and an int8 file, or files from different base models, run separately.
- **Results:** a split only regroups the same operations, so they stay within §4's tolerances of the files run whole.

`runtime/onnxfile.py` and `runtime/js/src/onnxSplit.ts` split the protobuf bytes directly, without an ONNX library. Each keeps the fields it doesn't change as they are, so equal encoders stay byte-identical.

**Conformance.** `golden.json` → `words` describes two files composed like shipped ones on the same encoder:
- `word_a.onnx`: a one-layer prototype head, the kind the app trains;
- `word_b.onnx`: an MLP head with other smoothing, refractory and gate settings.

Their scores come from each file run whole. Listening for both along with the golden stub, which has no `embedding` and runs whole, each file must match its own golden values within §4's tolerances, with identical gate decisions and events.
