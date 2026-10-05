# Few-shot wake-word trainer: plan

Last updated: 2026-10-05 (M0 done)

## 1. Goal and scope

A user records a few examples of a wake word in the browser. The server fine-tunes a detector from them and returns ONNX files that run continuously on a Raspberry Pi, a phone or a browser.

- **Few-shot from audio.** About 10 recordings of the wake word. The model never sees the wake word's text.
- **Languages.** The 12 high-resourced languages of the Multilingual Spoken Words Corpus (MSWC). See §6.
- **Dual runtime.** Every inference artifact runs identically on **onnxruntime-web (wasm)** and on **onnxruntime for Python**. See §3.
- **Two Python stages.**
  - `training/` builds the multilingual base model offline, on a GPU, before any end user touches the app.
  - `server/` runs the per-user fine-tuning.

**Out of scope for v1:** wake words enrolled from text, training on the device, WebGPU/WebGL backends, languages outside the 12, and native mobile apps. The ONNX files stay compatible with ORT Mobile, so a native app remains possible later.

## 2. Prerequisites (do these before M0)

**Already in place on the dev machine:**

| Item | Version or spec |
|---|---|
| Node | 22 |
| pnpm | 10 |
| Other tools | git-lfs, ffmpeg, Docker |
| NVIDIA driver | 590 (PyTorch wheels bundle CUDA, so no CUDA toolkit install is needed) |
| GPU | RTX 4060 Ti, 16 GB |
| CPU / RAM | 16 cores, 61 GB |
| Free disk | about 480 GB |

**Needs action from you:**

| # | What | Why it's on you |
|---|---|---|
| 1 | ✅ Install **uv**: `curl -LsSf https://astral.sh/uv/install.sh \| sh` | It edits your shell profile. The system Python is 3.9 (pyenv). Leave it alone: uv installs Python 3.12 for this project. |
| 2 | ✅ **Hugging Face account + read token**: after uv, run `uvx --from huggingface_hub hf auth login` | Only you can create the account. MSWC isn't gated and anonymous downloads work, but about 104 GB anonymously risks rate limits. |
| 3 | **Run the large downloads** (about 120 GB, several hours) with the resumable `training.download` script from M1, or let Claude start it in the background | Your bandwidth and your timing |
| 4 | ✅ Not needed on this machine: Chromium and Firefox ran without extra system libraries. CI installs them with `--with-deps`. | |
| 5 | **HTTPS for testing on a phone over the LAN** (M3): mkcert with its CA trusted on the phone, or a tunnel such as cloudflared | Needs your devices. Browsers only allow mic access on HTTPS or localhost. |
| 6 | **Hardware for M4**: a Raspberry Pi 4/5 with a USB mic, and a phone | |
| 7 | *(M5 only)* Decide whether to rent a larger GPU for the teacher model | Cost decision. See §7. |

## 3. Dual-runtime rule (wasm + Python)

**Scope.** These artifacts:
- `vad.onnx`
- `encoder.onnx`, `encoder.int8.onnx`
- `model.onnx`, `model.int8.onnx`

and the detector logic: ring buffer, VAD state, gating, smoothing, refractory period.

**Runtimes.** Pin the same ORT version on both sides and bump them together. Both are at **1.30.0** today.
- **Python:** `onnxruntime` + `numpy` only. `sounddevice` is needed only by the mic command-line tool. No torch, librosa or scipy.
- **Web:** `onnxruntime-web/wasm`, the wasm-only entry, with `numThreads = 1` and SIMD on. Its binary is 14.2 MB (3.7 MB gzipped), half the size of the default WebGPU/WebNN build. It doesn't depend on WebGPU or WebGL, and it doesn't need COOP/COEP headers. Under Node, onnxruntime-web's Node entry loads the same `.wasm` binary, so vitest exercises the kernels browsers run.

**Graph rules.** `core/export.py` (`check_graph`) enforces these on every file we export, and the tests check them again.
- **Opset ≤ 17.** Export uses the TorchScript exporter (`torch.onnx.export(dynamo=False)`) with torch pinned at 2.14.1. The dynamo exporter would need onnxscript and defaults to opset 18.
- **Op allowlist:** `ALLOWED_OPS` in `core/export.py`. That's math and nn ops, plus layout ops that are static except for batch, plus QuantizeLinear/DequantizeLinear.
- **Banned:** STFT/DFT, Loop/Scan/If, contrib ops and custom ops.
- One self-contained file, with no external data.
- float32 inputs and outputs. int8 only in QDQ format.
- Static shapes, except the batch dimension.
- **Freeze before export.** `freeze_for_export` bakes parameter-derived constants (PCEN smoother and exponents, sub-spectral-norm affines) into initializers. Without this step, the graph would contain `If`/`OneHot`/`Tile` and parameter math.
- **The Silero VAD is a pinned third-party file** and is exempt from these rules: it contains `LSTM` and `If`. Instead, it's tested directly in both runtimes against the official wrapper.
- **int8 never touches the frontend.** Quantizing raw audio to 8 bits would erase quiet speech, so `quantize_int8` excludes every `/frontend/` node.

**The feature frontend lives inside the graph,** so no DSP code has to be ported to each platform:
- STFT is a Conv1d with fixed DFT kernels.
- The mel filterbank is a 1×1 Conv1d.
- PCEN uses a truncated causal FIR depthwise Conv1d as its smoother, and Pow for its trainable α/δ/r.

**One spec, two implementations, golden tests.**
- `runtime/SPEC.md` defines the detector.
- `runtime/testdata/` holds WAV files plus the expected per-tick `vad_prob`, per-tick `score`, and trigger `events` for each.
- The Python and JS runtimes must match those values:

  | Model | Per-tick values | Trigger events |
  |---|---|---|
  | fp32 | within 1e-4 | identical |
  | int8 | within 2e-2, mean ≤ 1e-3 | identical (golden scores sit at least 0.025 from the threshold) |

  **Why int8 needs a wider tolerance.** int8 kernels differ between platforms: wasm SIMD, x86 VNNI, ARM dot-product. A rare one-step requantization difference moves one score by about 1e-2. M0 measured, wasm vs native: fp32 max 7.6e-7; int8 max 1.1e-2, mean 9e-5.

**Test matrix:**
- pytest with Python ORT on x86_64 (aarch64 on the Pi in M4).
- vitest running onnxruntime-web in Node, which exercises the wasm build.
- Playwright on Chromium and Firefox, against the production build. It runs the golden parity check inside a Web Worker, plus a live mic → AudioWorklet → Worker check. Chromium plays `speech.wav` as its fake mic; Firefox uses a fake tone.

**Resampling** happens only at the microphone and is outside the parity contract. Golden tests feed 16 kHz WAV files directly.
- **Browser:** the AudioWorklet resamples to 16 kHz.
- **Python:** request 16 kHz from PortAudio, with a small numpy polyphase fallback.

The server uses the Python `runtime` for VAD trimming and post-export checks, so it runs the same code that ships.

## 4. Contracts

### 4.1 Wake-word model (`model.onnx`, `model.int8.onnx`)

| | |
|---|---|
| Input | `audio`: float32 `[N, 24000]`, a 1.5 s window of 16 kHz mono audio in [-1, 1] |
| Output | `score`: float32 `[N]`, a probability |
| Size | Default encoder (width 32): 204k parameters, 1.6 MB fp32 (including 0.7 MB of fixed DFT/mel kernels), 1.2 MB int8. Width 24 is 125k parameters. |
| Compute per window | Backbone about 47.5M MACs. The conv-STFT frontend adds about 26M, because there's no FFT op. That's about 7 ms in single-threaded browser wasm, or about 7% of one core in real time with the gate open. |
| Metadata | `metadata_props` and `model.json` carry the same keys:<br>`format_version=1`, `sample_rate=16000`, `window_samples=24000`, `hop_samples=1536`<br>`threshold`, `smoothing_k=2`, `smoothing_n=3`, `refractory_ms=1500`<br>`wake_word`, `language`, `base_model_version`, `created_at`<br>`vad_file`, `vad_threshold=0.5`, `vad_hangover_ms=1500` |
| Download bundle | `model.onnx`, `model.int8.onnx`, `model.json`, `vad.onnx`, Silero's `LICENSE`, `ATTRIBUTION.md` (MSWC, CC-BY) |

`encoder.onnx` follows the same input contract and outputs `embedding: float32 [N, D]`.

The model is stateless on purpose. A model that streams with cached convolution state would be cheaper, but it would complicate every client.

### 4.2 VAD (`vad.onnx`)

- **File:** `silero_vad_16k_op15.onnx` from silero-vad **v6.2.3** (released 2026-09-23), 16 kHz only, opset 15.
  - **Inputs:** `input` `[1, 576]`, `state` `[2, 1, 128]`, `sr` (int64 scalar, 16000).
  - **Outputs:** `output`, `stateN`.
  - **Contains `If` and `LSTM` nodes** (see §3 for the exemption). My earlier note that it had no `If` and no `sr` was wrong.
- **Pinning:** committed at `runtime/assets/` with Silero's MIT `LICENSE`. sha256 `7ed98ddb…1b2c49`, identical to the GitHub tag and to the pip package.
- **Each call:** 512 new samples (32 ms), with the **last 64 samples of the previous call prepended**. State `[2, 1, 128]` carries over from call to call. With the context, both runtimes match the official wrapper exactly. Without it, outputs are off by up to 0.21, with no error.

### 4.3 Detector logic (`runtime/SPEC.md`)

1. **Ticks.** One tick is 3 VAD frames, 1536 samples (96 ms). The VAD runs on every frame. The wake model runs once per tick on the last 24000 samples.
2. **Gate.** The wake model runs only if the maximum VAD probability over the last 1.5 s is at least `vad_threshold`. When the gate is closed, `score` is 0. The gate can be switched off.
3. **Trigger.** At least `smoothing_k` of the last `smoothing_n` ticks reach `threshold`. After a trigger, no new trigger fires for `refractory_ms`.
4. **Calibration.** The server calibrates false accepts per hour with the gate **off**. That's the conservative number, since the gate can only remove false accepts. Its effect on recall is measured on the test page and on devices.

## 5. Architecture and repo layout

```
training/  (offline, GPU, once per base-model version)
  MSWC(12 langs) + MUSAN + RIRs + MLS/LibriSpeech
      → student (M1) / teacher → distilled student (M5)
      → models/base/<ver>/  encoder.pt, encoder(.int8).onnx, neg_bank.npz,
                            longform_cache.npz, longform_verify/, base_model.json
                                                          │
web/ (React) ──WAV uploads──► server/ API ──job──► server worker ◄┘
     ▲                                     (augment, train head, calibrate,
     │                                      compose ONNX, verify with runtime/)
     └── live test: runtime/js in a Worker (onnxruntime-web wasm)
                       ◄── model.onnx · model.int8.onnx · model.json · vad.onnx
Pi / any Python: runtime/ (onnxruntime + numpy)
```

```
wwd/
├── PLAN.md
├── pyproject.toml        # one uv project (no build step), run from repo root;
│                         # dependency sets: server (CPU torch), training (CUDA torch), dev
├── core/                 # shared: frontend, encoder, head, finetune(), export + graph checker, audio utils
├── training/             # python -m training.<step>
├── server/               # uvicorn server.api:app  ·  python -m server.worker
├── runtime/              # Python runtime, flat modules: spec.py, vad.py, detector.py, wav.py, listen.py
│   ├── requirements.txt  # onnxruntime, numpy, sounddevice (Pi install)
│   ├── SPEC.md           # detector spec (§4.3)
│   ├── assets/           # pinned Silero VAD + its LICENSE
│   ├── testdata/         # golden WAVs, stub models, golden.json, logic_cases.json
│   └── js/               # TS runtime @wwd/runtime: same detector + VAD wrapper
├── web/                  # React + Vite + TypeScript; src/audio = worklet, worker, pipeline
├── scripts/              # make_golden.py (regenerates runtime/testdata)
├── .github/workflows/    # ci.yml: Python, JS (wasm in Node), browsers
├── pnpm-workspace.yaml   # web, runtime/js
├── data/                 # gitignored: downloads, extracted clips, memmaps
└── models/               # gitignored: base/<ver>/
```

- **Module folders.** There are no nested package folders. `core/`, `training/`, `server/` and `runtime/` are the import packages themselves (`from core.export import …`).
- **Dependency sets.** These are extras: `server` (torch 2.14.1+cpu) and `training` (torch 2.14.1+cu130, plus `silero-vad` for golden oracles). They're declared as conflicting, so uv resolves each against its own PyTorch index. `uv run --extra training …` locally; CI uses `--extra server`.
- **import-linter boundaries:**
  - `server` never imports `training`.
  - `core` imports neither of them.
  - `runtime` imports nothing from the repo and never imports torch.
- **Installing on a Pi:** copy `runtime/`, run `pip install -r runtime/requirements.txt`, then `python -m runtime.listen model.onnx`.

## 6. Languages and data

### 6.1 Languages

These are MSWC's high-resourced languages (over 100 h). Hours come from the MSWC dataset card. Opus sizes were measured on Hugging Face across all splits.

| Code | Language | Hours | Opus GB |
|---|---|---|---|
| en | English | 1957 | 31.4 |
| de | German | 1083 | 17.3 |
| fr | French | 754 | 12.0 |
| ca | Catalan | 615 | 10.8 |
| rw | Kinyarwanda | 422 | 7.7 |
| es | Spanish | 349 | 6.6 |
| fa | Persian | 327 | 6.1 |
| it | Italian | 155 | 2.9 |
| ru | Russian | 137 | 2.7 |
| pl | Polish | 130 | 2.3 |
| eu | Basque | 118 | 2.2 |
| cy | Welsh | 108 | 1.6 |
| | **Total** | **6155** | **103.6** |

The API and the language picker accept only these 12 codes.

### 6.2 Sources

| Source | Used for | License | Download size |
|---|---|---|---|
| MSWC, opus, 12 languages (`MLCommons/ml_spoken_words`) | Base training, negative bank, long-form "word salad" | CC-BY 4.0 | about 104 GB |
| MUSAN (OpenSLR 17) | Noise and music augmentation, long-form noise | Mixed CC; check each subset | 11.1 GB |
| RIRS_NOISES (OpenSLR 28) | Reverb augmentation | Apache-2.0 | 1.3 GB |
| MLS dev+test: de, fr, es, it, pl | Natural long-form speech for false-accept measurement | CC-BY 4.0 | 5.0 GB |
| LibriSpeech test-clean + test-other (OpenSLR 12) | Natural long-form English | CC-BY 4.0 | about 0.7 GB |
| XLS-R 300M (`facebook/wav2vec2-xls-r-300m`) | Teacher model (M5) | Apache-2.0 | about 1.2 GB |
| Silero VAD v6.2.3, `silero_vad_16k_op15.onnx` | VAD | MIT | 1.3 MB |

**Common Voice isn't used.** Since October 2025 it's only available from Mozilla Data Collective, behind an account and its terms of service. MSWC, which was built from Common Voice, already covers the 12 languages.

### 6.3 How MSWC is fetched

- **Layout on Hugging Face:**
  - audio: `data/opus/{lang}/{split}/audio/{n}.tar.gz`
  - splits: `data/splits/{lang}/splits.tar.gz`, a CSV with link, word, valid, speaker and gender.
- **Download method:** the repo's `load_dataset` path is a loading script, which current `datasets` releases no longer run. Download the files directly with `huggingface_hub` and `allow_patterns`.
- **Shards are sorted alphabetically by word.** French shard 0 starts at "abad, abandonna, abandonne…". You can't take a subset of shards. Every shard of a language has to be streamed, keeping only the clips you selected.
- **Process, one language at a time:**
  1. Download the shards.
  2. Extract the selected clips as opus.
  3. Delete the archives.

  Peak extra disk is about one language's worth of archives, at most 31 GB (English).
- **Decoding:** clips stay opus (48 kHz) on disk. DataLoader workers decode them and resample to 16 kHz with soxr. Decoding everything to int16 would take about 700 GB.

### 6.4 Disk budget

The total stays under about 150 GB of the 480 GB free:
- extracted MSWC clips: ≤ about 100 GB
- MUSAN: 11 GB
- negative bank memmap: 6.4 GB
- everything else: a few GB

### 6.5 Licensing

- **MSWC (CC-BY 4.0):** credit it in the model card, the download bundle (`ATTRIBUTION.md`) and the app's about page.
- **Teacher:** XLS-R (Apache-2.0) is the default. It keeps commercial use open.

## 7. `training/`: base model pipeline

Every step runs as `python -m training.<step>` with a YAML config.

1. **`download`.** Fetches everything in §6.2. It's resumable and checks checksums.
2. **`prepare`.**
   - Builds a parquet manifest with lang, word, speaker, gender, split and path.
   - **Selects keyword classes.** Starting values: at least 20 clips and at least 5 speakers per class, capped at 500 clips per class.
   - **Eval sets.** The eval split holds out 10% of eligible words in each language and also uses MSWC's speaker-disjoint dev/test splits.
   - **Phrase augmentation.** Builds 2–3-word "phrase classes" by joining words from the same speaker with 50–150 ms gaps, so that phrases like "hey jarvis" fit the 1.5 s window.
3. **`train_student` (M1).**
   - **Model:** BC-ResNet with a PCEN frontend and attention pooling, about 128–300k parameters.
   - **Loss:** Sub-center ArcFace.
   - **Augmentation:** noise, RIR, speed 0.9–1.1, gain, SpecAugment, random EQ.
   - **Purpose:** no teacher. This yields a usable base model quickly and unblocks the app.
4. **`train_teacher` + `distill` (M5).**
   - **Teacher:** XLS-R 300M with a weighted sum of layers, attentive pooling and Sub-center ArcFace.
   - **Distillation:** the student learns from the teacher's embeddings with a cosine loss plus ArcFace.
   - **On the RTX 4060 Ti:** keep the XLS-R encoder **frozen**, so it only runs forward passes in bf16, and train only the head. Fully fine-tuning XLS-R would take much longer, which is the reason to consider renting a larger GPU.
5. **`eval_fewshot`.** K = 1/5/10 accuracy at 1% false-alarm rate on held-out words, reported per language. EdgeSpot's 82.0% at 10-shot was measured on English MSWC only, so treat it as a sanity reference, not a target.
6. **`eval_e2e`.** This is the number that matters.
   - For held-out words, enroll 10 clips from one speaker.
   - Run the same `core.finetune()` the server uses.
   - Measure recall on other speakers.
   - Measure false accepts per hour by running the **`runtime` Python detector** over the long-form set. That's the shipped code path.
7. **`build_bank`.** Builds the negative bank (§8) and the long-form sets:
   - **`longform_cache.npz`:** encoder embeddings at a 96 ms hop over about 1 h per language plus about 2 h of music and noise, used for calibration. The speech is MSWC test-split words joined into speech-like "word salad", plus MLS or LibriSpeech where available. Some segments mix speech with MUSAN noise.
   - **`longform_verify/`:** 1 h of held-out raw audio for checking the files that actually ship.
8. **`export`.** Writes `models/base/<ver>/`:
   - `encoder.pt`
   - `encoder.onnx`
   - `encoder.int8.onnx` (static QDQ quantization, calibrated on bank audio)
   - the bank files
   - `base_model.json` (version + hash)

   Each ONNX file must pass the graph checker and the dual-runtime golden tests.

## 8. Negative bank (200k clips)

The bank is a fixed sample of clips taken from the MSWC download that `training/` already makes.

1. **Selection, using the split CSVs only:**
   - MSWC **train** split, valid clips only, no speakers who appear in an eval set.
   - **Equal quota per language,** about 16.7k each across the 12. When a small language runs out under the caps, its leftover quota goes to the others.
   - **At most 3 clips per word,** plus a cap per speaker. This maximizes the number of distinct words, which is what hard-negative mining needs.
   - Save the result as `bank_manifest.parquet`. It stays fixed across base-model versions.
2. **Extraction:** decode the selected clips to 16 kHz int16 in a single memmap of about 6.4 GB. It stays on the training machine.
3. **Embedding, once per base-model version:** `neg_bank.npz` holds fp16 embeddings (about 100 MB at D = 256) plus word, language and speaker. **Only this file goes to the server.**
4. **Wake-word exclusion at fine-tune time.** If the user's wake word exists in MSWC, its clips would fight the positives. Drop bank clips that:
   - are the same word as the full wake-word label (normalized). This is the only use of the text label.
   - are very close to the user's prototype embedding. This catches spelling variants and the same word in other languages.

   Words that are only part of the phrase, like "hey" in "hey jarvis", stay in as useful negatives.

## 9. `server/`: API and fine-tune worker

**Stack:** FastAPI, Pydantic v2, SQLModel on SQLite (Postgres later), soxr, soundfile, torch (CPU), onnx, and `runtime`.
- **Jobs:** a DB table, taken one at a time by a single worker process. Move to Celery or arq only if scale requires it.
- **Storage:** a small interface over the local filesystem now, S3 later.

### 9.1 API

```
GET    /api/languages                          the 12 codes + reading paragraphs
POST   /api/wakewords                          {name, language}
POST   /api/wakewords/{id}/samples             multipart WAV, kind=positive|other_speech|ambience
DELETE /api/wakewords/{id}/samples/{sid}
POST   /api/wakewords/{id}/trainings           {mode: quick|quality, target_fa_per_hour}
GET    /api/trainings/{tid}                    status + metrics
GET    /api/trainings/{tid}/events             SSE progress
GET    /api/trainings/{tid}/artifacts/{file}
GET    /api/base-models
DELETE /api/wakewords/{id}                     deletes recordings + models (GDPR)
```

### 9.2 Upload checks

Each upload goes through these steps:
1. Keep the original file.
2. Resample to 16 kHz.
3. Trim and segment with the `runtime` VAD, the same file the browser uses.
4. Normalize loudness.
5. Reject clips that are clipped, silent, or longer than about 1.3 s of speech.
6. Compute each clip's similarity to the other samples, and flag outliers such as a mispronunciation or the wrong word.

### 9.3 Quick mode (default)

Target: under 2 minutes on CPU. No GPU on the server.

1. **Positives:** each recording × about 300 augmentations: shift within the window, speed 0.9–1.1, gain, RIR, MUSAN noise at 0–20 dB SNR, random EQ.
2. **Negatives:**
   - bank embeddings, weighted toward the user's language, with the wake word excluded (§8);
   - **hard negatives:** the MSWC words closest to the user's prototype;
   - **partial wake words:** the first or last 40–70% of each recording;
   - the user's own "other speech" and ambience windows.
3. **Head:** a small MLP on the L2-normalized embedding, trained with class-weighted BCE. The encoder stays frozen.
4. **Leave-2-out folds:** estimate recall and flag weak recordings. The final head trains on all recordings.
5. **Calibration:** run the head plus the §4.3 smoothing over `longform_cache.npz` with the gate off, and pick the threshold that meets `target_fa_per_hour` (default 0.5/h).
6. **Compose:**
   - Export the head from torch as a tiny ONNX graph.
   - Merge it with `encoder.onnx` and with `encoder.int8.onnx` using `onnx.compose`. Both encoders were quantized and validated once, in `training/`.
   - Run the graph checker.
   - Write the metadata.
7. **Verify:** run the *shipped* files through the `runtime` Python detector on `longform_verify/`, with the gate on and off. Check parity with the PyTorch scores. Report recall, false accepts per hour, and fp32 vs int8.

### 9.4 Quality mode (M6)

- Unfreeze the last encoder block, and use raw-audio negatives from a subset of the bank's audio.
- Needs a GPU and takes about 10 minutes.
- The full graph is exported from torch, then quantized with a calibration subset that ships with the base model.

## 10. `web/`: React app

**Stack:** Vite, React, TypeScript, TanStack Query, react-router, and `runtime/js` (onnxruntime-web, wasm).

**Audio pipeline:**
- An AudioWorklet captures the mic and resamples to 16 kHz.
  - Firefox won't connect a mic stream to an AudioContext running at a different sample rate, so the worklet must resample.
- The worklet sends 512-sample frames to a Web Worker. The Worker runs `runtime/js`, with the VAD on every frame and the wake model on every tick.
- The main thread only draws the UI.

**Settings:**
- `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` all `false`.
- Uploads are WAV files encoded in the browser. Don't use MediaRecorder: Opus artifacts would hurt the training data.
- Vite serves the ORT `.wasm` files (`ort.env.wasm.wasmPaths`).

**Flow:**
1. **Setup:** the wake-word label and the language (one of the 12).
2. **Record:**
   - Say the wake word 10 times in one continuous take; the VAD cuts it into clips. There's a live "speech detected" light, and takes with no speech are rejected.
   - Read a paragraph aloud (about 30 s, in the chosen language).
   - Record 15 s of room ambience.
3. **Review:** waveform, playback, quality and outlier badges, re-record.
4. **Train:** progress over SSE, then recall and false accepts per hour.
5. **Test:**
   - a live graph of the wake score and the VAD probability, with the threshold line and trigger events;
   - a gate on/off toggle;
   - a sensitivity slider that rewrites only the metadata.
6. **Download:** the bundle from §4.1.

**Two practical notes:**
- **Same mic as the device.** Open the app on the phone to record with the mic you'll deploy on. For the Pi, `python -m runtime.listen --record` uploads takes made on the Pi's own mic.
- **Reading paragraphs.** One per language, 12 in total. Have native speakers check the rw, fa, eu and cy texts. Persian is right-to-left.

## 11. `runtime/`: detectors

- **Python (`runtime/`):**
  - `vad.py` is the Silero wrapper (state + 64-sample context).
  - `detector.py` implements §4.3.
  - `listen.py` handles mic capture, prints trigger events or calls a hook, and records with `--record`.
  - It also includes a benchmark that reports CPU % and latency.
- **JS (`runtime/js`):** the same API and the same logic. The web app uses it, and it's published as a pnpm package for other web apps.
- **Phones:** run in the browser (wasm). Native apps can load the same `.onnx` files with ORT Mobile later; they're outside the test matrix for now.

## 12. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M0** ✅ | uv project + pnpm workspace; `core` frontend + stub encoder export; graph checker; pinned Silero VAD; `runtime/SPEC.md`; Python + JS detectors; golden vectors; CI (pytest, vitest with onnxruntime-web wasm in Node, Playwright) | **Done 2026-10-05.** pytest 33/33, vitest 21/21, Playwright 4/4 (Chromium + Firefox). Identical events in every runtime; VAD matches the official wrapper. |
| **M1** | `training`: download, prepare, student-only training, few-shot and end-to-end evals, bank, base **v0.1** | Per-language eval numbers recorded. Encoder files pass the dual-runtime tests. |
| **M2** | `server`: uploads + checks, jobs, quick mode, calibration, compose, verify, SSE | A curl-driven run produces a bundle that passes the dual-runtime tests |
| **M3** | `web`: record → review → train → live test → download | The full flow works in Chrome, Firefox and mobile Safari |
| **M4** | Devices: Pi 4/5 (Python ORT on aarch64), phone browser (wasm) | CPU %, latency and a 24 h real-room false-accept test (TV + speech) measured on each |
| **M5** | Base **v1**: XLS-R teacher + distillation | End-to-end metrics beat v0.1. Older user models keep their base version. |
| **M6** | Quality mode, multi-speaker enrollment, auth, S3, retention policy | — |

M2 and M3 can start right after M0, against the stub encoder, while M1 trains.

## 13. Decisions

**Made:**
- **Window:** 1.5 s (fits 2–3-word phrases), hop 1536 samples.
- **Languages:** the 12 MSWC high-resourced languages only. No Common Voice.
- **Runtimes:** onnxruntime-web (wasm) + Python onnxruntime, held to golden-vector parity.
- **int8 contract:** per-tick drift ≤ 2e-2 (mean ≤ 1e-3) across runtimes, with identical events. Before shipping int8 for a real model, M1 must also show that int8 agrees with fp32 on the evals. The M0 stub isn't representative of that.
- **Layout:** flat Python packages (`core/`, `training/`, `server/`, `runtime/`), one uv project.
- **Teacher:** XLS-R 300M (Apache-2.0), frozen on the local GPU.
- **Default export:** one merged file per wake word.

**Still open:**
- Also export a shared `encoder.onnx` plus a small `head.onnx` per word, for devices that run several wake words?
- Should the server get a GPU for quality mode?
- Build a native phone app later?

## 14. Risks

- **False accepts with about 10 samples.** These may miss the target rate. The calibration report shows it honestly, and a second-stage verifier can be added to the runtime later.
- **Always-on listening in a phone browser** only works while the page is in the foreground; iOS Safari stops the mic when the screen locks. True background listening needs a native app, and the ONNX contract keeps that possible.
- **Mismatch between the recording mic and the device mic.** Mitigated by EQ augmentation and by recording on the target device.
- **Languages with little data** (cy, eu, pl, ru: about 110–140 h each) may enroll less reliably. `eval_fewshot` and `eval_e2e` report results per language, so this will be visible.
- **Privacy.** Voice recordings are personal data under GDPR. Plan a retention policy and a delete endpoint (§9.1).

## Sources

- [EdgeSpot (arXiv 2601.16316)](https://arxiv.org/abs/2601.16316)
- [Gok et al., SSL few-shot KWS (arXiv 2506.17686)](https://arxiv.org/abs/2506.17686)
- [MLCommons/ml_spoken_words on Hugging Face](https://huggingface.co/datasets/MLCommons/ml_spoken_words)
- [silero-vad (`utils_vad.py`, `data/`)](https://github.com/snakers4/silero-vad)
- [facebook/multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech)
- [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [MUSAN (OpenSLR 17)](https://www.openslr.org/17/) · [RIRS_NOISES (OpenSLR 28)](https://www.openslr.org/28/) · [LibriSpeech (OpenSLR 12)](https://www.openslr.org/12/)
- [Common Voice moves to Mozilla Data Collective](https://github.com/common-voice/cv-dataset/)
