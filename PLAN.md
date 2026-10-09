# Few-shot wake-word trainer: plan

Last updated: 2026-10-09. The whole app now runs in the browser and is published as static files on GitHub Pages (§12.6). Usage docs: README.md and DEVELOPMENT.md.

## 1. Goal and scope

A user records a few examples of a wake word in the browser. The server fine-tunes a detector from them and returns ONNX files that run continuously on a Raspberry Pi, a phone or a browser.

- **Few-shot from audio.** About 10 recordings of the wake word. The model never sees the wake word's text.
- **Languages.** The 12 high-resourced languages of the Multilingual Spoken Words Corpus (MSWC). See §6.
- **Dual runtime.** Every inference artifact runs identically on **onnxruntime-web (wasm)** and on **onnxruntime for Python**. See §3.
- **Two Python stages.**
  - `training/` builds the multilingual base model offline, on a GPU, before any end user touches the app.
  - `server/` runs the per-user fine-tuning.

**Out of scope for v1:** wake words enrolled from text, training on the device, WebGPU/WebGL backends, languages outside the 12, and native mobile apps. The ONNX files stay compatible with ORT Mobile, so a native app remains possible later. Also out of scope (the former M6, dropped 2026-10-09): a GPU quality mode, multi-speaker enrollment, accounts and authentication, cloud storage, and a retention policy.

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
| 3 | ✅ **Large downloads** (about 135 GB): done by `training.download`, resumed once after an interruption | Your bandwidth and your timing |
| 4 | ✅ Not needed on this machine: Chromium and Firefox ran without extra system libraries. CI installs them with `--with-deps`. | |
| 5 | **HTTPS for testing on a phone over the LAN** (M3's real-device check, M4): mkcert with its CA trusted on the phone, or a tunnel such as `cloudflared tunnel --url http://localhost:8000`. The server serves the app and the API on one origin, so one tunnel is enough. | Needs your devices. Browsers only allow mic access on HTTPS or localhost. |
| 6 | ✅ **Hardware for M4**: a Raspberry Pi 4B and an Android phone (Chrome). For the phone, `adb reverse tcp:8000 tcp:8000` avoids HTTPS. | |
| 7a | *(optional)* `sudo apt install qemu-user-static`, so `scripts/test_aarch64.sh` can run the runtime on emulated aarch64 before a Pi | Installs a system package |
| 7 | ✅ *(M5)* No larger GPU needed: the frozen XLS-R teacher trained on the RTX 4060 Ti in about 4 h (§7, §12.5) | |

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

- **Server bundles** (a real model on minutes of audio) are checked the same way by `scripts/check_bundle.py` (Python) and `runtime/js/test/bundle.test.ts` (wasm). Two bounds are looser there, both measured in the M2 smoke run:
  - **VAD:** within 1e-4. Its LSTM state accumulates float differences over minutes: 1.2e-5 after 2 min.
  - **int8:** mean ≤ 1e-3, at most 1% of ticks off by more than 2e-2, none by more than 0.1. On the real encoder, 2 ticks of 1572 moved by up to 0.035, while 99% agreed within 1e-6.
  - **Events** must be identical, unless a tick's scores straddle the threshold across runtimes; the test lists such ticks. That's the only way score drift can change an event, since the golden tests prove the logic identical. None straddled in M2.

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
                            longform_cache(.int8).npz, longform_verify.npz, base_model.json
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
├── core/                 # shared: frontend, encoder, head, finetune(), calibrate, export + compose + graph checker, verify
├── training/             # python -m training.<step>
├── server/               # uvicorn server.api:app  ·  python -m server.worker
├── runtime/              # Python runtime, flat modules: spec.py, vad.py, detector.py, wav.py, listen.py
│   ├── requirements.txt  # onnxruntime, numpy, sounddevice (Pi install)
│   ├── SPEC.md           # detector spec (§4.3)
│   ├── assets/           # pinned Silero VAD + its LICENSE
│   ├── testdata/         # golden WAVs, stub models, golden.json, logic_cases.json
│   └── js/               # TS runtime @wwd/runtime: same detector + VAD wrapper
├── web/                  # React + Vite + TypeScript; src/audio = worklet, worker, pipeline
├── scripts/              # make_golden.py (runtime/testdata), m1_pipeline.sh, m2_smoke.sh + check_bundle.py
├── .github/workflows/    # ci.yml: Python, JS (wasm in Node), browsers, site → GitHub Pages
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
- **Shards are sorted alphabetically by word.** French shard 0 starts at "abad, abandonna, abandonne…". You can't take a subset of shards; every shard of a language has to be streamed.
- **Real size (M1):** 365 shards, **116 GB** of opus for the 12 languages. The 103.6 GB in §6.1 left out the `dev` split. The connection limits the download to about 11 MB/s, so it takes about 3 hours. OpenSLR files come from the EU mirror (`openslr.elda.org`, about twice as fast from France), with `www.openslr.org` as fallback.
- **As built (`training.download`):** each shard is downloaded, sha256-verified against its Hugging Face LFS hash, and streamed into a **packed store**: `data/mswc/{lang}/{split}/{n}.bin` holds the concatenated `.opus` files, and `{n}.parquet` holds word, clip, offset and length. The archive is then deleted. The parquet is written last, so it doubles as the done marker, and a re-run resumes. That's 134 GB and 365 + 365 files, instead of about 20M loose files.
- **Decoding:** clips stay opus on disk. `training/opus.py` calls the system libopus through ctypes and decodes **directly at 16 kHz**: about 1,500 clips/s per core, against 305 for soundfile + soxr at 48 kHz. Lengths match exactly; the median clip agrees with the soxr path to 55 dB. If libopus is missing, it falls back to soundfile. Decoding everything to int16 would take about 700 GB.
- **MSWC's own splits share speakers.** For example, 1,160 of the 1,175 Welsh dev speakers are also in train. They can't measure generalization to new speakers, so `prepare` defines its own splits (§7).

### 6.4 Disk budget

Measured in M1, about 165 GB in total:
- MSWC packed store: 134 GB
- MUSAN: 12 GB
- RIRs: 3.6 GB
- MLS dev+test: 5 GB
- negative bank: 6.4 GB
- long-form streams: 1.8 GB

### 6.5 Licensing

- **MSWC (CC-BY 4.0):** credit it in the model card, the download bundle (`ATTRIBUTION.md`) and the app's about page.
- **Teacher:** XLS-R (Apache-2.0) is the default. It keeps commercial use open.

## 7. `training/`: base model pipeline

Every step runs as `python -m training.<step>`, configured with CLI flags. `scripts/m1_pipeline.sh` chains all of them. It's meant to run detached (`setsid nohup`) so it survives the editor session ending, every step resumes, and `START_AT=<step>` skips steps already done.

1. **`download`.** Fetches everything in §6.2. It's resumable and checks checksums.
2. **`prepare`.** Our own splits, by stable hash, because MSWC's share speakers (§6.3):
   - **Eval speakers:** 10% of each language's speakers. None of their clips are trained on.
   - **Held-out words:** 10% of eligible words. They're never trained on, and few-shot and end-to-end evaluation use them.
   - **Classes:** eligible words (at least 20 clips and at least 5 speakers, counting non-eval speakers only), capped at 300 clips per word. The cap takes speakers round-robin, to keep speaker diversity.
   - **Result:** **85,258 classes and 7.33M training clips.** English has 15.2k words and 1.78M clips; Welsh has 2.4k words and 192k clips.
   - **Deferred to v0.2:** phrase classes (2–3 words joined, so phrases like "hey jarvis" fit the 1.5 s window). v0.1 trains on single words.
3. **`train_student` (M1).**
   - **Model:** BC-ResNet width 32 with a PCEN frontend, temporal self-attention and attentive pooling: 204k parameters, 256-dim embedding.
   - **Loss:** Sub-center ArcFace (k = 3, s = 30, m = 0.3). That's a 65.5M-parameter head for 85k classes. `--margin-warmup` ramps the margin; v0.1 runs without it.
   - **Augmentation:** speed 0.9–1.1 and random placement happen in the loader. Level, RIR, MUSAN noise/music/babble at 0–25 dB SNR, random EQ and the noise floor run on the GPU (`core/augment.py`, which the server reuses on CPU). SpecAugment runs in the encoder and only in training.
   - **Sampling:** language share is proportional to clips^0.5 (`--lang-temperature`). English gets 15% and Basque 4.7%, instead of 24% and 2%.
   - **Run:** batch 256, 80k steps (about 2.8 epochs), AdamW (fused), learning rate 2e-3 with cosine decay, bf16 autocast. The frontend always runs in fp32, because PCEN divides energies as small as 1e-9.
   - **Throughput:** about 1,400 clips/s, about 4 h. The 4060 Ti is memory-bandwidth-bound (288 GB/s): BatchNorm and depthwise convs dominate the backbone, and the 85k-class head adds about 57 ms per step. Partial-FC is the next lever if retraining speed matters.
   - **Early convergence is slow with 85k classes.** 10-shot recall@1%FAR was 8% at step 4k and 24% at step 8k.
4. **`train_teacher`, `cache_teacher`, `train_student --teacher` (M5).** `scripts/m5_pipeline.sh` chains them with the export and the evaluations.
   - **Teacher (`training/teacher.py`):** XLS-R 300M (pinned revision, `transformers`, training extra only), **frozen**, bf16, `torch.compile`d. It runs only its first **18 of 24 layers**: a probe of each layer alone (`train_teacher --probe`) peaks at layers 12–17 and falls off above 18.
     - **Head (5.1M parameters):** a softmax-weighted mix of the 19 layer-normalized states, a 1024 → 512 projection, 2 pre-norm transformer layers, attentive pooling, and a 256-dim embedding.
     - **Training:** Sub-center ArcFace on the student's clips, sampling and augmentation. 28k steps of 256 (about one epoch) at 469 clips/s, about 4.2 h.
     - **Class centers start from data:** each one is the mean teacher embedding of 2 clips of its word. With random centers the first attempt collapsed: few-shot recall fell from 31% (untrained head) to 5.5% after 4k steps, with training accuracy near 0. With data-initialized centers it reached 97.4% (10-shot) after 1,500 steps.
   - **Cache (`cache_teacher`):** the teacher's embedding of every training clip, clean and centered: 7.3M × 256 float16 (3.75 GB), about 3.9 h at 520 clips/s, resumable.
   - **Distillation (`train_student --teacher`):** the same student and recipe as v0.1. Its loss is ArcFace plus 10 × (1 − cosine) to the teacher's clean embedding of the clip the student sees augmented. The student's ArcFace class centers start from the teacher's.
   - **Why frozen:** on the RTX 4060 Ti, fully fine-tuning XLS-R would take far longer; that was the reason to consider renting a larger GPU.
5. **`evaluate`** (few-shot). For held-out words, average K = 1/5/10 enrollment clips into a prototype. Score other clips of the word against clips of other held-out words in the same language, and report recall at 1% false alarms plus the EER, per language. EdgeSpot's 82.0% at 10-shot was measured on English MSWC only, so it's a sanity reference, not a target.
6. **`eval_e2e`.** This is the number that matters.
   - **Enrollment:** for each held-out word, 10 clips from as few speakers as possible. MSWC rarely has 10 clips of one word from one speaker.
   - **Fine-tune:** the same `core.finetune()` the server uses.
   - **Calibration:** the threshold is set on the long-form cache for 0.5, 1 and 2 false accepts/h.
   - **Recall** on other speakers' clips, streamed over 6 consecutive ticks with the 2-of-3 rule.
   - **False accepts per hour** on the held-out verification streams, out of sample. These are counted with `core.calibrate.count_triggers`, which is tested equal to the runtime `Trigger`. They're computed on torch embeddings, though; checking the shipped ONNX files through the runtime is the server's verification step (M2).
7. **`build_bank`.** Model-independent data: the negative bank (§8) and the long-form streams. Measured: **15.8 h in 185 five-minute streams.**
   - **Word salad, about 1 h per language:** eval-speaker clips of non-held-out words, as utterances of 3–10 words with pauses.
   - **Real read speech, 20 min each:** MLS for de/fr/es/it/pl, LibriSpeech for en.
   - **Noise and music:** 2 h of MUSAN streams. MUSAN noise is also mixed into 30% of the speech streams.
   - **Word timeline:** where every MSWC word was placed, so a user's wake word spoken in the long-form audio isn't counted as a false accept.
   - **Split:** 10% of streams (by hash) are held out for verification.
8. **`export`.** Writes `models/base/<ver>/`:
   - `encoder.pt`
   - `encoder.onnx` and `encoder.int8.onnx` (static QDQ, calibrated on bank clips)
   - `neg_bank.npz`: augmented embeddings of the 200k bank clips
   - `longform_cache.npz`: per-tick calibration embeddings, plus the word timeline
   - `longform_cache.int8.npz` (since M2): the same ticks through `encoder.int8.onnx`, on onnxruntime, so int8 files get their own threshold (§9.3)
   - `longform_verify.npz`: raw verification audio
   - `augment_assets.npz`: 30 min of noise, 10 min of babble and 500 RIRs, for the CPU server
   - `base_model.json`: version, sources, file hashes

   Each ONNX file must pass the graph checker.

## 8. Negative bank (200k clips)

The bank is a fixed sample of clips taken from the MSWC download that `training/` already makes.

1. **Selection:**
   - Valid clips from non-eval speakers, any MSWC split.
   - **Equal quota per language,** about 16.7k each across the 12. When a small language runs out under the caps, its leftover quota goes to the others.
   - **At most 3 clips per word,** and at most 200 per speaker. This maximizes the number of distinct words, which is what hard-negative mining needs.
   - Saved as `data/bank/manifest.parquet`. It stays fixed across base-model versions.
   - **Measured (M1):** 200,000 clips, about 17.6k for most languages. Welsh (11.6k), Kinyarwanda (14.6k) and Russian (15.3k) ran out of distinct words, and the others took up the slack.
2. **Extraction:** the clips are decoded to 16 kHz int16 in `data/bank/audio.npy` (6.4 GB). It stays on the training machine.
3. **Embedding, once per base-model version:** `neg_bank.npz` holds fp16 embeddings plus word and language. **Only this file goes to the server.**
   - Each clip is embedded **with the same augmentation as the user's positives**, at a random position in the window. With clean negatives and noisy positives, the head would learn "noisy = wake word".
4. **Wake-word exclusion at fine-tune time.** If the user's wake word exists in MSWC, its clips would fight the positives. Drop bank clips that:
   - are the same word as the full wake-word label (normalized). This is the only use of the text label.
   - are very close to the user's prototype embedding. This catches spelling variants and the same word in other languages.

   Words that are only part of the phrase, like "hey" in "hey jarvis", stay in as useful negatives.

## 9. `server/`: API and fine-tune worker

**Stack:** FastAPI (native SSE), Pydantic v2, SQLModel on SQLite (Postgres later), soxr, soundfile, torch (CPU), onnx, and `runtime`.
- **Jobs:** a DB table, taken one at a time by a single worker process. Move to Celery or arq only if scale requires it.
- **Storage:** a small interface over the local filesystem (`server/storage.py`).
- **Configuration:** environment variables (`server/settings.py`): `WWD_SERVER_DATA`, `WWD_MODELS`, `WWD_BASE_VERSION`, `WWD_DEVICE`, `WWD_VERIFY_WORKERS`.

### 9.1 API

Built in M2 (`server/api.py`; OpenAPI docs at `/docs`). There's no authentication: IDs are random UUIDs, so a wake word is reachable only through its link.

```
GET    /api/health
GET    /api/languages                              the 12 codes + reading paragraphs
GET    /api/base-models                            versions, evaluation summary, default
POST   /api/wakewords                              {name, language}
GET    /api/wakewords/{id}                         the wake word, its samples and trainings
DELETE /api/wakewords/{id}                         deletes recordings, models and rows (GDPR)
POST   /api/wakewords/{id}/samples                 multipart file + kind=positive|other_speech|ambience
GET    /api/wakewords/{id}/samples/{sid}/audio     the processed 16 kHz clip (review, playback)
DELETE /api/wakewords/{id}/samples/{sid}
POST   /api/wakewords/{id}/trainings               {mode: quick|quality, target_fa_per_hour, base_version?}
GET    /api/trainings/{tid}                        status, stage, progress, report
GET    /api/trainings/{tid}/events                 SSE: `progress` events, then `done`|`failed`|`cancelled`
GET    /api/trainings/{tid}/artifacts/{file}       model.onnx, model.int8.onnx, model.json, vad.onnx,
                                                   LICENSE-silero-vad, ATTRIBUTION.md, README.md,
                                                   report.json, bundle.zip
GET    /api/trainings/{tid}/bundle.zip?fa_per_hour=x   the bundle at another operating point (M3):
                                                   same files, only the thresholds in the metadata change
```

- **The built web app** is served at `/` when `web/dist` exists (`WWD_WEB_DIST`), with `index.html` for client-side routes. The app and the API then share one origin, so a phone needs a single HTTPS tunnel.
- **`WWD_WORKER_THREAD=1`** runs the worker inside the API process, for development and the end-to-end tests. Production runs `python -m server.worker` separately.

- **Jobs:** `POST …/trainings` snapshots the sample IDs and queues a `Training` row. `python -m server.worker` claims queued rows one at a time; progress goes to `TrainingEvent` rows, which the SSE endpoint streams (it resumes after `Last-Event-ID`). A worker that starts finds jobs left `running` by a crash and queues them again.
- **One active training per wake word** (409 otherwise). `mode` accepts only `quick` (422 otherwise).
- **Deletion while training:** the worker checks between stages, stops, and removes what it wrote.
- **Storage:** everything a user owns lives under `wakewords/<id>/` (originals, clips, trainings), so deletion is one prefix. SQLite runs in WAL mode, shared by the API and the worker.

### 9.2 Upload checks

Each upload goes through these steps (`server/audio.py`):
1. Keep the original file, as received.
2. Decode anything libsndfile reads (WAV, FLAC, OGG), downmix, resample to 16 kHz with soxr. Limits: 25 MB, 180 s.
3. **Positives:** one take holds several utterances. The `runtime` Silero VAD, the same file the browser runs, cuts it, with Silero's hysteresis (start at p ≥ 0.5, end after 0.35 s below 0.35), so the short pause inside "hey jarvis" doesn't split the phrase. Each clip keeps up to 0.2 s around its speech, less for long words, so it fits the 1.5 s window.
4. **Rejected clips** are kept with a reason, so the UI can say why: `too_long` (speech over 1.3 s), `too_short` (under 0.2 s), `clipped` (0.1% of samples at full scale), `too_quiet` (speech under −55 dBFS). A whole upload is refused (422) when it's silent or holds no speech.
5. Normalize loudness: speech RMS to −20 dBFS, peak at most −1 dBFS, 10 ms fades.
6. **Outliers:** each accepted recording's cosine to the prototype of the others, through `encoder.onnx`. It's flagged below 0.5, or 0.2 below the median. Flagged recordings are left out of training; the UI should offer a re-record.
7. **Other speech** needs at least 2 s of speech. **Ambience** only has to be something other than digital silence.

### 9.3 Quick mode

Target: under 2 minutes on CPU. No GPU on the server.

1. **Positives:** each recording × about 300 augmentations: shift within the window, speed 0.9–1.1, gain, RIR, MUSAN noise at 0–20 dB SNR, random EQ.
2. **Negatives:**
   - bank embeddings, weighted toward the user's language, with the wake word excluded (§8);
   - **hard negatives:** the MSWC words closest to the user's prototype;
   - **partial wake words:** the first or last 40–70% of each recording;
   - the user's own "other speech" and ambience windows.
3. **Scorer (changed in M1): a pushed prototype, not a trained MLP.**
   - **Prototype:** the mean of the clean enrollment embeddings.
   - **Push:** subtract 0.5 × the mean of the hardest negatives, then renormalize.
   - **Score:** `logit = 20 · cos(embedding, direction) + bias`, with the bias putting the sigmoid midpoint between typical augmented positives and the 99.9th-percentile negative. It's one linear layer (`core.model.PrototypeHead`), so the compose step below doesn't change. The encoder stays frozen.
   - **Why:** on 48 held-out words at 0.5 false accepts/h, the trained MLP head reached 18% recall and the plain prototype 73%. The MLP fits augmentations of only 10 clips, never sees continuous speech, and is overconfident on long-form audio (calibrated thresholds land around 0.99).
   - **Push strength:** 0.4–0.6 is best (79% recall); 0.8 and above collapses onto the negatives. The MLP head remains available as `FinetuneConfig(scorer="mlp")`.
   - Augmented positives are still embedded; they set the bias and feed the recall estimate.
4. **Leave-2-out folds:** for each pair of recordings, fit the head on the others, calibrate it like the final model, and measure streaming recall on the held-out pair: 10 augmented copies at 6 consecutive tick offsets, 2-of-3 detection (`recall_noisy`), plus a clean copy (`recall_clean`). Embedding happens once (`core.finetune.embed_enrollment`); each fold only refits the head (`fit_head`). A recording is flagged weak when its noisy recall is under 0.5 and 0.25 below the median. The final prototype uses all recordings.
5. **Calibration, once per shipped file:** run the head plus the §4.3 smoothing over the long-form cache with the gate off, and pick the threshold that meets `target_fa_per_hour` (default 0.5/h). Occurrences of the wake word in the long-form audio are masked.
   - `model.onnx` is calibrated on `longform_cache.npz`; `model.int8.onnx` on `longform_cache.int8.npz`, the same streams embedded through `encoder.int8.onnx` itself. Each file's metadata carries its own threshold.
   - **Why separate thresholds (measured in M2):** int8 embeddings sit at cosine 0.98 from fp32 on average (1st percentile 0.88), and that moves each wake word's scores by a different amount. With fp32's threshold, int8 lost recall on some words (0.70 vs 0.85) and doubled false accepts on others (0 vs 1.4/h).
   - Base models exported before M2 lack the int8 cache; `training.export --int8-cache-only` adds it (152 s for v0.1). Without it, the int8 file shares the fp32 threshold and the report says so.
6. **Compose** (`core.export.compose_head`):
   - Export the head plus a Sigmoid from torch as a tiny ONNX graph, its names prefixed `head/`.
   - Merge it after `encoder.onnx` and after `encoder.int8.onnx` with `onnx.compose.merge_models`. Both encoders were quantized and validated once, in `training/`, and their sha256 must match `base_model.json`. The head stays float32 in both files.
   - Run the graph checker, and write the same metadata into both files.
7. **Verify** (`core.verify`): run the *shipped* files through the Python `runtime.Detector` over the 1.4 h of `longform_verify`.
   - **Scores:** one single-threaded runtime session per stream, run in parallel threads at batch 1. At batch 64, 16 parallel fp32 sessions are memory-bound and 7× slower in aggregate.
   - **Gate on:** the model is stateless, so a gate-open tick scores the same either way. The gate-off scores are replayed through the runtime's own `Gate` and `Trigger` with the stream's VAD probabilities, which are computed once per base model and cached.
   - **Each recording** is also run alone between pauses, through the full runtime path with the VAD gate (in-sample, a sanity check).
   - **Parity:** runtime scores against PyTorch on 128 ticks: the 64 the fp32 file scored highest, plus 64 at random.
   - **Report:** false accepts per hour for fp32 and int8 with the gate on and off, enrollment detections, parity, fold recall and per-stage timings, in `report.json` and the training row.
8. **Bundle** (`server/bundle.py`): `model.onnx`, `model.int8.onnx`, `model.json` (the metadata, file hashes and evaluation), `vad.onnx`, Silero's license, `ATTRIBUTION.md`, a `README.md`, and `bundle.zip` of all of them.

## 10. `web/`: React app

**Stack:** Vite, React 19, TypeScript, TanStack Query, react-router 8, and `runtime/js` (onnxruntime-web, wasm). Plain CSS with light and dark tokens, mobile first. Built in M3.

**Audio pipeline** (`web/src/audio/`):
- An AudioWorklet captures the mic and resamples it to 16 kHz.
  - Firefox won't connect a mic stream to an AudioContext running at a different sample rate, so the worklet must resample.
- The worklet sends 512-sample frames straight to a Web Worker:
  - **live test:** `detector.worker.ts` runs `runtime/js`, with the VAD on every frame and the wake model on every tick;
  - **recording:** `recorder.worker.ts` runs the same Silero VAD, keeps the audio, and drives the live feedback.
- The main thread only draws the UI.
- **Safari:** the AudioContext is created synchronously inside the tap handler. An `await` before it ends the user gesture, and Safari would refuse to start it.

**Settings:**
- `getUserMedia` with `echoCancellation`, `noiseSuppression` and `autoGainControl` all `false`.
- **Microphone dialog** (top bar, `components/MicSettings.tsx`, `audio/mic.ts`). The button's light shows the permission. The dialog offers:
  - the permission state, with "Allowed for this visit" when the browser grants without remembering (Firefox's default stays `prompt` in the Permissions API);
  - which microphone: kept in `localStorage` and used by every capture, falling back to the default if it's gone;
  - a level test, which also asks for access;
  - how to reset or remove the permission in this browser (Chrome, Android, Firefox, Safari, iOS), since a page can't do it.

  Microphone errors in the recorder, live test and field test link to the dialog.
- Uploads are 16-bit WAV files encoded in the browser at 16 kHz. Don't use MediaRecorder: Opus artifacts would hurt the training data.
- Vite serves the ORT `.wasm` files. In development it proxies `/api` to the server (`WWD_API`); in production the server serves the app.

**Flow:**
1. **Home** (`/`): the wake-word label and the language, as native names. Wake words of 4 letters or fewer get a warning; M1 measured them at about 50% recall. Wake words made on this device are listed from `localStorage`, since there are no accounts.
2. **Recordings** (`/w/:id`): three cards. Each has a record button, a live speech light and a level meter, plus **Import a file instead**, for takes recorded elsewhere (`python -m runtime.listen --record` on a Pi).
   - **The wake word, 10 times in one take.** A live counter shows utterances as the server will cut them: `segments.ts` mirrors `server/audio.py` `speech_regions` and is unit-tested on the same cases. It warns when one utterance runs past 1.3 s, and stops by itself after the 10th utterance and a 0.8 s pause.
   - **Reading the paragraph** in the chosen language, right-to-left for Persian. Takes with under 2 s of speech are refused in the browser, before any upload.
   - **15 s of room sound,** stopping by itself.
   - **Review:** each clip has its waveform, playback, a badge (good, rejected with its reason, unlike the others, or weak after a training) and delete.
   - **Train:** a checklist (usable recordings, reading, room sound), the false-alarm target and the Train button.
3. **Training** (`/w/:id/t/:tid`): progress over SSE, as a stage list and a progress bar. You can leave the page; the job continues on the server.
4. **Results:**
   - how many recordings the shipped file detects;
   - false alarms per hour, measured out of sample;
   - the in-job recall estimate, labelled as rough (§12.2);
   - the training time;
   - weak recordings, with a link back to re-record them.
5. **Test:**
   - a live canvas chart of the wake score and the VAD probability, with the threshold line and detection marks;
   - fp32 or int8, and a gate on/off toggle;
   - a **sensitivity slider** over the training's operating points, from 0.1 to 5 false alarms per hour, each with its calibrated threshold per file (§9.3). Changes apply while listening: the detector worker re-initializes without dropping the mic.
   - **Each change starts a fresh view** and resets the detection count. The new detector numbers its ticks from 0, so the worker tags its messages with a generation and the page drops stale ones.
   - **A short chime on each detection,** played on the microphone's AudioContext, which the tap on "Start listening" already unlocked. It can be muted.
6. **Download:** the bundle at the chosen operating point (`bundle.zip?fa_per_hour=…`), plus the training report.

**Two practical notes:**
- **Same mic as the device.** Open the app on the phone to record with the mic you'll deploy on. For a Pi, record with `python -m runtime.listen --record take.wav` and import the file.
- **Reading paragraphs.** One per language, 12 in total. Native speakers should check the rw, fa, eu and cy texts.

## 11. `runtime/`: detectors

- **Python (`runtime/`):**
  - `vad.py` is the Silero wrapper (state + 64-sample context).
  - `detector.py` implements §4.3.
  - `listen.py` handles mic capture and prints trigger events. Its modes:
    - `--record` saves a take from the device's own mic, for the app's import;
    - `--log DIR` writes a field-test log;
    - `--input file.wav` replays a recording instead of the mic.
  - `bench.py` reports per-tick latency, CPU % in real time and the real-time factor, per model file, with the gate on and off.
  - `fieldlog.py` defines the field-test log format, written by `listen --log` and by the web app's device test (through the server):
    - `meta.json`;
    - `events.jsonl`, with a clip of 3 s before and 1 s after each trigger;
    - `stats.jsonl`, once a minute: CPU, busy share, latency, overruns, battery;
    - `ticks.f32`: every tick's score and VAD.
  - `fieldreport.py` summarizes a log: hours, false accepts per wall hour and per speech hour, CPU, latency, and battery. Its threshold sweep replays the runtime's trigger over the logged scores, so one 24 h run gives the trigger rate at any threshold.
  - `README.md` is the device guide. `scripts/pack_runtime.sh [bundle.zip]` packs the kit: the runtime, its golden tests and test data, and the bundle (8 MB).
- **JS (`runtime/js`):** the same API and the same logic. The web app uses it, and it's published as a pnpm package for other web apps.
- **Phones:** run in the browser (wasm). Native apps can load the same `.onnx` files with ORT Mobile later; they're outside the test matrix for now.

## 12. Milestones

| # | Deliverable | Done when |
|---|---|---|
| **M0** ✅ | uv project + pnpm workspace; `core` frontend + stub encoder export; graph checker; pinned Silero VAD; `runtime/SPEC.md`; Python + JS detectors; golden vectors; CI (pytest, vitest with onnxruntime-web wasm in Node, Playwright) | **Done 2026-10-05.** pytest 33/33, vitest 21/21, Playwright 4/4 (Chromium + Firefox). Identical events in every runtime; VAD matches the official wrapper. |
| **M1** ✅ | `training`: download, prepare, student-only training, few-shot and end-to-end evals, bank, base **v0.1** | **Done 2026-10-06.** Few-shot (10 clips) recall@1%FAR **95.1%** on unseen words. End to end **77% recall at 0.5 false accepts/h** (82% at 1/h), with calibration holding out of sample (0.60/h measured). Encoder parity: wasm vs Python 1.3e-7 (fp32), int8 bit-identical. See §12.1. |
| **M2** ✅ | `server`: uploads + checks, jobs, quick mode, calibration, compose, verify, SSE | **Done 2026-10-07.** `scripts/m2_smoke.sh` drives the API with curl against v0.1: a French held-out word, a 10-utterance take, 54 s of training on the CPU, 0 false accepts in 1.4 h of held-out audio, and 48/50 (fp32) and 47/50 (int8) other speakers' utterances detected. Python and wasm runtimes fire identically on the bundle. On 24 held-out words, the shipped fp32 file reaches 80.9% recall at 0.36 false accepts/h out of sample (PyTorch: 79.7%); int8 reaches 78.9% at 0.38/h. See §12.2. |
| **M3** ✅ | `web`: record → review → train → live test → download | **Done 2026-10-07, on emulated devices.** The full flow passes in Chromium, Firefox and WebKit (iPhone 15 emulation), in the browser end-to-end tests against a real API. With the real v0.1 model in Chromium, the live test detected 19 of 20 other speakers saying the word. Still to do on your side: a real iPhone over HTTPS (§2, item 5). See §12.3. |
| **M4** ✅ | Devices: Pi 4/5 (Python ORT on aarch64), phone browser (wasm) | CPU %, latency and a 24 h real-room false-accept test (TV + speech) measured on each. **Done 2026-10-08, with a shortened field test (your call).** It runs on a Raspberry Pi 4B and in Chrome on Android. On the Pi 4: 40 ms per tick, 36–42% of one core. No false accepts in about 1 h on the Pi, 1 h on the phone and 3 h on a computer, for models calibrated to 0.25/h. The 24 h runs and the phone benchmark weren't done; the tooling is ready for them. See §12.4. |
| **M5** ✅ | Base **v1**: XLS-R teacher + distillation | End-to-end metrics beat v0.1. Older user models keep their base version. **Done 2026-10-09.** End to end **82.4% recall at the 0.5 false accepts/h target** (v0.1: 77.0%), +4.6 points at the same measured rate. The shipped fp32 file: 85.2% (v0.1: 80.9%). Few-shot (10 clips) 96.7% (v0.1: 95.1%). Same student, so the same cost on every device. New trainings use v1; existing ones keep v0.1. See §12.5. |

M2 and M3 can start right after M0, against the stub encoder, while M1 trains.

### 12.1 M1 results (base model v0.1)

The base model is BC-ResNet (204k parameters, 256-dim embedding), trained for 80k steps of Sub-center ArcFace over 85,258 words in the 12 languages, about 4 h on the RTX 4060 Ti. It lives in `models/base/v0.1/`; `base_model.json` holds every number below.

**Few-shot, on held-out words** (300 words × 40 clips per language, recall at 1% false alarms per clip):
- **Mean:** 1 clip 82.4%, 5 clips 93.9%, 10 clips **95.1%**. EER at 10 clips: 1.9%.
- **Per language, 10 clips:** ru 98.6, ca 98.3, de 97.9, eu 97.7, it 97.3, pl 97.3, cy 95.2, es 95.0, en 94.1, fr 93.8, fa 93.6, rw 82.5.

**End to end** (`eval_e2e`, 240 held-out words). Enrollment is 10 clips; testing uses other speakers and streaming 2-of-3 detection.

| Language | rw | en | fa | es | fr | cy | it | eu | pl | ca | ru | de | **mean** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Recall @ 0.5 FA/h | 46 | 68 | 72 | 72 | 74 | 79 | 81 | 85 | 85 | 86 | 87 | 91 | **77** |

- **Other operating points:** 82% at 1 false accept/h, 85% at 2/h.
- **Out-of-sample false accepts:** 0.60, 1.14 and 2.28 per hour for targets of 0.5, 1 and 2. Calibration slightly undershoots, by about 15–20%.
- **Per word:** median 84%; 59% of words reach at least 80%, and 12% fall below 50%.
- **Word length matters most:** 50% recall for words of 4 characters or fewer, 73% for 5–6, 83% for 7–8, 87% for 9 or more.
- **More enrollment speakers help:** 71% with at most 3 speakers, 85% with 7–10.
- **This test is cross-speaker,** which is harder than a user detecting their own voice.
- **The scorer matters more than anything else measured:** the MLP head planned for §9.3 gave **20%** on the same protocol. §9.3 explains why it was replaced.

**int8:** embeddings agree with fp32 at cosine 0.98 on average. Few-shot recall with 10 clips drops 95.6 → 95.1, and with 1 clip 84.6 → 82.8. Whether int8 holds at the 0.5 FA/h operating point is the M2 verification step's job (answered in §12.2: yes, with its own threshold).

**Next levers (v0.2)**, roughly by expected value:
1. Phrase classes, so multi-word wake words fit the 1.5 s window and train as single units.
2. Partial-FC plus more steps: 2.8 epochs is short for 85k classes, and accuracy was still rising at the end.
3. Better Kinyarwanda coverage.
4. In the app, guide users toward long wake words and several voices at enrollment.

### 12.2 M2 results (fine-tuning server)

**Shipped files on 24 held-out words** (`training.eval_shipped`: 2 words per language, the same draw as `eval_e2e`). Each word goes through the server's own quick mode (`server.quick.run`). Then the Python runtime runs the shipped files over a stream of 10 to 50 other speakers saying the word, 3 s apart, plus the 1.4 h verification audio. The target is 0.5 false accepts/h.

| | Recall, gate off | Recall, gate on | False accepts/h (out of sample) |
|---|---|---|---|
| PyTorch (`eval_e2e`, same words) | 79.7% | | 0.36 |
| `model.onnx` (fp32) | **80.9%** | 80.1% | **0.36** (12 in 34 h) |
| `model.int8.onnx`, own threshold | 78.9% | 78.5% | 0.38 (13) |
| `model.int8.onnx`, fp32's threshold | 79.0% | 78.7% | 0.44 (15) |

- **The shipped fp32 file reproduces PyTorch.** That closes M1's open gap.
  - Per-tick difference: at most 2.9e-6.
  - Per-word recall: within 6 points, 1.1 points higher on average. The protocols differ slightly: here every tick is streamed, while `eval_e2e` uses 6 shifts.
  - Median word recall is 82%. 13 of 24 words reach 80%; one (Kinyarwanda) falls below 50%.
- **The speech gate** costs under 1 point of recall. Over these streams it removed no false accepts: they happen during speech, which the gate lets through.
- **int8 needs its own threshold.**
  - int8 and fp32 scores differ by up to 0.16 on single ticks (mean over words of each word's worst tick), and the shift depends on the word. With fp32's threshold, int8 recall dropped 15 points on one word, and another word's false accepts went from 0 to 1.4/h.
  - Each file is now calibrated on its own scores (§9.3 step 5). Thresholds differ by 0.038 on average, up to 0.09. int8 then hits the target on the calibration audio exactly like fp32 (0.487/h).
  - **The remaining cost of int8:** about 2 points of recall at the same false-accept rate.
- **The in-job recall estimate (leave-2-out folds) is pessimistic and noisy.** It averages 62%, against 81% measured on other speakers, and correlates only 0.41 with the measured recall per word. It is fit on 8 recordings and tested under 0–25 dB SNR noise. Present it in the UI as a rough quality signal, not a promise.
- **Speed.** Quick mode takes 52–54 s per wake word on the CPU (16 threads), against the 2-minute target:

  | Stage | Time |
  |---|---|
  | Embedding (3000 augmented positives, 360 partials, recall windows) | 11–12 s |
  | Folds | 1.3 s |
  | Calibration (both files) | 0.3 s |
  | Compose | under 0.1 s |
  | Verify (both files over 1.4 h, parallel runtime sessions) | 39 s |

  With the encoder on the GPU, it takes 38 s.
- **Upload checks on real takes.** VAD segmentation cut a 10-utterance take into 10 clips. Leave-one-out similarity was 0.81–0.95; the one flagged outlier (0.70) is a truncated MSWC clip.
- **Smoke run** (`scripts/m2_smoke.sh fr`, the M2 done-check; it needs the `training/` data to build its inputs):
  - **Inputs:** the held-out word "morceaux", 10 enrollment clips from one speaker in one take, 40 other words by that speaker as other speech, and 15 s of MUSAN ambience.
  - **Training:** 9 recordings kept, 53.9 s on the CPU. Calibrated thresholds: 0.064 (fp32) and 0.081 (int8).
  - **On the shipped files:** 0 false accepts in 1.4 h. Through the Python runtime, 48/50 (fp32) and 47/50 (int8) of other speakers' utterances detected, with no other events.
  - **Parity:** the wasm runtime reproduces the Python runtime on both test files (9/9 tests; events identical; no tick straddles a threshold).

### 12.3 M3 results (web app)

**Automated, in CI** (`web/e2e/app.spec.ts`). The test API runs on a tiny base model (`server/tests/tiny_base.py`) with the worker in-process, and the app is the production build. One test walks the whole flow in each engine:
1. Create a wake word.
2. Import a take: the server cuts it into 6 clips.
3. Record the reading paragraph through the real mic pipeline (Chromium's fake mic plays speech, so the take is kept and the speech light must turn on).
4. Import the room sound, and train.
5. Wait for the results.
6. Listen live, and switch model, gate and sensitivity while listening: ticks must keep flowing, and in Chromium the VAD must hear the speech.
7. Download the bundle retargeted to 5 false alarms/h, and delete everything.

| Engine | Flow | Runtime golden parity | Mic → worklet → worker |
|---|---|---|---|
| Chromium | ✓ 19 s | ✓ (fp32 7.6e-7, int8 1.1e-2) | ✓ 48 kHz → 16 kHz, VAD max 1.00 |
| Firefox | ✓ 22 s (its tone mic: the app refuses the no-speech take, then a file is imported) | ✓ same values | ✓ |
| WebKit, iPhone 15 emulation | ✓ 19 s | ✓ same values | ✓ 44.1 kHz → 16 kHz, VAD max 0.79 |

**Real model, by script** (not in CI: it needs v0.1 and the M2 smoke inputs). The API serves the built app (single origin) with the v0.1 base model.
- **Setup:** enroll the French word "morceaux" through the UI by importing the 10-utterance take, other speech and room sound.
- **Training:** 55 s. 9 of 10 clips kept; the truncated one was flagged "unlike the others".
- **On the held-out audio:** 9/9 recordings detected, 0 false alarms in 1.4 h, 86% in-job recall estimate.
- **Live test:** Chromium's fake mic played other speakers saying the word, 3 s apart, through the 48 kHz → 16 kHz worklet. The fp32 model detected **19 of the 20** utterances in 61 s, with no other detections; the Python runtime got 96% on the same audio in M2.

**Not done:** a real iPhone and a real Android phone over HTTPS. Playwright's WebKit is Safari's engine at phone size, with touch and mock capture devices, but not iOS's audio stack or its permission prompts. That check needs your devices and an HTTPS tunnel (§2, item 5), and belongs with M4's device work.

### 12.4 M4 results (devices)

**Protocol, the same on every device:**
1. **Parity:** the runtime's golden tests on the device's own onnxruntime build (`python -m pytest runtime/tests` on a Pi; the browser runs the same checks in `/harness.html`).
2. **Benchmark:** 60 s (Python) or 30 s (browser) of recorded speech, streamed one 96 ms tick at a time. It reports latency per tick (p50/p95/max) and the CPU the detector needs in real time, for fp32 and int8 with the gate on and off.
3. **Field test, 24 h** where the device will live: TV on part of the day, people talking, the wake word not said (or the times noted).
   - Every trigger is logged with its clip. You listen to the clips, mark the real wake words, and the rest are false accepts.
   - Also logged: CPU and latency every minute, battery on phones, every tick's score.
4. **Recall in the room (optional):** 20 utterances at about 1 m and 20 at about 3 m with the TV on.

**Where each runs:**
- **Pi:** `scripts/pack_runtime.sh bundle.zip`, copy `dist/wwd-runtime.tar.gz`, then follow `runtime/README.md`: install, `pytest`, `runtime.bench`, then `runtime.listen --log` under `systemd-run`, and `runtime.fieldreport` at the end. It needs the 64-bit OS: onnxruntime has no 32-bit ARM wheels. The kit was checked in a fresh Python 3.11 venv with only its own requirements: 31 tests pass and the benchmark runs.
- **Android (Chrome):** the app's **Device test** page (`/w/:id/t/:tid/device`, linked from a finished model).
  - **Benchmark:** saved with the model.
  - **Field test:** keeps the screen on (Screen Wake Lock) and uploads its log every minute, in the Pi's format, so `runtime.fieldreport` reads both and a killed tab loses at most a minute.
  - **Connecting the phone:** over USB, `adb reverse tcp:8000 tcp:8000` makes the server the phone's `localhost`, a secure context with no HTTPS needed.
  - **What a page can't measure:** process CPU. "Busy" is the detector's own time in its worker over the audio time; Chrome adds the battery level.
- **aarch64 before a Pi:** `scripts/test_aarch64.sh` runs the runtime tests on the Pi's onnxruntime build (Python 3.11, aarch64 wheels, Debian arm64 libraries, all fetched without root) under qemu, once `qemu-user-static` is installed. CI also has a `runtime-arm64` job on GitHub's arm64 runners, for when the repo has a remote.

**Reference numbers (2026-10-08, base v0.1, the French "morceaux" model):**

| Device | Runtime | Per tick, fp32 (p50/p95) | Per tick, int8 | CPU / busy in real time, fp32 gate on |
|---|---|---|---|---|
| Ryzen 7 7700X | Python onnxruntime 1.30, 1 thread | 2.1 / 2.5 ms | 2.3 / 2.5 ms | 1.9% of a core (int8 2.0%) |
| Ryzen 7 7700X | Chromium, wasm, 1 thread | 6.8 / 7.5 ms | 11.3 / 12.3 ms | 6.0% busy (int8 9.8%) |
| Ryzen 7 7700X | Firefox, wasm, 1 thread | 7.0 / 8.0 ms | 12.0 / 13.0 ms | 6.3% busy (int8 10.1%) |
| Raspberry Pi 4B (Cortex-A72, 4 cores) | Python 3.13 onnxruntime 1.30, aarch64, 1 thread | 40.4 / 40.8 ms | 41.7 / 41.9 ms | 36% of a core (gate off 42%; int8 37% / 44%) |
| Android phone | Chrome, wasm | not measured | not measured | not measured |

- **int8 is faster on no device measured,** so `model.onnx` is the one to deploy, and the bundle's README now says so:
  - **Browsers:** about 1.7× slower than fp32, because QDQ adds work the wasm kernels don't recover.
  - **x86:** even with fp32.
  - **Pi 4:** even. One window takes 36.3 ms fp32 and 36.4 ms int8. The Cortex-A72 is ARMv8.0, without the int8 dot-product instructions (`sdot`) that make int8 pay off.
  - int8 still saves 0.4 MB of file and costs about 2 points of recall (§12.2). A Pi 5 (Cortex-A76, ARMv8.2, with `sdot`) may differ; it's unmeasured.
- **Raspberry Pi 4B in detail** (the 60 s benchmark):
  - **Time per tick:** the wake model takes 36.3 ms per window; three VAD frames take 1.58 ms each, 4.7 ms per tick.
  - **Headroom:** a tick takes 40 ms against its 96 ms, a real-time factor of 0.36–0.42, so one core keeps up with 2.4× to spare.
  - **CPU:** the detector uses about a third of one of the four cores while people talk. In silence the gate leaves only the VAD, about 5% of a core.
  - **Memory:** 85 MB peak. A Ryzen 7700X is 19× faster per tick.
  - **If a Pi 4 ever needs less CPU:** the conv-STFT frontend is about a third of the multiply-adds and recomputes overlapping windows every tick, so a streaming frontend is the lever.
- **The field test, shortened to about 5 h (your call).** You ran it about 1 h on the Pi 4B, 1 h in Chrome on Android, and 3 h on a computer, with "hey jarvis" models (French, 10 recordings) calibrated to 0.25 false accepts per hour. There were **no false accepts**.
  - **Expected:** about 1.25 at the calibrated rate, so seeing none has a 29% chance and is consistent.
  - **What it bounds:** the rate stays below roughly 0.6/h at 95% confidence (3 / 5 h), across all three devices.
  - **Still open:**
    - 24 h runs with TV and conversation, to get tighter numbers;
    - the Android benchmark (one minute on the Device test page), for phone latency and CPU;
    - the golden parity tests on the Pi itself (`python -m pytest runtime/tests` in the kit).
- **The gate** saves less than 20% here, because the benchmark audio is speech 81% of the time. In a quiet room it closes most of the time.
- **Peak memory** of the Python runtime: 82 MB.
- **The logger, checked by replay:** the M2 test recording (50 utterances) through `runtime.listen --input --log` gives 48 triggers with clips. `fieldreport`'s replay at the run's threshold reproduces those 48 exactly.

### 12.5 M5 results (base model v1)

`scripts/m5_pipeline.sh v1` ran the teacher, its cache, the distilled student, the export and every evaluation unattended. That's about 12.5 h of GPU time on the RTX 4060 Ti, plus about 1 h lost to the first teacher attempt. The result lives in `models/base/v1/`, and `base_model.json` holds every number below. v1's student is the same BC-ResNet as v0.1 (204k parameters), so runtime cost, file size and the ONNX contract don't change.

**Teacher** (`models/runs/teacher`): frozen XLS-R, 18 layers, 28k steps, 4.1 h.
- **Learned layer mix:** it settled on layers 18, 17 and 9.
- **The first attempt collapsed** with random ArcFace class centers (§7 step 4). Starting each center from 2 clips of its word fixed it: 97.4% at step 1,500, 98.4% at 4k, then a plateau from about 12k.

**Few-shot on held-out words** (`training.evaluate`, 300 words × 40 clips per language, recall at 1% false alarms):

| | 1 clip | 5 clips | 10 clips | EER, 10 clips |
|---|---|---|---|---|
| v0.1 | 82.4% | 93.9% | 95.1% | 1.88% |
| **v1** | **86.2%** | **95.8%** | **96.7%** | **1.41%** |
| Teacher (doesn't ship) | 95.3% | 98.5% | 98.7% | 0.74% |

- **Per language, 10 clips:** every language improves, by 0.4 (ru) to 3.5 points (rw: 82.5 → 86.0).
- **How much of the teacher v1 captured:** about 30% of its 1-clip lead and 45% of its 10-clip lead. The student ended at a mean cosine of 0.71 to the teacher's embeddings.
- **Training speed:** distillation made the student converge an order of magnitude faster. It had 95.1% at step 8k (on the in-training subset), where v0.1 had 23.9%; v0.1 needed about 50k steps to get there.

**End to end** (`training.eval_e2e`): the same 240 held-out words and protocol as §12.1.

| Target false accepts/h | 0.5 | 1 | 2 |
|---|---|---|---|
| v0.1 recall (false accepts/h measured out of sample) | 77.0% (0.60) | 81.7% (1.14) | 85.4% (2.28) |
| **v1 recall** | **82.4%** (0.69) | **85.4%** (1.27) | **88.0%** (2.43) |

- **At the same measured false-accept rate,** v1 gains **4.6 points**: v0.1 interpolates to 77.7% at 0.69/h. v1 at 0.69/h even beats v0.1 at 1.14/h.
- **v1's calibration undershoots a little more:** 232 false accepts against 203 over the same 338 word-hours (37% over target, against 20%). That's within noise (about 1.4σ), and calibration on the cache is identical (0.487/h for both).
- **Per word:** 155 words better, 30 worse, 55 equal. The median goes from 84% to 89%. Words reaching 80% go from 59% to 70%; words below 50% go from 12% to 7%.
- **Per language, at 0.5/h:**

  | Language | ca | cy | de | en | es | eu | fa | fr | it | pl | ru | rw |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|
  | v0.1 | 86 | 79 | 91 | 68 | 72 | 85 | 72 | 74 | 81 | 85 | 87 | 46 |
  | **v1** | **92** | **85** | **94** | **74** | **79** | **91** | **78** | **82** | **87** | **90** | **92** | 44 |

- **Kinyarwanda didn't move** (46% → 44%), even though its few-shot recall improved (82.5% → 86.0%). Why isn't measured yet (§12.5, next levers).
- **Short words gain the most:** words of 4 characters or fewer go from 50% to **59%**. Words of 5–6 characters go 73% → 79%, 7–8 go 83% → 88%, and 9 or more go 87% → 90%.
- **Enrollment speakers:** with at most 3 speakers 71% → 76%, with 7–10 speakers 85% → 91%.

**Shipped files** (`training.eval_shipped`): 24 words, the server's quick mode, then the Python runtime. Target 0.5/h, gate off.

| | Recall | False accepts/h (in 34 h) |
|---|---|---|
| v0.1 `model.onnx` | 80.9% | 0.36 (12) |
| **v1 `model.onnx`** | **85.2%** | 0.41 (14) |
| v0.1 `model.int8.onnx` | 78.9% | 0.38 (13) |
| **v1 `model.int8.onnx`** | **82.9%** | 0.41 (14) |

- **PyTorch reference:** 85.3%, so the shipped fp32 file reproduces it again. The gate costs under 1 point.
- **int8 still costs about 2 points of recall.** Its worst single-tick shift from fp32 averages 0.15 per word (v0.1: 0.16). One word reached 0.68, which its own threshold absorbs.
- **In-job recall estimate:** 67% (v0.1: 62%). It's still pessimistic, as §12.2 explains.
- **Quick mode is as fast as before:** 38 s per word with the encoder on the GPU, 53 s on the CPU.

**Through the server and both runtimes** (`scripts/m2_smoke.sh fr v1`): the same "morceaux" take as M2.
- **Training:** 9 recordings kept, 52.6 s on the CPU.
- **Held-out audio:** 0 false accepts in 1.4 h, 9/9 recordings detected.
- **Python runtime:** 47/50 (fp32) and 45/50 (int8) of other speakers' utterances, with no other events. v0.1 got 48/50 and 47/50, a difference within noise for one word.
- **wasm runtime:** reproduces the Python runtime (9/9 bundle tests).

**Serving v1:** the server's default base is the newest, so new trainings use v1, while each existing training keeps the base it recorded (§9). Bundles built on v1 credit XLS-R in `ATTRIBUTION.md`, because `base_model.json` records the teacher.

**Next levers (v1.1)**, roughly by expected value:
1. **Phrase classes** (§12.1, lever 1). Multi-word wake words like "hey jarvis" are still trained as single words.
2. **Close more of the teacher gap.** The student has captured 30–45% of it. Options: distill on long-form speech windows too (unlabeled, and where false accepts happen), a longer schedule, or a width-40 student if devices can afford the cost (§12.4 has the Pi's headroom).
3. **Kinyarwanda:** better embeddings didn't help it end to end. Look at its false accepts and its long, agglutinative words.

### 12.6 Everything in the browser (on `main` since 2026-10-09)

**Goal:** the web app does the whole job in the browser, served as static files with no server. That covers recording, upload checks, training, calibration, verification, the live test, the download and the device tests. This lifts §1's "training on the device" exclusion for browsers. Quick mode makes it possible: the encoder is frozen and the head is fitted in closed form (§9.3), so a training is about 4,200 encoder forward passes plus vector maths, with no backprop and no WebGPU.

**What replaces the server** (`web/src/local/`):

| Server | Browser |
|---|---|
| SQLite + files (`server/db.py`, `storage.py`) | IndexedDB (`db.ts`). Files are stored as ArrayBuffers: WebKit can't store Blobs in ephemeral sessions (private browsing, automated tests). |
| Upload checks (`server/audio.py`) | `checks.ts`: the runtime's Silero VAD + `audio/segments.ts`, same thresholds and messages. Imports are decoded by the browser (`OfflineAudioContext` at 16 kHz). |
| Quick mode (`server/quick.py`, `core/finetune.py`, `core/calibrate.py`, `core/augment.py`) | `train.ts`, `finetune.ts`, `augment.ts` + `fft.ts`, stage for stage, in a compute worker (`backend.worker.ts`). Embedding runs in up to 4 workers (`pool.ts`), each with its own single-threaded onnxruntime-web: parallel without SharedArrayBuffer, so no cross-origin-isolation headers are needed. |
| `compose_head` (onnx library) | `compose.ts`: the export ships the encoder composed with a marker head and the byte offsets of its Gemm weight and bias. The browser writes the trained head there and appends `metadata_props` entries (protobuf merges fields that come later). |
| Verification through the Python runtime | False accepts on the held-out ticks' embeddings, through `runtime/js`'s own `Gate` and `Trigger`. Every recording goes through `Detector` running the composed file. The file is checked against the head's maths. |
| Bundle (`server/bundle.py`) | `bundle.ts` + a stored-zip writer (`zip.ts`). |
| Device runs (`server/device.py`, `runtime/fieldreport.py`) | `backend.ts` + `fieldreport.ts`. |
| SSE progress | An in-tab subscription. Each training holds a Web Lock, so another tab knows it's alive. A training whose tab closed is marked failed. |

**Static assets** (`scripts/export_web_assets.py models/base/v1` → `web/public/base/`): **230 MB** for v1, downloaded on the first training and kept in Cache Storage. Every URL carries its file's hash.
- **Calibration ticks** (14.4 h): 138 MB as int8 with per-row scales.
- **Negative bank** (200k): 51 MB.
- **Verification ticks:** 14 MB.
- **Augmentation:** 5 min of noise, 2 min of babble, 200 RIRs: 17 MB.
- **Model template and encoder:** 3 MB.

What the reductions cost, measured with `eval_e2e` on the 240 words (fed back into the Python pipeline with `--variant`):

| Assets | Size | Recall at 0.5/h (measured FA/h) |
|---|---|---|
| v1 as shipped to the server | 940 MB | 82.35% (0.69) |
| int8 bank and cache, smaller augmentation set | 230 MB | **82.41%** (0.68): no cost |
| + bank 100k, cache 7.0 h | 131 MB | 81.24% (0.62): −1.2 points |

**Deliberate differences from the server:**
- **`model.onnx` only.** int8 would need its own 138 MB calibration cache, and it's slower in browsers anyway (§12.4). The app hides the fp32/int8 choice for browser trainings.
- **Different random streams.** Seeded, with one derived generator per window, so the result doesn't depend on which worker embeds what. The distributions match the server's.
- **Reverb and EQ in one pass.** When a window gets both, they're applied together in the frequency domain, with the reverb's loudness match after the EQ. The level is re-randomized anyway.

**Results:**
- **The ported maths match Python:**
  - trigger counts and calibrated thresholds (to 1e-12);
  - stream recall, quantiles, folds, word normalization, speed perturbation;
  - the pushed-prototype head (to 1e-4);
  - the composed model's scores (within 1e-5, metadata identical to `compose_head`'s).

  Golden values come from `scripts/make_web_fixtures.py`, checked by `web/src/local/golden.test.ts`.
- **Same models as the server** (`scripts/web_quality.py`):
  - **Method:** the browser code trained the 24 held-out words of `eval_shipped` under Node with the real v1 assets, and the Python runtime scored the composed files exactly as it scores the server's.
  - **Recall: 85.22%**, the server's figure, identical word for word.
  - **False accepts:** 15 against 14 in 34 h (0.44 vs 0.41/h).
  - **Self-check:** the browser's own estimate, from quantized embeddings, was 0.41/h.
  - **Speed:** 40 s per word on one Node thread.
- **In real browsers** (Ryzen 7700X, assets served from localhost; "célébrer", a 10-utterance take):
  - the upload checks take 1.8 s;
  - training takes **13.7 s in Chromium and 13.8 s in Firefox** (about 8.6 s embedding across 4 workers, 2.5 s of folds, 0.5 s calibration, 1.5 s verification);
  - the first training adds the 230 MB download: 16.2 s and 16.9 s over localhost;
  - 8/9 recordings detected, 0 false alarms in 1.4 h, 86% estimated recall in noise.
- **End-to-end tests** (`web/e2e/`, on the tiny base): 15/15 in Chromium, Firefox and WebKit (iPhone).
  - The `vite preview` static build covers the whole flow, with training in the page, the live test and the download.
  - It also covers the device benchmark and field test, the microphone dialog, and runtime parity.
  - Chromium needed one fix: caching a response clone while reading the original failed with "network error", so files are cached once fully read.

**Publishing** (`.github/workflows/ci.yml`): every push builds the site. On `main`, once every test job passes, the CI downloads the base model's files from a GitHub release (`base.tar.gz`, 208 MB, made by `scripts/release_base.sh`; too big for git, and one file is 138 MB), builds under the Pages sub-path and deploys. The router uses hash URLs (`/wwd/#/w/…`) so every page loads from any static host without a fallback rule. The end-to-end tests run the build under `/wwd/` too.

**Open:**
- **The 230 MB first download** is fine on a desktop and heavy on mobile data. The 131 MB variant costs 1.2 points.
- **Not measured on a phone yet:** training speed, memory (the compute worker holds about 230 MB of base data, plus 4 encoder workers), and whether iOS keeps the tab alive for the training. The tab must stay open while training.
- **Browser storage is best-effort.** Under storage pressure the browser may evict the cache and the recordings. The app doesn't call `navigator.storage.persist()`, because Firefox prompts for it. The downloaded zip is the durable copy.
- **The quality check runs under Node,** not in a browser. The browser runs differ only in the wasm build's kernels, which `runtime.spec.ts` holds to golden parity.

## 13. Decisions

**Made:**
- **Window:** 1.5 s (fits 2–3-word phrases), hop 1536 samples.
- **Languages:** the 12 MSWC high-resourced languages only. No Common Voice.
- **Runtimes:** onnxruntime-web (wasm) + Python onnxruntime, held to golden-vector parity.
- **int8 contract:** per-tick drift ≤ 2e-2 (mean ≤ 1e-3) across runtimes on the golden vectors, with identical events. For real bundles the bounds are those measured in M2 (§3). int8 agrees with fp32 on the evals (§12.1, §12.2).
- **Thresholds per file:** `model.onnx` and `model.int8.onnx` each carry a threshold calibrated on their own scores (§9.3 step 5).
- **Verification replays the gate:** the gate-on numbers reuse the gate-off scores through the runtime's own `Gate` and `Trigger`, which is exact because the model is stateless (§9.3 step 7).
- **Outliers stay out of training:** recordings flagged at upload are excluded, and the UI asks for a re-record.
- **Sensitivity = operating points:** the slider picks a calibrated false-alarm rate (0.1–5/h), not a raw threshold, and a download at another point only rewrites the metadata.
- **One origin:** the server serves the built app, so phones need one HTTPS tunnel and there's no CORS.
- **Mobile Safari stand-in in CI:** Playwright WebKit with iPhone emulation. Real devices are checked by hand (M4).
- **Layout:** flat Python packages (`core/`, `training/`, `server/`, `runtime/`), one uv project.
- **Teacher:** XLS-R 300M (Apache-2.0), frozen on the local GPU, first 18 layers, class centers initialized from data (§7 step 4).
- **Base v1 is the default** for new trainings (§12.5); trainings made on v0.1 keep it.
- **Default export:** one merged file per wake word.
- **Quick-mode scorer:** the pushed prototype (§9.3), replacing the MLP head: 77% vs 20% recall at 0.5 FA/h.

**Still open:**
- Also export a shared `encoder.onnx` plus a small `head.onnx` per word, for devices that run several wake words?
- Build a native phone app later?

## 14. Risks

- **False accepts with about 10 samples.** Measured in M2: the target holds out of sample on average (0.36/h for a 0.5/h target), but 1.4 h of verification audio per job can only resolve about 0.7/h, so a single word's report is coarse. A second-stage verifier can be added to the runtime later.
- **The in-job recall estimate is a weak predictor** (correlation 0.41 with measured recall, §12.2). Better: more held-out audio per fold, or a learned predictor from `eval_shipped` runs.
- **Always-on listening in a phone browser** only works while the page is in the foreground; iOS Safari stops the mic when the screen locks. True background listening needs a native app, and the ONNX contract keeps that possible.
- **Mismatch between the recording mic and the device mic.** Mitigated by EQ augmentation and by recording on the target device.
- **Languages with little data** (cy, eu, pl, ru: about 110–140 h each) turned out fine (v1: 85–92% end to end). The weak one is **Kinyarwanda (44% in v1)**, despite 422 h: it has only 322 speakers, and its words are long and agglutinative. Distillation improved its few-shot recall but not its end-to-end recall (§12.5). Short wake words (4 characters or fewer) are weak in every language: 59% in v1, up from 50%.
- **Privacy.** Voice recordings are personal data under GDPR. The delete endpoint exists (M2: it removes every file and row, and stops a running job). There's no retention policy and no authentication (out of scope, §1): recordings stay until their wake word is deleted, so the server suits personal use, not an open deployment.
- **Reading paragraphs** (`server/languages.py`) were written without native speakers. The rw, fa, eu and cy texts are flagged `needs_review`.

## Sources

- [EdgeSpot (arXiv 2601.16316)](https://arxiv.org/abs/2601.16316)
- [Gok et al., SSL few-shot KWS (arXiv 2506.17686)](https://arxiv.org/abs/2506.17686)
- [MLCommons/ml_spoken_words on Hugging Face](https://huggingface.co/datasets/MLCommons/ml_spoken_words)
- [silero-vad (`utils_vad.py`, `data/`)](https://github.com/snakers4/silero-vad)
- [facebook/multilingual_librispeech](https://huggingface.co/datasets/facebook/multilingual_librispeech)
- [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [MUSAN (OpenSLR 17)](https://www.openslr.org/17/) · [RIRS_NOISES (OpenSLR 28)](https://www.openslr.org/28/) · [LibriSpeech (OpenSLR 12)](https://www.openslr.org/12/)
- [Common Voice moves to Mozilla Data Collective](https://github.com/common-voice/cv-dataset/)
