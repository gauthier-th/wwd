# wwd: few-shot wake-word trainer

Record a wake word in the browser, fine-tune on the server, and get ONNX files that run
identically on onnxruntime (Python, e.g. a Raspberry Pi) and onnxruntime-web (wasm). See
[PLAN.md](PLAN.md) for the design and [runtime/SPEC.md](runtime/SPEC.md) for the runtime contract.

## Setup

```sh
uv sync --extra training      # CUDA torch (GPU box); use --extra server for CPU-only
pnpm install
```

## Fine-tuning server

```sh
uv run --extra server uvicorn server.api:app       # API on :8000 (docs at /docs)
uv run --extra server python -m server.worker      # fine-tune jobs, one at a time
```

On the GPU box, `--extra training` runs both too (it carries the API stack).

Base models are read from `models/base/<version>/` (`WWD_MODELS`, `WWD_BASE_VERSION`); the
database and recordings go to `data/server/` (`WWD_SERVER_DATA`). `scripts/m2_smoke.sh [lang]`
drives a whole run with curl against a real base model and checks the bundle in both runtimes.

## Base models

Built offline on a GPU by `training/` (PLAN.md §7). The newest is the server's default; each
training keeps the base it was made with.

```sh
scripts/m1_pipeline.sh v0.1   # data download, splits, student-only base model, evaluations
scripts/m5_pipeline.sh v1     # XLS-R teacher, its embedding cache, distilled student, evaluations
```

Both run for hours and resume where they stopped: start them detached, e.g.
`setsid nohup scripts/m5_pipeline.sh v1 >> data/m5.log 2>&1 < /dev/null &`.

## Web app (static, trains in the browser)

On this branch the web app needs no server. Recording, the upload checks, training,
calibration, the live test and the download all run in the browser (Web Workers, onnxruntime-web
wasm). Recordings, trainings and models stay in IndexedDB, so the site is plain static files.

1. Export the base model's files for the browser, once per base model (about 230 MB):

   ```sh
   uv run --extra training python -m scripts.export_web_assets models/base/v1   # -> web/public/base/
   ```

2. Build, then serve `web/dist/` with any static file server:

   ```sh
   pnpm --filter @wwd/web build        # WWD_BASE_PATH=/sub/path/ to serve it under a sub-path
   pnpm --filter @wwd/web preview      # or: pnpm --filter @wwd/web dev
   ```

The first training downloads the base model's files (about 230 MB), and Cache Storage keeps them
for later visits. Browsers only open the microphone on `https://` or `localhost`, so serve the
site over HTTPS, for example GitHub Pages or `cloudflared tunnel --url http://localhost:4173`.
No cross-origin-isolation headers are needed.

The fine-tuning server (`server/`) still exists, and the evaluation tools use its quick mode,
but this app doesn't call it.

## Tests

```sh
uv run pytest                 # core, server (tiny base model) + Python runtime (golden parity)
uv run ruff check . && uv run lint-imports
pnpm --filter @wwd/runtime test        # JS runtime, onnxruntime-web wasm under Node
uv run python -m scripts.make_web_fixtures   # tiny base model + web assets + Python goldens
pnpm --filter @wwd/web test            # app unit tests: the ported maths against Python,
                                       # a whole browser training on the tiny base under Node
pnpm --filter @wwd/web test:e2e        # Chromium, Firefox, WebKit (iPhone): the static app on the
                                       # tiny base, training in the browser; runtime parity, the mic

# Browser training against the server's, on 24 held-out words (needs training/ data):
uv run --extra training python -m scripts.web_quality prepare data/web_quality
WWD_QUALITY=../data/web_quality pnpm --filter @wwd/web exec vitest run quality
uv run --extra training python -m scripts.web_quality score data/web_quality
```

## Run a model

```sh
python -m runtime.listen model.onnx --vad vad.onnx      # files from the downloaded bundle
python -m runtime.listen --record take.wav              # a take from the device's own mic,
                                                        # for the app's "Import a file instead"
```

`/harness.html` in the web app runs the M0 runtime checks (golden parity, microphone pipeline).

## Measure a device (M4)

- **Raspberry Pi or any Python host:** `scripts/pack_runtime.sh bundle.zip` builds
  `dist/wwd-runtime.tar.gz`; [runtime/README.md](runtime/README.md) walks through parity tests,
  `python -m runtime.bench` and a 24 h field test (`runtime.listen --log`, then `runtime.fieldreport`).
- **Phone:** open a finished model in the app, then "Measure this device". On Android over USB,
  `adb reverse tcp:8000 tcp:8000` lets the phone open `http://localhost:8000` with the microphone.
- **aarch64 without a Pi:** `scripts/test_aarch64.sh` (needs `qemu-user-static`).

Regenerate the golden test data (stub models, expected values) with
`uv run --extra training python -m scripts.make_golden`.
