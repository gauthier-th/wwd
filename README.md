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

## Web app

```sh
pnpm --filter @wwd/web dev                         # http://localhost:5173, /api proxied to :8000
```

Or build it once and let the API serve it, so the app and the API share one origin:

```sh
pnpm --filter @wwd/web build
WWD_WORKER_THREAD=1 uv run --extra server uvicorn server.api:app   # app + API + worker on :8000
```

Browsers only open the microphone on `https://` or `localhost`. To try it on a phone, put the
server behind HTTPS, for example `cloudflared tunnel --url http://localhost:8000` or a mkcert
certificate trusted by the phone.

## Tests

```sh
uv run pytest                 # core, server (tiny base model) + Python runtime (golden parity)
uv run ruff check . && uv run lint-imports
pnpm --filter @wwd/runtime test        # JS runtime, onnxruntime-web wasm under Node
pnpm --filter @wwd/web test            # app unit tests
pnpm --filter @wwd/web test:e2e        # Chromium, Firefox, WebKit (iPhone): the app against a real
                                       # API on a tiny base model, plus runtime parity and the mic
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
