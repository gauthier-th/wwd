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

## Tests

```sh
uv run pytest                 # core, server (tiny base model) + Python runtime (golden parity)
uv run ruff check . && uv run lint-imports
pnpm --filter @wwd/runtime test        # JS runtime, onnxruntime-web wasm under Node
pnpm --filter @wwd/web test:e2e        # Chromium + Firefox: Worker + AudioWorklet
```

## Run a model

```sh
python -m runtime.listen model.onnx --vad runtime/assets/silero_vad_16k_op15.onnx
pnpm --filter @wwd/web dev    # then open /harness.html
```

Regenerate the golden test data (stub models, expected values) with
`uv run --extra training python -m scripts.make_golden`.
