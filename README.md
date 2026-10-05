# wwd: few-shot wake-word trainer

Record a wake word in the browser, fine-tune on the server, and get ONNX files that run
identically on onnxruntime (Python, e.g. a Raspberry Pi) and onnxruntime-web (wasm). See
[PLAN.md](PLAN.md) for the design and [runtime/SPEC.md](runtime/SPEC.md) for the runtime contract.

## Setup

```sh
uv sync --extra training      # CUDA torch (GPU box); use --extra server for CPU-only
pnpm install
```

## Tests

```sh
uv run pytest                 # core + Python runtime (golden parity)
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
