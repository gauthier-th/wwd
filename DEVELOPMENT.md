# Development

| Folder | What's in it |
|---|---|
| `web/` | The app (React, Vite). `src/local/` is the training that runs in the browser. |
| `runtime/` | The detector in Python (`runtime/`) and TypeScript (`runtime/js/`), held to the same golden tests ([SPEC.md](runtime/SPEC.md)). |
| `core/` | Shared Python: encoder, classifier, augmentation, calibration, ONNX export. |
| `training/` | Builds the multilingual base model, offline on a GPU. |
| `server/` | The original Python training server. The app no longer uses it; the browser port is tested against its `quick.py`. |
| `scripts/` | Pipelines, exports and test fixtures. |

[PLAN.md](PLAN.md) has the design and every measurement.

## Setup

Needs Node 22, pnpm 10 and [uv](https://docs.astral.sh/uv/).

```sh
pnpm install
uv sync --extra server        # CPU PyTorch, for the tests. On a CUDA machine: --extra training
```

## Run the app

The app loads the base model from `web/public/base/`. Download the published one, then start Vite:

```sh
curl -L https://github.com/gauthier-th/wwd/releases/download/base-v1/base.tar.gz | tar -xz -C web/public
pnpm --filter @wwd/web dev
```

## Tests

```sh
uv run pytest && uv run ruff check . && uv run lint-imports
pnpm -r typecheck && pnpm --filter @wwd/runtime test
uv run python -m scripts.make_web_fixtures   # a tiny base model for the web tests
pnpm --filter @wwd/web test                  # the browser training against Python's results
pnpm --filter @wwd/web test:e2e              # the whole app in Chromium, Firefox and WebKit
```

Run `pnpm --filter @wwd/web exec playwright install` once before the end-to-end tests.

## Publish

CI builds and tests every push. On `main`, once everything passes, it publishes the site to GitHub
Pages. Before the first deploy, set **Settings → Pages → Source** to **GitHub Actions**.

The base model's files are too big for git, so they're a release asset that the CI downloads.
After you export a new base model, publish its files and point the CI at them:

```sh
scripts/release_base.sh base-v2      # then set BASE_RELEASE: base-v2 in .github/workflows/ci.yml
```

To host the app elsewhere, build it (`WWD_BASE_PATH=/sub/path/` for a sub-path) and serve
`web/dist/` over HTTPS, which the microphone needs. No special headers are needed.

## Train a base model

This needs a CUDA GPU with 16 GB of memory, about 200 GB of disk and a
[Hugging Face](https://huggingface.co) login (`uvx --from huggingface_hub hf auth login`). Each
pipeline runs for hours and resumes where it stopped.

```sh
uv sync --extra training
scripts/m1_pipeline.sh v0.1     # downloads the datasets, then builds and evaluates base v0.1
scripts/m5_pipeline.sh v1       # XLS-R teacher and distilled student: base v1
uv run --extra training python -m scripts.export_web_assets models/base/v1   # -> web/public/base/
```
