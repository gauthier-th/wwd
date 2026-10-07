#!/usr/bin/env bash
# API for the end-to-end tests: a tiny base model (server/tests/tiny_base.py), a fresh database
# and the fine-tune worker in-process. Needs the Python environment (uv sync --extra server, or
# --extra training on the GPU box).
set -euo pipefail
cd "$(dirname "$0")/../.."
E2E=web/.e2e
unset VIRTUAL_ENV
uv run --no-sync python -m server.tests.tiny_base "$E2E/models" --fixtures "$E2E/fixtures"
rm -rf "$E2E/server"
export WWD_SERVER_DATA="$E2E/server" WWD_MODELS="$E2E/models" WWD_BASE_VERSION=tiny \
  WWD_WORKER_THREAD=1 WWD_WEB_DIST= WWD_VERIFY_WORKERS=2 \
  WWD_FINETUNE='{"augment_per_clip": 16, "partial_per_clip": 4, "hard_negatives": 50, "random_negatives": 200}'
exec uv run --no-sync uvicorn server.api:app --host 127.0.0.1 --port "${E2E_API_PORT:-8766}"
