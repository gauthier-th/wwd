#!/usr/bin/env bash
# The static app for the end-to-end tests: the tiny base model exported as web assets
# (scripts/make_web_fixtures.py), built into dist/ and served by `vite preview`. No API: the
# app records, trains and tests in the browser. Needs the Python environment for the fixtures.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
uv run --no-sync python -m scripts.make_web_fixtures
cd web
WWD_PUBLIC=.e2e/public pnpm build
exec pnpm preview --port 4173 --strictPort
