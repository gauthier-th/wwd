#!/usr/bin/env bash
# The static app for the end-to-end tests: the tiny base model exported as web assets
# (scripts/make_web_fixtures.py), built under /wwd/ like the GitHub Pages site, and served by
# `vite preview`. It builds into .e2e/dist/, so web/dist/ is left alone. Needs the Python
# environment for the fixtures. WWD_E2E_PORT changes the port (default 4173).
set -euo pipefail
cd "$(dirname "$0")/../.."
unset VIRTUAL_ENV
uv run --no-sync python -m scripts.make_web_fixtures
cd web
export WWD_BASE_PATH=/wwd/ # for the build and for preview, which serves under it
WWD_PUBLIC=.e2e/public pnpm build --outDir .e2e/dist
exec pnpm preview --outDir .e2e/dist --port "${WWD_E2E_PORT:-4173}" --strictPort
