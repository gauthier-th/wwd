#!/usr/bin/env bash
# Publish the base model's browser files (web/public/base/, from scripts/export_web_assets.py)
# as a GitHub release asset, base.tar.gz: they're too big for git. The website build downloads
# the release named by BASE_RELEASE in .github/workflows/ci.yml.
#
#   scripts/release_base.sh base-v1
set -euo pipefail
tag=${1:?usage: scripts/release_base.sh <tag, e.g. base-v1>}
cd "$(dirname "$0")/.."
test -f web/public/base/index.json || { echo "web/public/base/ is empty: run scripts.export_web_assets first" >&2; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
tar -czf "$tmp/base.tar.gz" -C web/public base
gh release create "$tag" "$tmp/base.tar.gz" --latest=false --title "Base model files ($tag)" --notes \
  "The base model's files for the web app (\`web/public/base/\`). Built from datasets under CC-BY 4.0 and other licenses: see the README's credits."
echo "Now set BASE_RELEASE: $tag in .github/workflows/ci.yml."
