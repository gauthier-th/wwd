#!/usr/bin/env bash
# The device kit (PLAN.md M4): the runtime package with its golden tests and its guide, plus a
# downloaded bundle if given.
#   scripts/pack_runtime.sh [bundle.zip]   ->  dist/wwd-runtime.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
kit="$stage/wwd-runtime"
mkdir -p "$kit"
tar cf - --exclude=js --exclude=__pycache__ --exclude=.pytest_cache runtime | tar xf - -C "$kit"
mv "$kit/runtime/README.md" "$kit/README.md"
if [[ $# -ge 1 ]]; then
  mkdir -p "$kit/bundle"
  unzip -q -j "$1" -d "$kit/bundle"  # the app's zip holds wake-word/<files>
fi
mkdir -p dist
tar czf dist/wwd-runtime.tar.gz -C "$stage" wwd-runtime
echo "dist/wwd-runtime.tar.gz ($(du -h dist/wwd-runtime.tar.gz | cut -f1)):"
tar tzf dist/wwd-runtime.tar.gz | grep -vE "/$" | sed 's|^wwd-runtime/||' | awk -F/ '{print $1"/"$2}' | sort -u | head -30
