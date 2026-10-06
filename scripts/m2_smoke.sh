#!/usr/bin/env bash
# M2 done-check (PLAN.md §12): a curl-driven run against a real base model whose bundle must
# pass the dual-runtime parity checks.
#   scripts/m2_smoke.sh [lang] [base-version]
# 1. writes a user's recordings for a held-out word (scripts.m2_inputs)
# 2. starts the API and the worker on a scratch data dir
# 3. drives the API with curl: create, upload, train, follow the SSE stream, download
# 4. runs the bundle through both runtimes: Python (scripts.check_bundle) and onnxruntime-web
#    wasm under Node (runtime/js/test/bundle.test.ts)
set -euo pipefail
cd "$(dirname "$0")/.."
LANG_CODE="${1:-fr}"
BASE="${2:-v0.1}"
OUT="data/smoke/$LANG_CODE"
PORT="${PORT:-8765}"
API="http://127.0.0.1:$PORT/api"
py() { env -u VIRTUAL_ENV uv run --extra training python "$@"; }

rm -rf "$OUT"
mkdir -p "$OUT"
py -m scripts.m2_inputs --lang "$LANG_CODE" --base "$BASE" --out "$OUT/inputs"

if curl -sf "$API/health" > /dev/null; then
  echo "something already listens on port $PORT: stop it or set PORT" >&2
  exit 1
fi
export WWD_SERVER_DATA="$OUT/server" WWD_BASE_VERSION="$BASE"
# Each server in its own process group, so the trap stops uv *and* its python child.
setsid env -u VIRTUAL_ENV uv run --extra training python -m uvicorn server.api:app --port "$PORT" \
  > "$OUT/api.log" 2>&1 &
API_PID=$!
setsid env -u VIRTUAL_ENV uv run --extra training python -m server.worker > "$OUT/worker.log" 2>&1 &
WORKER_PID=$!
trap 'kill -- -$API_PID -$WORKER_PID 2>/dev/null || true' EXIT
for _ in $(seq 120); do curl -sf "$API/health" > /dev/null && break; sleep 0.5; done
kill -0 $API_PID $WORKER_PID  # both still up

WORD=$(jq -r .word "$OUT/inputs/meta.json")
echo "=== wake word: $WORD ($LANG_CODE), base $BASE"
WID=$(curl -sf -X POST "$API/wakewords" -H 'content-type: application/json' \
  -d "$(jq -n --arg n "$WORD" --arg l "$LANG_CODE" '{name: $n, language: $l}')" | jq -r .id)
upload() {
  curl -sf -F "kind=$1" -F "file=@$2;type=audio/wav" "$API/wakewords/$WID/samples"
}
upload positive "$OUT/inputs/take.wav" | jq -c '.samples[] | {accepted, reason, speech_s, similarity: (.similarity * 1000 | round / 1000), outlier}'
upload other_speech "$OUT/inputs/other_speech.wav" | jq -c '.samples[] | {kind, speech_s}'
upload ambience "$OUT/inputs/ambience.wav" | jq -c '.samples[] | {kind, duration_s}'

TID=$(curl -sf -X POST "$API/wakewords/$WID/trainings" -H 'content-type: application/json' \
  -d '{"mode": "quick", "target_fa_per_hour": 0.5}' | jq -r .id)
echo "=== training $TID"
curl -sfN "$API/trainings/$TID/events" | tee "$OUT/events.txt" | sed -un 's/^data: //p' \
  | jq --unbuffered -r 'if .message then "\(.progress * 100 | floor)% \(.stage): \(.message)" else "final: \(.status)" end'
STATUS=$(curl -sf "$API/trainings/$TID" | jq -r .status)
[[ "$STATUS" == done ]] || { echo "training ended $STATUS"; curl -sf "$API/trainings/$TID" | jq .error; exit 1; }

curl -sf -o "$OUT/bundle.zip" "$API/trainings/$TID/artifacts/bundle.zip"
curl -sf -o "$OUT/report.json" "$API/trainings/$TID/artifacts/report.json"
unzip -oq "$OUT/bundle.zip" -d "$OUT"
BUNDLE="$(realpath "$OUT/wake-word")"
jq '{threshold, calibration, cross_validation: (.cross_validation | del(.per_recording)),
     verification, timings_s}' "$OUT/report.json"

echo "=== Python runtime (onnxruntime)"
py -m scripts.check_bundle "$BUNDLE" "$OUT/inputs/test_word.wav" "$OUT/inputs/test_speech.wav" \
  --labels "$OUT/inputs/meta.json"
echo "=== JS runtime (onnxruntime-web wasm)"
WWD_BUNDLE="$BUNDLE" pnpm --filter @wwd/runtime exec vitest run test/bundle.test.ts
echo "=== M2 smoke run passed: $BUNDLE"
