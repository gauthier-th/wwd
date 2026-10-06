#!/usr/bin/env bash
# M1 end to end: data -> base model -> evaluations. Safe to run detached:
#   setsid nohup scripts/m1_pipeline.sh v0.1 >> data/m1.log 2>&1 < /dev/null &
# Every step is resumable, so re-running after an interruption continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:-v0.1}"
START_AT="${START_AT:-download}" # skip earlier steps: download, prepare, build_bank, train, ...

started=0
run() {
  local step="$1"
  shift
  [[ "$step" == "$START_AT" ]] && started=1
  if [[ $started == 0 ]]; then
    echo "=== skip $step"
    return
  fi
  echo "=== $(date '+%F %T') $step: $*"
  env -u VIRTUAL_ENV uv run --extra training python -m "$@"
}

# A download may still be running in its own session: wait for it.
if [[ -f data/download.pid ]]; then
  while kill -0 "$(cat data/download.pid)" 2>/dev/null; do sleep 30; done
fi
run download training.download # no-op when complete; retries anything that failed
run prepare training.prepare
run build_bank training.build_bank
run train training.train_student --name "$VERSION"
run export training.export "models/runs/$VERSION/best.pt" --version "$VERSION"
run evaluate training.evaluate "models/base/$VERSION/encoder.pt"
run e2e training.eval_e2e "models/base/$VERSION"
echo "=== $(date '+%F %T') M1 pipeline finished: models/base/$VERSION"
