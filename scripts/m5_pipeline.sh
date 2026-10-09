#!/usr/bin/env bash
# M5 end to end: XLS-R teacher -> teacher cache -> distilled student -> base model -> evaluations.
# Safe to run detached:
#   setsid nohup scripts/m5_pipeline.sh v1 >> data/m5.log 2>&1 < /dev/null &
# Every step is resumable, so re-running after an interruption continues where it stopped.
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="${1:-v1}"
TEACHER="${TEACHER:-teacher}"
START_AT="${START_AT:-teacher}" # skip earlier steps: teacher, teacher_eval, cache, student, ...

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

run teacher training.train_teacher --name "$TEACHER" --layers 18
run teacher_eval training.evaluate "models/runs/$TEACHER/best.pt"
run cache training.cache_teacher "models/runs/$TEACHER"
run student training.train_student --name "$VERSION" --teacher "models/runs/$TEACHER"
run export training.export "models/runs/$VERSION/best.pt" --version "$VERSION"
run evaluate training.evaluate "models/base/$VERSION/encoder.pt"
run e2e training.eval_e2e "models/base/$VERSION"
run shipped training.eval_shipped "models/base/$VERSION"
echo "=== $(date '+%F %T') M5 pipeline finished: models/base/$VERSION"
