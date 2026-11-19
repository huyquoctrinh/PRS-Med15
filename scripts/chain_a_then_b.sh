#!/usr/bin/env bash
# Wait for the running stage A job (pid in artifacts/runs/full_stage_a/train.pid) to exit, then
# launch stage B from the final stage A checkpoint on the same GPUs (train_full.sh re-checks that
# no other user's process is on them).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
gpus="${1:-3,5}"
final="${2:-artifacts/runs/full_stage_a/checkpoint-3000}"
pid="$(cat artifacts/runs/full_stage_a/train.pid)"
while kill -0 "$pid" 2>/dev/null; do sleep 60; done
echo "$(date) stage A pid $pid exited"
if [[ ! -d "$final" ]]; then
  echo "$(date) final stage A checkpoint $final missing; NOT launching stage B" >&2
  exit 1
fi
sleep 30
STAGE_A_CHECKPOINT="$final" bash scripts/train_full.sh B "$gpus"
