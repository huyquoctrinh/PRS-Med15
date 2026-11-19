#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing GPU smoke test" >&2
  exit 1
fi

current_user="$(id -un)"
selected_gpu=""
while IFS=',' read -r raw_index raw_uuid; do
  gpu_index="${raw_index//[[:space:]]/}"
  gpu_uuid="${raw_uuid//[[:space:]]/}"
  foreign=0
  while read -r pid; do
    pid="${pid//[[:space:]]/}"
    [[ -z "$pid" || "$pid" == "[N/A]" ]] && continue
    owner="$(ps -o user= -p "$pid" | xargs || true)"
    if [[ -n "$owner" && "$owner" != "$current_user" ]]; then
      foreign=1
      break
    fi
  done < <(nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)
  if [[ "$foreign" -eq 0 ]]; then
    selected_gpu="$gpu_index"
    break
  fi
done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

if [[ -z "$selected_gpu" ]]; then
  echo "No GPU is free of other users' compute processes; refusing to run" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="$selected_gpu"
export PRSMED15_ALLOW_HEAVY=1
python_bin="${PRSMED15_PYTHON:-/home/gdp2988/miniconda3/envs/huy/bin/python}"

"$python_bin" scripts/smoke_gpu_checks.py parity
"$python_bin" scripts/smoke_gpu_checks.py forward

"$python_bin" -m prs_med15.cli train \
  --config configs/train/stage_a.yaml \
  --limit 8 \
  --results_dir artifacts/smoke_gpu/stage_a \
  --set train.max_steps=200 \
  --set train.checkpoint_every=200

"$python_bin" -m prs_med15.cli train \
  --config configs/train/stage_b.yaml \
  --limit 8 \
  --results_dir artifacts/smoke_gpu/stage_b \
  --set train.max_steps=200 \
  --set train.checkpoint_every=200

"$python_bin" scripts/smoke_gpu_checks.py dialogue
