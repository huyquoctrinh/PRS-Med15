#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 STAGE GPUS  (example: $0 A 3,5)" >&2
  exit 2
fi

stage="${1,,}"
gpu_list="$2"
if [[ "$stage" != "a" && "$stage" != "b" && "$stage" != "b2" && "$stage" != "b2e" ]]; then
  echo "STAGE must be A, B, B2 or B2E" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable; refusing training launch" >&2
  exit 1
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
current_user="$(id -un)"
IFS=',' read -r -a requested_gpus <<< "$gpu_list"
for gpu in "${requested_gpus[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
    echo "Invalid GPU index: $gpu" >&2
    exit 2
  fi
  if ! nvidia-smi -i "$gpu" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
    echo "GPU $gpu does not exist" >&2
    exit 1
  fi
  while read -r pid; do
    pid="${pid//[[:space:]]/}"
    [[ -z "$pid" || "$pid" == "[N/A]" ]] && continue
    owner="$(ps -o user= -p "$pid" | xargs || true)"
    if [[ -n "$owner" && "$owner" != "$current_user" ]]; then
      echo "GPU $gpu has process $pid owned by $owner; refusing launch" >&2
      exit 1
    fi
  done < <(
    nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true
  )
done

python_root="${PRSMED15_PYTHON_ROOT:-/home/gdp2988/miniconda3/envs/huy/bin}"
torchrun_bin="$python_root/torchrun"
if [[ ! -x "$torchrun_bin" ]]; then
  echo "torchrun is unavailable at $torchrun_bin" >&2
  exit 1
fi

run_name="full_stage_${stage}"
config="configs/train/${run_name}.yaml"
run_dir="artifacts/runs/${run_name}"
mkdir -p "$run_dir"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_path="$run_dir/train_${timestamp}.log"
pid_path="$run_dir/train.pid"
extra_args=()
if [[ "$stage" == b* ]]; then
  # Stage B starts from a stage A checkpoint: STAGE_A_CHECKPOINT=path, default = latest
  # checkpoint-* under artifacts/runs/full_stage_a; STAGE_A_CHECKPOINT=none trains B from scratch.
  init_ckpt="${STAGE_A_CHECKPOINT:-}"
  if [[ -z "$init_ckpt" ]]; then
    init_ckpt="$(ls -d artifacts/runs/full_stage_a/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
  fi
  if [[ "$init_ckpt" == "none" ]]; then
    echo "Stage B without stage A initialisation"
  elif [[ -n "$init_ckpt" && -d "$init_ckpt" ]]; then
    echo "Stage B initialised from $init_ckpt"
    extra_args=(--init_from "$init_ckpt")
  else
    echo "No stage A checkpoint found (set STAGE_A_CHECKPOINT=none to train B from scratch)" >&2
    exit 1
  fi
fi

# RESUME_CHECKPOINT=path continues a run (optimizer, scheduler, step, data position).
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT/trainer_state.json" ]]; then
    echo "RESUME_CHECKPOINT has no trainer_state.json: $RESUME_CHECKPOINT" >&2
    exit 1
  fi
  echo "Resuming from $RESUME_CHECKPOINT"
  extra_args+=(--set "train.resume=$RESUME_CHECKPOINT")
fi

nohup env \
  CUDA_VISIBLE_DEVICES="$gpu_list" \
  PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}" \
  PRSMED15_ALLOW_HEAVY=1 \
  HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$torchrun_bin" \
  --nproc_per_node="${#requested_gpus[@]}" \
  -m prs_med15.cli train \
  --config "$config" \
  "${extra_args[@]}" \
  >"$log_path" 2>&1 &
pid="$!"
echo "$pid" > "$pid_path"
echo "Launched stage ${stage^^}: pid=$pid log=$log_path"
