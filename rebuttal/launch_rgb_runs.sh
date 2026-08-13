#!/bin/bash
# Launch the 5 libero_object RGB residual training runs, one docker container each.
#
# One container per run is not cosmetic: RLinf's scheduler attaches to an
# existing Ray cluster by scanning /proc for a process carrying --gcs-address
# (ray/_private/services.py:_find_address_from_flag). Runs sharing a PID
# namespace therefore share one Ray cluster, and killing any single run tears
# down all of them via rlinf/scheduler/cluster/cluster.py:signal_handler.
# Separate containers give each run its own PID namespace and its own cluster.
#
# GPU count is limited at the device level (--gpus device=...) rather than via
# cluster.component_placement, so the scheduler's hardware probe sees exactly
# the intended cards and the committed config needs no placement edit.
# task1 gets 2 GPUs because its config uses total_num_envs=64 (tasks 6-9 use
# 32); this keeps envs-per-GPU equal across runs without touching any
# hyperparameter.
set -euo pipefail

REPO_HOST=/mnt/another/data2/joycexu/RLinf-cl
HF_HOST=/mnt/another/data2/joycexu/.cache/huggingface
IMAGE=rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

# task -> GPU device list
declare -A GPUS=( [1]="0,1" [6]="2" [7]="3" [8]="4" [9]="5" )

mkdir -p "$REPO_HOST/rebuttal/runlogs"

for t in 1 6 7 8 9; do
  name="rlinf-t${t}"
  docker rm -f "$name" >/dev/null 2>&1 || true
  docker run -d \
    --gpus "\"device=${GPUS[$t]}\"" \
    --shm-size 64g \
    --name "$name" \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    -e HF_HOME=/workspace/hf \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -v "$REPO_HOST":/workspace/RLinf \
    -v "$HF_HOST":/workspace/hf \
    "$IMAGE" sleep infinity >/dev/null

  docker exec -d -w /workspace/RLinf "$name" bash -lc "
    source switch_env openvla-oft
    bash examples/embodiment/run_embodiment.sh _run_rgb_t${t} \
      > /workspace/RLinf/rebuttal/runlogs/t${t}.log 2>&1"

  echo "launched ${name} on GPU(s) ${GPUS[$t]}"
done

docker ps --format '{{.Names}}\t{{.Status}}' | grep rlinf
