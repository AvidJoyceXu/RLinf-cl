#!/bin/bash
# BEHAVIOR-Qwen online-GRPO launcher. Run inside the unified container
# (rlinf-behavior-rl), which carries both venvs.
#
#   # 1. once: build the RL prompt dataset (needs the env venv for the loaders)
#   with-env python -m rlinf.envs.behavior.build_rl_dataset --out /data/behavior-data/rl
#
#   # 2. bring up the env-server pool and keep it running in another shell
#   scripts/launch_behavior_env_servers.sh \
#       --activities picking_up_trash,bringing_in_wood --replicas 4
#
#   # 3. train (this script) — reads the servers.yaml the launcher printed
#   bash examples/agent/behavior_qwen/run_train.sh behavior_grpo_qwen25_7b_1gpu
#
# The trainer half must run under `with-trainer`; running it in the env venv
# fails on `import sglang`, and running the env servers in the trainer venv fails
# on `import omnigibson`. That is the venv split, and it is why this script pins
# the interpreter rather than trusting whatever the shell happens to have active.
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
# This host's driver has a reproduced NVLS regression: multi-rank actor-to-rollout
# synchronization can surface as CUDA illegal memory access. Ray captures the
# setting when it starts, so establish the safe default before train.py creates it.
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# The config logs to tensorboard AND wandb. Without a key, `wandb.init` would abort
# the run at startup over a *logging* credential, so fall back to offline mode: the
# run still records everything locally and `wandb sync <dir>` uploads it later.
if [ -z "$WANDB_API_KEY" ] && [ ! -f "$HOME/.netrc" ]; then
    export WANDB_MODE=${WANDB_MODE:-offline}
    echo "NOTE: no WANDB_API_KEY / ~/.netrc -> WANDB_MODE=$WANDB_MODE" >&2
    echo "      to log online: export WANDB_API_KEY=... (or run 'wandb login')" >&2
fi

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname $(dirname "$CONFIG_PATH")))
MEGATRON_PATH=${MEGATRON_PATH:-/opt/venv/reason/Megatron-LM}
SPATIALCODE_PARENT=${SPATIALCODE_PARENT:-/workspace}

export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:${SPATIALCODE_PARENT}:$PYTHONPATH

CONFIG_NAME=${1:-behavior_grpo_qwen25_7b}
shift $(( $# > 0 ? 1 : 0 ))

# No env-server pool to start any more: the tool worker holds OmniGibson in process,
# so there is no servers.yaml to check for. The first rollout pays the Kit boot
# (~3-4 min with cameras off) inside the worker instead.

python ${REPO_PATH}/examples/agent/behavior_qwen/train.py \
    --config-path ${CONFIG_PATH}/config/ \
    --config-name $CONFIG_NAME \
    "$@"
