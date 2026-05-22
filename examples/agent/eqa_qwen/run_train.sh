#!/bin/bash
# EQA-Qwen launcher. Inside rlinf-eqa container:
#     bash examples/agent/eqa_qwen/run_train.sh                       # full train
#     bash examples/agent/eqa_qwen/run_train.sh train_qwen25vl_7b_scripted   # scripted-policy smoke (M2)
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
# H20-3e: NVLink SHARP (NVLS) causes CUDA error 1 on barrier() in containerized env.
export NCCL_NVLS_ENABLE=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname $(dirname "$CONFIG_PATH")))
MEGATRON_PATH=${MEGATRON_PATH:-/opt/Megatron-LM}

# /workspace is the parent of the bind-mounted spatialcode/embodied so the
# launcher can `import spatialcode.embodied.*` (see INSTALL.md §1.4).
SPATIALCODE_PARENT=${SPATIALCODE_PARENT:-/workspace}

export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:${SPATIALCODE_PARENT}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="train_qwen25vl_7b"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/agent/eqa_qwen/train.py \
    --config-path ${CONFIG_PATH}/config/ \
    --config-name $CONFIG_NAME
