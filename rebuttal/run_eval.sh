#!/bin/bash
# Single-GPU evaluation driver.
#
#   bash rebuttal/run_eval.sh <config_name> <gpu_id> <tag> [hydra overrides...]
#
# Unlike examples/embodiment/eval_single_task.sh this passes arbitrary hydra
# overrides through, which is needed for res_scale=0 (frozen-base baseline),
# residual_policy.res_scale sweeps, and per-task specific_reset_id.
#
# It derives a temp config pinned to one GPU (component_placement carries a
# comma in its key, so it cannot be overridden on the hydra command line) and
# switches the metric logger to tensorboard (no wandb credentials on this host).
set -euo pipefail

CONFIG_NAME=$1
GPU_ID=$2
TAG=$3
shift 3

REPO=/workspace/RLinf
CFG_DIR=$REPO/examples/embodiment/config
TMP_CONFIG="_eval_${TAG}"

sed -e "s|^    actor,env,rollout: all\$|    actor,env,rollout: \"${GPU_ID}\"|" \
    -e "s|^    logger_backends: \[\"wandb\"\].*\$|    logger_backends: [\"tensorboard\"]|" \
    "$CFG_DIR/${CONFIG_NAME}.yaml" > "$CFG_DIR/${TMP_CONFIG}.yaml"

RESULTS_DIR=$REPO/results/${TAG}
mkdir -p "$RESULTS_DIR"

export EMBODIED_PATH=$REPO/examples/embodiment
export PYTHONPATH=$REPO
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export NVIDIA_DRIVER_CAPABILITIES=all

/opt/venv/openvla-oft/bin/python "$EMBODIED_PATH/eval_embodied_agent.py" \
    --config-path "$CFG_DIR/" \
    --config-name "$TMP_CONFIG" \
    runner.only_eval=True \
    runner.logger.log_path="$RESULTS_DIR" \
    "$@" 2>&1 | tee "$RESULTS_DIR/eval.log"
