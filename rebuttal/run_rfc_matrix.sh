#!/bin/bash
# Pairwise RFC (correction vector field) over a task set, on one GPU.
#
#   bash rebuttal/run_rfc_matrix.sh <gpu_id> <tag> <config_name> <ckpt_root> <task...>
#
# e.g. privileged libero_object task-experts straight out of the HF cache:
#   bash rebuttal/run_rfc_matrix.sh 3 obj_priv eval_lora_config_object \
#        /workspace/hf/hub/models--AvidJoyce--icml-2025-checkpoints/snapshots/<rev>/task-expert/libero-object 1 6 7 8 9
#
# analyze_correction_vector_field.py uses a plain torch.device("cuda") and does
# not go through the RLinf scheduler, so CUDA_VISIBLE_DEVICES does pin it here
# (unlike the training/eval entrypoints, where it is ignored).
#
# Probe-state protocol is the one the paper uses: base_rollout, 5 episodes per
# task with the frozen base policy, states unioned across the pair.
set -uo pipefail

GPU=$1; TAG=$2; CONFIG=$3; CKPT_ROOT=$4; shift 4
TASKS=("$@")

# Residual observation space. Must match what the checkpoints were trained with:
# an rgb checkpoint has an 846-wide fc1_A and will fail to load in privileged mode.
OBS_MODE=${OBS_MODE:-privileged}
# Probe episodes per task; the paper uses 5.
MAX_DEMOS=${MAX_DEMOS:-5}

REPO=/workspace/RLinf
ANALYSIS_DIR=$REPO/examples/embodiment/eval/correction_field_analysis
OUT=$REPO/results/rfc_${TAG}
mkdir -p "$OUT"

export EMBODIED_PATH=$REPO/examples/embodiment
export PYTHONPATH=$REPO
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=$GPU

cd "$ANALYSIS_DIR"

for ((a=0; a<${#TASKS[@]}; a++)); do
  for ((b=a+1; b<${#TASKS[@]}; b++)); do
    i=${TASKS[$a]}; j=${TASKS[$b]}
    log="$OUT/pair_${i}_${j}.log"
    echo "=== pair ($i, $j) -> $log ==="
    /opt/venv/openvla-oft/bin/python analyze_correction_vector_field.py \
      --config "configs/${CONFIG}.yaml" \
      --task_i "$i" --task_j "$j" \
      --checkpoint_i "${CKPT_ROOT}/task${i}" \
      --checkpoint_j "${CKPT_ROOT}/task${j}" \
      --state_method base_rollout \
      --max_demos "$MAX_DEMOS" \
      --obs_mode "$OBS_MODE" \
      > "$log" 2>&1
    echo "  exit=$? $(grep -aoE 'Overall Direction Consistency: [-0-9.]+' "$log" | tail -1)"
  done
done

echo "done -> $OUT"
