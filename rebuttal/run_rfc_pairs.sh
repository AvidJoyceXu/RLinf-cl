#!/bin/bash
# Pairwise RFC for an explicit list of pairs, on one GPU.
#
#   bash rebuttal/run_rfc_pairs.sh <gpu_id> <tag> <config_name> <ckpt_root> <i:j> [<i:j> ...]
#
# analyze_correction_vector_field.py uses a plain torch.device("cuda") and does
# not go through the RLinf scheduler, so CUDA_VISIBLE_DEVICES pins it here.
# Probe states follow the paper's protocol: base_rollout, 5 episodes per task,
# unioned across the pair (N = 2400 when no episode terminates early).
set -uo pipefail

GPU=$1; TAG=$2; CONFIG=$3; CKPT_ROOT=$4; shift 4
# Probe episodes per task. The paper uses 5, which turns out to give a
# cross-repeat std of ~0.32 on mean_dir -- as wide as the whole signal range --
# so this is exposed for variance-reduction runs.
MAX_DEMOS=${MAX_DEMOS:-5}
# Residual observation space; must match what the checkpoints were trained with.
OBS_MODE=${OBS_MODE:-privileged}
# Trim every task to exactly this many probe states, so N is identical across pairs.
NUM_PROBE_STATES=${NUM_PROBE_STATES:-}

REPO=/workspace/RLinf
ANALYSIS_DIR=$REPO/examples/embodiment/eval/correction_field_analysis
OUT=$REPO/results/rfc_${TAG}
mkdir -p "$OUT"

export EMBODIED_PATH=$REPO/examples/embodiment
export PYTHONPATH=$REPO
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export CUDA_VISIBLE_DEVICES=$GPU

cd "$ANALYSIS_DIR"

for pair in "$@"; do
  i=${pair%%:*}; j=${pair##*:}
  log="$OUT/pair_${i}_${j}.log"
  echo "=== pair ($i, $j) ==="
  /opt/venv/openvla-oft/bin/python analyze_correction_vector_field.py \
    --config "configs/${CONFIG}.yaml" \
    --task_i "$i" --task_j "$j" \
    --checkpoint_i "${CKPT_ROOT}/task${i}" \
    --checkpoint_j "${CKPT_ROOT}/task${j}" \
    --state_method base_rollout \
    --max_demos "$MAX_DEMOS" \
    --obs_mode "$OBS_MODE" \
    ${NUM_PROBE_STATES:+--num_probe_states "$NUM_PROBE_STATES"} \
    > "$log" 2>&1
  rc=$?
  echo "  exit=$rc  $(grep -aoE 'Overall Direction Consistency: [-0-9.]+ . [0-9.]+' "$log" | tail -1)  $(grep -aoE 'Dangerous States: [0-9/]+ \([0-9.]+%\)' "$log" | tail -1)"
done

echo "done -> $OUT"
