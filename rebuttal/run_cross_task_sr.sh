#!/bin/bash
# Cross-task success-rate matrix: every policy evaluated on every task.
#
#   bash rebuttal/run_cross_task_sr.sh <gpu_id> <tag> <config_name> <policy_spec...> -- <task...>
#
# policy_spec is name=path. Example:
#   bash rebuttal/run_cross_task_sr.sh 6 obj_priv libero_object_task1_lora_residual_sac_openvlaoft \
#       P1=/workspace/hf/.../task-expert/libero-object/task1 ... -- 1 6 7 8 9
#
# batch_eval_cross_task.sh is not used here: its SKIP_TASKS list is hard-coded
# (line 76) and its per-task policy slots are positional, which makes it easy to
# silently evaluate the wrong set.
set -uo pipefail

GPU=$1; TAG=$2; CONFIG=$3; shift 3
POLICIES=()
while [ "${1:-}" != "--" ] && [ $# -gt 0 ]; do POLICIES+=("$1"); shift; done
shift  # drop the --
TASKS=("$@")

REPO=/workspace/RLinf
OUT=$REPO/results/sr_${TAG}
mkdir -p "$OUT"

for spec in "${POLICIES[@]}"; do
  name=${spec%%=*}; path=${spec#*=}
  for task in "${TASKS[@]}"; do
    tag="${TAG}_${name}_on_t${task}"
    log="$OUT/${name}_on_t${task}.log"
    if grep -aq "eval/success_once" "$log" 2>/dev/null; then
      echo "skip $name on task $task (already done)"; continue
    fi
    echo "=== $name on task $task ==="
    bash "$REPO/rebuttal/run_eval.sh" "$CONFIG" "$GPU" "$tag" \
      runner.eval_policy_path="$path" \
      env.eval.specific_reset_id="$task" \
      > "$log" 2>&1
    echo "  SR=$(grep -aoE "'eval/success_once': array\([0-9.]+" "$log" | tail -1 | grep -oE '[0-9.]+$')"
  done
done

echo "done -> $OUT"
