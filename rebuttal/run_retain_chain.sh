#!/bin/bash
# RETAIN continual chain (paper Eq. 4) over libero_object, evaluated with the
# authors' own LIBERO script so the continuous L1 action head is used.
#
#   theta~_0 = SFT(task1)                       # chain start
#   theta~_n = (1-alpha)*theta~_{n-1} + alpha*SFT(task_n)   for n in 6 7 8 9
#
# Each stage is evaluated on all five object tasks (run_libero_eval.py hardcodes
# task_ids=[1,6,7,8,9]), then the previous stage's weights are deleted -- peak
# disk is two checkpoints (~30 GB) rather than five.
#
#   bash rebuttal/run_retain_chain.sh <alpha> <trials_per_task> <gpu_in_container>
set -uo pipefail

ALPHA=${1:-0.5}
TRIALS=${2:-10}
GPU=${3:-0}

REPO=/workspace/RLinf
CKPTS=$REPO/libero-object-checkpoints
BL=$REPO/rebuttal/baselines/BL-openvla-oft
WORK=$REPO/results/retain_a${ALPHA}
PY=/opt/venv/openvla-oft/bin/python

mkdir -p "$WORK"
export CUDA_VISIBLE_DEVICES=$GPU
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

prev=$CKPTS/task1
for n in 6 7 8 9; do
  out=$WORK/stage$n
  echo "=== merging task$n into chain (alpha=$ALPHA) -> $out ==="
  if [ ! -f "$out/model.safetensors.index.json" ]; then
    PYTHONPATH=$REPO $PY $REPO/rebuttal/retain_merge.py \
      --a "$prev" --b "$CKPTS/task$n" --alpha "$ALPHA" --out "$out" || exit 1
  else
    echo "  (already merged, reusing)"
  fi

  echo "=== evaluating stage$n on tasks 1 6 7 8 9 (T=$TRIALS) ==="
  ( cd "$BL" && PYTHONPATH=$BL $PY experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "$out" \
      --task_suite_name libero_object \
      --num_trials_per_task "$TRIALS" \
      --local_log_dir "$WORK/evallog_stage$n" ) \
    > "$WORK/eval_stage$n.log" 2>&1
  echo "  exit=$?"

  # Free the previous stage; task1 is an input asset and is never removed.
  if [ "$prev" != "$CKPTS/task1" ]; then
    rm -rf "$prev"
    echo "  freed $prev"
  fi
  prev=$out
done

echo "done -> $WORK"
