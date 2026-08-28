#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "usage: $0 MODEL_DIR OUTPUT_DIR REPLICATE_ID [BUDGET_K ...]" >&2
    exit 2
fi
: "${RLINF_REVISION:?set RLINF_REVISION to the exact nested revision}"
: "${OUTER_REVISION:?set OUTER_REVISION to the exact outer revision}"
: "${CONTAINER_IMAGE:?set CONTAINER_IMAGE to the running image id/tag}"

model_dir=$1
output_dir=$2
replicate_id=$3
shift 3
if [ "$#" -eq 0 ]; then
    budget_ks=(1 2 3)
else
    budget_ks=("$@")
fi
mkdir -p "${output_dir}"

for budget_k in "${budget_ks[@]}"; do
    output="${output_dir}/k${budget_k}.jsonl"
    if [ -e "${output}" ]; then
        echo "refusing to overwrite ${output}" >&2
        exit 2
    fi
    python -m rlinf.envs.behavior.textworld_rollout \
        --model "${model_dir}" \
        --activities /data/behavior-data/text_detect_scan_v5/rl/val_s2.jsonl \
        --obs-mode detect_scope_scan \
        --budget-k "${budget_k}" \
        --replicate-id "${replicate_id}" \
        --seed 20260825 \
        --resume \
        --out "${output}"
    # Policy parse/interface failures remain scored outcomes; only incomplete
    # provenance aborts the formal curve launcher.
    python -m rlinf.envs.behavior.audit_text_rollouts "${output}" \
        --min-tool-call-rate 0 \
        --require-provenance
done
