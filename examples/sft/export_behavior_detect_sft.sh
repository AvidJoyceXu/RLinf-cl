#!/bin/bash
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: $0 ACTOR_CHECKPOINT_DIR HF_OUTPUT_DIR" >&2
    exit 2
fi

actor_dir=$1
output_dir=$2
weights_path="${actor_dir}/model_state_dict/full_weights.pt"

if [ ! -f "${weights_path}" ]; then
    echo "missing FSDP full weights: ${weights_path}" >&2
    exit 2
fi
if [ -e "${output_dir}" ]; then
    echo "refusing to overwrite existing HF output: ${output_dir}" >&2
    exit 2
fi

python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf \
    --config-name fsdp_qwen25_convertor \
    "convertor.ckpt_path=${weights_path}" \
    "convertor.save_path=${output_dir}"

# Pin every exported shard/tokenizer/config file. Build outside the output directory
# so the manifest cannot accidentally hash an empty/in-progress copy of itself.
manifest_tmp=$(mktemp)
(
    cd "${output_dir}"
    find . -maxdepth 1 -type f ! -name EXPORT_MANIFEST.sha256 -print0 \
        | sort -z \
        | xargs -0 sha256sum
) > "${manifest_tmp}"
mv "${manifest_tmp}" "${output_dir}/EXPORT_MANIFEST.sha256"
