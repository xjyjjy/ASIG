#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"

usage() {
    cat <<'EOF'
Usage:
  bash make_arg_map.sh dataset [W1 W2 ...]
  bash make_arg_map.sh panorama [extra args...]

Examples:
  bash make_arg_map.sh dataset
  bash make_arg_map.sh dataset 1024 896 768 640 512 112
  bash make_arg_map.sh panorama
EOF
}

mode="${1:-dataset}"
if [[ $# -gt 0 ]]; then
    shift
fi

case "${mode}" in
    dataset)
        if [[ $# -eq 0 ]]; then
            widths=(1024 896 768 640 512 112)
        else
            widths=("$@")
        fi
        python "${ROOT_DIR}/models/make_arg_map_dataset.py" --W "${widths[@]}"
        ;;
    panorama)
        if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
            python "${ROOT_DIR}/models/make_arg_map_test_dataset.py" "$@"
        else
            CUDA_VISIBLE_DEVICES=1 python "${ROOT_DIR}/models/make_arg_map_test_dataset.py" "$@"
        fi
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 1
        ;;
esac
