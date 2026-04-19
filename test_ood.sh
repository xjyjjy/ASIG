#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"

MODEL_PATH="${MODEL_PATH:-checkpoints/inference/model.pth}"
PROMPT_ROOT="${PROMPT_ROOT:-dataset/360/ood_prompt}"
SAVE_ROOT="${SAVE_ROOT:-save}"
CFG_SCALE="${CFG_SCALE:-10}"
SEED="${SEED:-14}"

for category in City Weather Nature Fantasy Mixed; do
  lower_name="$(printf '%s' "${category}" | tr '[:upper:]' '[:lower:]')"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python gen_samples_ood.py \
    --model "${MODEL_PATH}" \
    --batch-size 1 \
    -o "${SAVE_ROOT}" \
    --text-data "${PROMPT_ROOT}/${category}" \
    --output-sizes "ood/${lower_name}" \
    --bf16 \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}"
done
