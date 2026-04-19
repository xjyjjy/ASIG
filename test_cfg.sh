#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"

MODEL_PATH="${MODEL_PATH:-checkpoints/inference/model.pth}"
PROMPT_ROOT="${PROMPT_ROOT:-dataset/360/mv/text_test_big}"
OUTPUT_ROOT="${OUTPUT_ROOT:-save}"
OUTPUT_NAME="${OUTPUT_NAME:-test_cfg}"
CFG_SCALE="${CFG_SCALE:-6}"
SEED="${SEED:-14}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python gen_samples.py \
  --model "${MODEL_PATH}" \
  --batch-size 1 \
  -o "${OUTPUT_ROOT}" \
  --text-data "${PROMPT_ROOT}" \
  --output-sizes "${OUTPUT_NAME}" \
  --bf16 \
  --cfg_scale "${CFG_SCALE}" \
  --seed "${SEED}"
