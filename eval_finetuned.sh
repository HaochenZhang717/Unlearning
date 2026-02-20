#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=7

FORGET_RATIOS=(5)

OUT_PATH="./results"
DATA_SPLIT_DIR="/playpen-shared/haochenz/UMU-Bench"
MODEL_ID="llava-hf/llava-1.5-7b-hf"

mkdir -p ${OUT_PATH}

for FORGET_RATIO in "${FORGET_RATIOS[@]}"; do
#    CACHE_PATH="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset"
    CACHE_PATH="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset_lr2e-5_effbs16"
    OUTPUT_FILE="fintuned_lr2e-5_effbs16.json"

    echo "========================================="
    echo "CACHE_PATH=${CACHE_PATH}"
    echo "OUTPUT_FILE=${OUTPUT_FILE}"
    echo "========================================="

    python eval.py \
      --model_id "${MODEL_ID}" \
      --cache_path "${CACHE_PATH}" \
      --forget_ratio "${FORGET_RATIO}" \
      --data_split_dir "${DATA_SPLIT_DIR}" \
      --output_path "${OUT_PATH}" \
      --output_file "${OUTPUT_FILE}"

done