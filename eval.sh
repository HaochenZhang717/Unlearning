#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=6

FORGET_RATIOS=(5)
METHODS=("NPO" "GA" "PO")

OUT_PATH="/playpen-shared/haochenz/UMU-Bench-result/results"
DATA_SPLIT_DIR="/playpen-shared/haochenz/UMU-Bench"
MODEL_ID="llava-hf/llava-1.5-7b-hf"

mkdir -p ${OUT_PATH}

for FORGET_RATIO in "${FORGET_RATIOS[@]}"; do
  for METHOD in "${METHODS[@]}"; do

    CACHE_PATH="/playpen-shared/haochenz/UMU-Bench-result/ckpts/llava_${METHOD}_forget${FORGET_RATIO}"
    OUTPUT_FILE="${METHOD}_forget${FORGET_RATIO}.json"

    echo "========================================="
    echo "Running eval: METHOD=${METHOD}, FORGET_RATIO=${FORGET_RATIO}"
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
done



export CUDA_VISIBLE_DEVICES=6
FORGET_RATIOS=(5)
METHODS=("KL")

OUT_PATH="/playpen-shared/haochenz/UMU-Bench-result/results"
DATA_SPLIT_DIR="/playpen-shared/haochenz/UMU-Bench"
MODEL_ID="llava-hf/llava-1.5-7b-hf"

mkdir -p ${OUT_PATH}

for FORGET_RATIO in "${FORGET_RATIOS[@]}"; do
  for METHOD in "${METHODS[@]}"; do

    CACHE_PATH="/playpen-shared/haochenz/UMU-Bench-result/ckpts/llava_${METHOD}_forget${FORGET_RATIO}_gamma1.0_alpha1.0"
    OUTPUT_FILE="${METHOD}_forget${FORGET_RATIO}.json"

    echo "========================================="
    echo "Running eval: METHOD=${METHOD}, FORGET_RATIO=${FORGET_RATIO}"
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
done