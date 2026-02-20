#!/bin/bash
set -e

# --- Hardware & Base Paths ---
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/playpen/haochenz/hf_cache

OUT_PATH="./results/sweep_results"
DATA_SPLIT_DIR="/playpen/haochenz/UMU-Bench"
MODEL_ID="llava-hf/llava-1.5-7b-hf"
BASE_CHECKPOINT_DIR="/playpen/haochenz/checkpoints/llava-sweep"

mkdir -p ${OUT_PATH}

# --- 💡 Match these to your Training Sweep lists ---
LR_LIST=(1e-4)
LORA_R_LIST=(8 16 32 64)
FORGET_RATIOS=(5) # Keep as list if you want to test multiple forget ratios per model

# --- Nested Loops to match the Training Structure ---
for LR in "${LR_LIST[@]}"; do
  for R in "${LORA_R_LIST[@]}"; do
    for FORGET_RATIO in "${FORGET_RATIOS[@]}"; do

      # 1. Dynamically locate the checkpoint folder created during training
      RUN_NAME="llava-lr-${LR}-r-${R}"
      CACHE_PATH="${BASE_CHECKPOINT_DIR}/${RUN_NAME}"

      # 2. Dynamically name the output file so results don't overwrite each other
      # Format: eval_lr-1e-4_r-8_fr-5.json
      OUTPUT_FILE="eval_lr-${LR}_r-${R}_fr-${FORGET_RATIO}.json"

      # Check if the checkpoint directory actually exists before running eval
      if [ -d "$CACHE_PATH" ]; then
        echo "========================================="
        echo "🚀 Evaluating: ${RUN_NAME}"
        echo "LR: ${LR} | LoRA R: ${R} | Forget Ratio: ${FORGET_RATIO}"
        echo "CACHE_PATH: ${CACHE_PATH}"
        echo "========================================="

        python custom_eval.py \
          --model_id "${MODEL_ID}" \
          --cache_path "${CACHE_PATH}" \
          --forget_ratio "${FORGET_RATIO}" \
          --data_split_dir "${DATA_SPLIT_DIR}" \
          --output_path "${OUT_PATH}" \
          --output_file "${OUTPUT_FILE}"

        echo "✅ Finished Evaluation for ${RUN_NAME}"
      else
        echo "⚠️ Skipping: ${CACHE_PATH} not found."
      fi

    done
  done
done

echo "All sweep evaluations completed. Results are in ${OUT_PATH}"