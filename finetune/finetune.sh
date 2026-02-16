#!/bin/bash
set -e

# ===========================
# Config
# ===========================
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_DIR="/playpen-shared/haochenz/UMU-bench/full_data/train.parquet"
SAVE_DIR="/playpen-shared/haochenz/UMU-bench-result/ckpts/finetuned_llava_fullset"

BATCH_SIZE=4
LR=2e-5
NUM_EPOCHS=5
MAX_LENGTH=384

export CUDA_VISIBLE_DEVICES=0

# ===========================
# Run
# ===========================
echo "Running finetuning..."
echo "MODEL_ID = ${MODEL_ID}"
echo "DATA_DIR = ${DATA_DIR}"
echo "SAVE_DIR = ${SAVE_DIR}"

accelerate launch train_finetune.py \
  --model_id ${MODEL_ID} \
  --save_dir ${SAVE_DIR} \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --max_length ${MAX_LENGTH}

echo "Done."