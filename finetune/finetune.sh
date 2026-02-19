#!/bin/bash
set -e

# ===========================
# Config
# ===========================

BATCH_SIZE=2
LR=2e-5
NUM_EPOCHS=5
MAX_LENGTH=384

MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_DIR="/playpen-shared/haochenz/UMU-Bench/full_data/train-00000-of-00001.parquet"
effective_bs=BATCH_SIZE * 8
SAVE_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset_lr${LR}_bs${effective_bs}"



mkdir -p /playpen-shared/haochenz/hf_cache

export HF_HOME=/playpen-shared/haochenz/hf_cache
export TRANSFORMERS_CACHE=/playpen-shared/haochenz/hf_cache
export HF_DATASETS_CACHE=/playpen-shared/haochenz/hf_cache
export WANDB_DIR=/playpen-shared/haochenz/wandb
export WANDB_CACHE_DIR=/playpen-shared/haochenz/wandb_cache

export CUDA_VISIBLE_DEVICES=4

# ===========================
# Run
# ===========================
echo "Running finetuning..."
echo "MODEL_ID = ${MODEL_ID}"
echo "DATA_DIR = ${DATA_DIR}"
echo "SAVE_DIR = ${SAVE_DIR}"

accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --gradient_accumulation_steps 8 \
  finetune.py \
  --model_id ${MODEL_ID} \
  --save_dir ${SAVE_DIR} \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --max_length ${MAX_LENGTH}

echo "Done."