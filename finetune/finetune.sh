#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=3,4
export HF_HOME=/playpen-shared/haochenz/hf_cache
export TRANSFORMERS_CACHE=/playpen-shared/haochenz/hf_cache
export HF_DATASETS_CACHE=/playpen-shared/haochenz/hf_cache
export WANDB_DIR=/playpen-shared/haochenz/wandb
export WANDB_CACHE_DIR=/playpen-shared/haochenz/wandb_cache

BATCH_SIZE=8
GRAD_ACC=1
NUM_GPUS=2
LR=2e-5
NUM_EPOCHS=5
MAX_LENGTH=384

MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_DIR="/playpen-shared/haochenz/UMU-Bench/full_data/train-00000-of-00001.parquet"

EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACC * NUM_GPUS))

SAVE_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset_lr${LR}_effbs${EFFECTIVE_BS}"

mkdir -p /playpen-shared/haochenz/hf_cache

echo "Effective batch size = ${EFFECTIVE_BS}"
echo "Running finetuning..."

accelerate launch \
  --num_processes ${NUM_GPUS} \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  finetune.py \
  --model_id ${MODEL_ID} \
  --save_dir ${SAVE_DIR} \
  --data_dir ${DATA_DIR} \
  --batch_size ${BATCH_SIZE} \
  --gradient_accumulation_steps ${GRAD_ACC} \
  --lr ${LR} \
  --num_epochs ${NUM_EPOCHS} \
  --max_length ${MAX_LENGTH}

echo "Done."