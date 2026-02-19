#!/bin/bash
set -e

mkdir -p /playpen-shared/haochenz/hf_cache

export HF_HOME=/playpen-shared/haochenz/hf_cache
export HF_DATASETS_CACHE=/playpen-shared/haochenz/hf_cache
export WANDB_DIR=/playpen-shared/haochenz/wandb
export WANDB_CACHE_DIR=/playpen-shared/haochenz/wandb_cache

export CUDA_VISIBLE_DEVICES=7,9

NUM_GPUS=2
BATCH_SIZE=1
GRAD_ACC=8
LR=8e-6
NUM_EPOCHS=5
MAX_LENGTH=384

MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_DIR="/playpen-shared/haochenz/UMU-Bench/full_data/train-00000-of-00001.parquet"

EFFECTIVE_BS=$((BATCH_SIZE * GRAD_ACC * NUM_GPUS))

SAVE_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/fullfinetuned_llava_fullset_lr${LR}_effbs${EFFECTIVE_BS}"

echo "Effective batch size = ${EFFECTIVE_BS}"
echo "Running finetuning on ${NUM_GPUS} GPUs..."

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