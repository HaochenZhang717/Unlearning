#!/bin/bash

# ==============================
# Basic config
# ==============================
export CUDA_VISIBLE_DEVICES=7

MODEL_ID="llava-hf/llava-1.5-7b-hf"

# 你本地已经下载好的 llava 权重目录（vanilla_dir）
VANILLA_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset"

# 数据 split 目录（里面应该有 forget_5/retain_95/... 这种结构）
DATA_SPLIT_DIR="/playpen-shared/haochenz/UMU-Bench"

FORGET_RATIO=${1}
BATCH_SIZE=6
ALPHA=1.0
GAMMA=1.0
LR=1e-5
EPOCHS=5
MAX_LEN=384

# 保存输出模型的目录
SAVE_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/llava_PO_forget${FORGET_RATIO}"


# ==============================
# Run with accelerate
# ==============================
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --gradient_accumulation_steps ${BATCH_SIZE} \
  PO.py \
  --model_id ${MODEL_ID} \
  --vanilla_dir ${VANILLA_DIR} \
  --save_dir ${SAVE_DIR} \
  --data_split_dir ${DATA_SPLIT_DIR} \
  --gamma ${GAMMA} \
  --forget_split_ratio ${FORGET_RATIO} \
  --batch_size 1 \
  --alpha ${ALPHA} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --max_length ${MAX_LEN}