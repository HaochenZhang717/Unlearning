#!/bin/bash

# ==============================
# Basic config
# ==============================
export CUDA_VISIBLE_DEVICES=1

MODEL_ID="llava-hf/llava-1.5-7b-hf"

# 你本地已经下载好的 llava 权重目录（vanilla_dir）
VANILLA_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset"

# Oracle 模型路径（用于 retain KL 约束）
# ⚠️ 这里要填你 oracle llava checkpoint 的路径
ORACLE_MODEL_ID="/playpen-shared/haochenz/UMU-Bench-result/ckpts/finetuned_llava_fullset"

# 数据 split 目录（里面应该有 forget_5/retain_95/... 这种结构）
DATA_SPLIT_DIR="/playpen-shared/haochenz/UMU-Bench"

FORGET_RATIO=5
BATCH_SIZE=6
ALPHA=1.0
GAMMA=1.0
LR=1e-5
EPOCHS=5
MAX_LEN=384

# 保存输出模型的目录
SAVE_DIR="/playpen-shared/haochenz/UMU-Bench-result/ckpts/llava_KL_forget${FORGET_RATIO}_gamma${GAMMA}_alpha${ALPHA}"


# ==============================
# Run with accelerate
# ==============================
accelerate launch \
  --num_processes 1 \
  --num_machines 1 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  --gradient_accumulation_steps ${BATCH_SIZE} \
  KL.py \
  --model_id ${MODEL_ID} \
  --vanilla_dir ${VANILLA_DIR} \
  --oracle_model_id ${ORACLE_MODEL_ID} \
  --save_dir ${SAVE_DIR} \
  --data_split_dir ${DATA_SPLIT_DIR} \
  --forget_split_ratio ${FORGET_RATIO} \
  --batch_size 1 \
  --alpha ${ALPHA} \
  --gamma ${GAMMA} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --max_length ${MAX_LEN}