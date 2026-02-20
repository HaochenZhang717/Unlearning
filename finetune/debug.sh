#!/bin/bash

# --- 硬件配置 ---
# 这里的 nproc_per_node 通常设置为你的 GPU 数量
GPUS_PER_NODE=8
MASTER_PORT=29500

# --- 路径配置 ---
MODEL_ID="/Users/zhc/Downloads/llava-1.5-7b-hf"
DATA_DIR="/Users/zhc/Downloads/UMU-Bench/full_data/train-00000-of-00001.parquet"
SAVE_DIR="../unlearning_checkpoints/llava-finetune-ddp"

# --- 训练参数 ---
BATCH_SIZE=2            # 每张显卡的 batch size
GRAD_ACCUM=4            # 梯度累积步数
EPOCHS=3
LR=2e-5

# 创建保存目录
mkdir -p $SAVE_DIR

echo "Starting training with $GPUS_PER_NODE GPUs..."

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$MASTER_PORT \
    finetune.py \
    --model_id $MODEL_ID \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_epochs $EPOCHS \
    --lr $LR

echo "Training completed."