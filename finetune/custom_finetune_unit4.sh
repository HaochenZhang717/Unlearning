#!/bin/bash

# --- 硬件配置 ---
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/playpen/haochenz/hf_cache
MASTER_PORT=29501 # 建议换一个端口防止冲突

# --- 固定路径配置 ---
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_DIR="/playpen/haochenz/UMU-Bench/full_data/train-00000-of-00001.parquet"
BASE_SAVE_DIR="../checkpoints/llava-sweep"

# --- 训练参数 ---
BATCH_SIZE=3
GRAD_ACCUM=2
EPOCHS=5 # 论文建议是 5

# --- 💡 定义要测试的参数列表 ---
LR_LIST=(1e-4)
LORA_R_LIST=(8 16 32 64)

# --- 开始嵌套循环 Sweep ---
for LR in "${LR_LIST[@]}"
do
  for R in "${LORA_R_LIST[@]}"
  do
    # 动态生成保存路径和任务名称，区分 LR 和 R
    RUN_NAME="llava-lr-${LR}-r-${R}"
    CURRENT_SAVE_DIR="$BASE_SAVE_DIR/$RUN_NAME"

    mkdir -p $CURRENT_SAVE_DIR

    echo "------------------------------------------------"
    echo "🚀 Starting Sweep: $RUN_NAME | LR=$LR, LoRA_R=$R"
    echo "------------------------------------------------"

    export WANDB_NAME=$RUN_NAME

    torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        custom_finetune.py \
        --model_id $MODEL_ID \
        --data_dir $DATA_DIR \
        --save_dir $CURRENT_SAVE_DIR \
        --batch_size $BATCH_SIZE \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_epochs $EPOCHS \
        --lr $LR \
        --lora_r $R \
        --lora_alpha 16

    echo "✅ Finished Sweep for LR=$LR, R=$R"
  done
done

echo "All sweeps completed."