#!/bin/bash

# --- 硬件配置 ---
GPUS_PER_NODE=1
export CUDA_VISIBLE_DEVICES=2

export HF_HOME=/playpen/haochenz/hf_cache
MASTER_PORT=29500

# --- 固定路径配置 ---
MODEL_ID="llava-hf/llava-1.5-7b-hf"
DATA_DIR="/playpen/haochenz/UMU-Bench/full_data/train-00000-of-00001.parquet"
BASE_SAVE_DIR="../checkpoints/llava-sweep"

# --- 固定训练参数 ---
BATCH_SIZE=3
GRAD_ACCUM=2
EPOCHS=3  # Sweep 时可以适当减少 Epoch 以节省时间

# --- 💡 定义要测试的学习率列表 ---
LR_LIST=(1e-5 2e-5 5e-5 1e-4)

# --- 开始循环 Sweep ---
for LR in "${LR_LIST[@]}"
do
    # 动态生成保存路径和任务名称，例如：llava-lr-1e-5
    RUN_NAME="llava-lr-$LR"
    CURRENT_SAVE_DIR="$BASE_SAVE_DIR/$RUN_NAME"

    mkdir -p $CURRENT_SAVE_DIR

    echo "------------------------------------------------"
    echo "🚀 Starting Sweep: $RUN_NAME with LR=$LR"
    echo "------------------------------------------------"

    # 使用 torchrun 启动
    # 注意：我在后面添加了 --run_name 参数（假设你在 python 代码里处理它）
    # 或者通过环境变量传递给 wandb
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
        --lr $LR

    echo "✅ Finished Sweep for LR=$LR"
done

echo "All sweeps completed."