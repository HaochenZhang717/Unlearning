import os
import argparse
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from torch.utils.data import ConcatDataset
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_scheduler,
    AutoTokenizer
)

# 导入你的自定义数据处理逻辑
from ft_dataset import (
    Muitimodal_Dataset, Unimodal_Dataset,
    train_collate_fn_llava_hybrid
)


def setup():
    # 初始化进程组，使用环境变量（RANK, WORLD_SIZE, MASTER_ADDR/PORT）
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def load_model_and_processor(model_id, local_rank):
    print(f"Loading LLAVA model on rank {local_rank}...")
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(local_rank)

    processor = AutoProcessor.from_pretrained(model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    return model, processor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    # 1. 初始化分布式环境
    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    # 2. 加载模型与处理器
    model, processor = load_model_and_processor(args.model_id, local_rank)
    # 3. LoRA 配置
    # 1️⃣ freeze vision
    for param in model.model.vision_tower.parameters():
        param.requires_grad = False
    # 2️⃣ enable projector
    for param in model.model.multi_modal_projector.parameters():
        param.requires_grad = True
    llm = model.model.language_model

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    llm = get_peft_model(llm, lora_config)
    model.model.language_model = llm


    print(model)
    breakpoint()
    # 4. 包装为 DDP 模型
    # find_unused_parameters=True 应对 vision_tower 冻结不参与更新的情况
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 5. 数据准备 (关键：使用 DistributedSampler)
    df = pd.read_parquet(args.data_dir)
    multimodal_dataset = Muitimodal_Dataset(df=df)
    unimodal_dataset = Unimodal_Dataset(df=df)

    combined_dataset = ConcatDataset([multimodal_dataset, unimodal_dataset])

    sampler = DistributedSampler(combined_dataset, shuffle=True)

    train_dataloader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda x: train_collate_fn_llava_hybrid(x, processor),
        num_workers=4,
        pin_memory=True
    )

    # 6. 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_epochs
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 7. 日志 (仅在主进程运行)
    if global_rank == 0:
        wandb.init(project="UMU-bench", name=f"ddp_finetune_{args.model_id.split('/')[-1]}", config=vars(args))
        os.makedirs(args.save_dir, exist_ok=True)

    # 8. 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        sampler.set_epoch(epoch)

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", disable=(global_rank != 0))

        for i, batch in enumerate(progress_bar):
            # 更加优雅的数据移动方式，处理 batch 字典中所有的 tensor
            batch = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # 建议加上 autocast 匹配 bfloat16
                # 使用 **batch 自动解包 input_ids, pixel_values, labels 等
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_rank == 0:
                wandb.log({"loss": loss.item() * args.gradient_accumulation_steps, "lr": lr_scheduler.get_last_lr()[0]},
                          step=global_step)
                progress_bar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)

    # 9. 保存模型
    dist.barrier()
    if global_rank == 0:
        # 1. 先从 DDP 中取出 PEFT 模型
        peft_model = model.module
        # 2. 合并 LoRA 权重到基础模型 (如果你想保存完整模型权重)
        # 注意：merge_and_unload 需要大量内存，如果内存不够，仅 save_pretrained 即可
        final_model = peft_model.merge_and_unload()
        final_model.save_pretrained(args.save_dir)
        processor.save_pretrained(args.save_dir)
        print(f"Model successfully saved to: {args.save_dir}")


if __name__ == "__main__":
    main()