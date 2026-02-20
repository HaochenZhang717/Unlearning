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
from peft import LoraConfig, get_peft_model
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_scheduler
)

# 假设你的 ft_dataset.py 中已经分别写好了这两个 collate 函数
# 如果没写，建议把原来的逻辑拆开：一个处理图文（加占位符），一个处理纯文（不加占位符）
from ft_dataset import (
    Muitimodal_Dataset, Unimodal_Dataset,
    train_collate_fn_llava_muitimodal,  # 处理图文
    train_collate_fn_llava_unimodal,  # 处理纯文
)


def setup():
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
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    # 这里的 accumulation 指的是这对 (Multi + Uni) 组合跑几次才更新梯度
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    args = parser.parse_args()

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)

    # 1. 模型加载与 LoRA 配置
    model, processor = load_model_and_processor(args.model_id, local_rank)

    for param in model.model.vision_tower.parameters():
        param.requires_grad = False
    for param in model.model.multi_modal_projector.parameters():
        param.requires_grad = True

    target_modules = ".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # 2. 准备两个独立的 DataLoader
    df = pd.read_parquet(args.data_dir)

    mm_dataset = Muitimodal_Dataset(df=df)
    mm_sampler = DistributedSampler(mm_dataset, shuffle=True)
    mm_loader = DataLoader(
        mm_dataset, batch_size=args.batch_size, sampler=mm_sampler,
        collate_fn=lambda x: train_collate_fn_llava_muitimodal(x, processor),
        num_workers=4, pin_memory=True
    )

    uni_dataset = Unimodal_Dataset(df=df)
    uni_sampler = DistributedSampler(uni_dataset, shuffle=True)
    uni_loader = DataLoader(
        uni_dataset, batch_size=args.batch_size, sampler=uni_sampler,
        collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor),
        num_workers=4, pin_memory=True
    )

    # 3. 优化器 (总步数取较长的那个 loader)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    num_training_steps = len(mm_loader) * args.num_epochs // args.gradient_accumulation_steps
    lr_scheduler = get_scheduler("linear", optimizer, 0, num_training_steps)

    if global_rank == 0:
        wandb.init(project="UMU-bench", name=f"dual_forward_llava", config=vars(args))
        os.makedirs(args.save_dir, exist_ok=True)

    # 4. 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        mm_sampler.set_epoch(epoch)
        uni_sampler.set_epoch(epoch)

        # 使用 zip 将两个数据源对齐（如果长度不一，zip 到短的停止）
        # 也可以手动处理迭代器来确保覆盖所有数据
        loader_combined = zip(mm_loader, uni_loader)
        progress_bar = tqdm(loader_combined, total=len(mm_loader), disable=(global_rank != 0))

        for i, (batch_mm, batch_uni) in enumerate(progress_bar):

            # 判断是否需要同步梯度（只有在累积结束的那一步同步）
            is_sync_step = (i + 1) % args.gradient_accumulation_steps == 0

            # --- 第一步：处理多模态数据 ---
            batch_mm = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch_mm.items()}

            # 在非同步步，使用 no_sync 提速
            context = model.no_sync() if not is_sync_step else torch.inference_mode(False)  # DDP 默认同步

            # 注意：DDP model.no_sync() 只在不进行梯度同步时使用
            # 我们这里采用：第一个 backward 永远 no_sync，第二个 backward 才触发同步
            with model.no_sync():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss_mm = model(**batch_mm).loss / (args.gradient_accumulation_steps * 2)
                loss_mm.backward()

            # --- 第二步：处理纯文本数据 ---
            batch_uni = {k: v.to(local_rank) if isinstance(v, torch.Tensor) else v for k, v in batch_uni.items()}

            # 如果是累积的最后一步，去掉 no_sync，让这一次 backward 触发 DDP 的梯度 All-Reduce
            if is_sync_step:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    loss_uni = model(**batch_uni).loss / (args.gradient_accumulation_steps * 2)
                loss_uni.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                global_step += 1
            else:
                with model.no_sync():
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        loss_uni = model(**batch_uni).loss / (args.gradient_accumulation_steps * 2)
                    loss_uni.backward()

            if global_rank == 0:
                total_loss = (loss_mm.item() + loss_uni.item()) * args.gradient_accumulation_steps * 2
                wandb.log({"loss": total_loss, "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)
                progress_bar.set_postfix(loss=total_loss)

    # 5. 保存
    dist.barrier()
    if global_rank == 0:
        # 只保存 LoRA 适配器（省内存省空间）或 merge 后保存
        model.module.save_pretrained(args.save_dir)
        processor.save_pretrained(args.save_dir)
        print("Model saved.")


if __name__ == "__main__":
    main()