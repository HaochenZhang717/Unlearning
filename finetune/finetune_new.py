import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    get_scheduler,
    AutoTokenizer,
    AdamW
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)
from accelerate import Accelerator
import wandb

# 导入你的自定义 Dataset 逻辑
from ft_dataset import (
    Muitimodal_Dataset, Unimodal_Dataset,
    train_collate_fn_llava_muitimodal, train_collate_fn_llava_unimodal
)


def find_all_linear_names(model):
    """
    动态查找所有线性层以应用 LoRA。
    对于 LLaVA，重点通常在语言模型的 q_proj, v_proj 等。
    """
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            # 过滤掉不需要的层，确保兼容性
            if 'lm_head' not in names and 'vision_tower' not in names:
                lora_module_names.add(names[-1])
    return list(lora_module_names)


def load_model_and_processor(model_id):
    print(f"Loading model: {model_id}...")

    # 注意：配合 Accelerator 使用时，尽量不要在此时指定 device_map="auto"
    # 我们通过 Accelerator 统一管理设备
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    processor = AutoProcessor.from_pretrained(model_id)
    # 确保 pad_token 存在，LLaVA 1.5 默认可能没有 pad_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    processor.tokenizer.padding_side = "right"
    return model, processor


def main(args):
    # 1. 初始化 Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="wandb"
    )

    # 2. 加载模型和处理器
    model, processor = load_model_and_processor(args.model_id)

    # 3. LoRA 配置
    # target_modules 建议显式指定或使用动态查找
    target_modules = find_all_linear_names(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 准备 k-bit 训练（如果后续想用量化可以开启）并获取 PEFT 模型
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 4. 准备数据
    df = pd.read_parquet(args.data_dir)
    multimodal_ds = Muitimodal_Dataset(df=df)

    train_dataloader = DataLoader(
        multimodal_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_muitimodal(x, processor, args),
        num_workers=4,  # 建议开启多线程预处理图像
        pin_memory=True
    )

    # 5. 优化器与调度器
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # 计算总步数
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(max_train_steps * 0.03),
        num_training_steps=max_train_steps,
    )

    # 6. 使用 Accelerator 包装
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 7. 初始化 WandB
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="UMU-bench",
            config=vars(args),
            init_kwargs={"wandb": {"name": f"finetune_{args.model_id.split('/')[-1]}"}}
        )

    # 8. 训练循环
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0

        progress_bar = tqdm(
            train_dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}"
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                # 假设 batch 是由 collate_fn 准备好的字典
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # 仅在梯度同步时更新（处理 Accumulation 逻辑）
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                total_loss += loss.detach().item()

            if accelerator.is_main_process:
                accelerator.log({"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

            global_step += 1
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})

    # 9. 保存模型
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        os.makedirs(args.save_dir, exist_ok=True)
        # 注意：这里我们保存的是 Adapter 权重
        # 如果要合并，建议另写脚本或在非量化环境下执行 merge_and_unload
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.save_dir,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model)
        )
        processor.save_pretrained(args.save_dir)
        print(f"Model and processor saved to: {args.save_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    args = parser.parse_args()
    main(args)