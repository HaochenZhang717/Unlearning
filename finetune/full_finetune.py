import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import torch
from transformers import (
    BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor,
    get_scheduler, AutoTokenizer
)
from torch.optim import AdamW

from ft_dataset import (
    Muitimodal_Dataset, Unimodal_Dataset,
    train_collate_fn_llava_muitimodal, train_collate_fn_llava_unimodal
)
from accelerate import Accelerator
import wandb


# Identify all linear layers in the model for applying LoRA

def load_model_and_processor(model_id):
    if model_id.startswith("llava"):
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


######################### Main Training Entry #################################
def main(args):
    # Load model and processor
    model, processor = load_model_and_processor(args.model_id)
    model.train()
    print("Processor Tokenizer Length: ", len(processor.tokenizer))

    # Load tokenizer and ensure embedding size matches
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    os.makedirs(args.save_dir, exist_ok=True)

    # LoRA configuration

    print("This is NOT a PEFT model.")

    # Load dataset
    df = pd.read_parquet(args.data_dir)
    multimodel_dataset = Muitimodal_Dataset(df=df)
    unimodel_dataset = Unimodal_Dataset(df=df)

    # Build dataloaders for multimodal and unimodal training
    if args.model_id.startswith("llava"):
        train_dataloader_multimodal = DataLoader(
            multimodel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_muitimodal(x, processor, args)
        )
        train_dataloader_unimodal = DataLoader(
            unimodel_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor, args)
        )

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Optimizer and learning rate scheduler setup
    optimizer = AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader_multimodal) * args.num_epochs,
    )

    # Prepare model and dataloaders with accelerator
    model, optimizer, train_dataloader_multimodal, train_dataloader_unimodal, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_multimodal, train_dataloader_unimodal, lr_scheduler
    )

    wandb.init(
        project="UMU-bench",
        name=f"finetune_{args.model_id.split('/')[-1]}",
        config=vars(args)
    )

    global_step = 0

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        multi_progress_bar = tqdm(train_dataloader_multimodal, desc=f"Epoch {epoch + 1}")


        for batch in multi_progress_bar:
            input_ids, attention_mask, pixel_values, labels = batch
            with accelerator.accumulate(model):
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
            total_loss += loss.item()

            wandb.log({
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch
            }, step=global_step)
            global_step += 1


            multi_progress_bar.set_postfix(loss=total_loss / len(multi_progress_bar))
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader_multimodal)}")

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for configurable options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, default="llava-hf/llava-1.5-7b-hf", help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save the model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory for the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps"
    )
    args = parser.parse_args()
    main(args)
