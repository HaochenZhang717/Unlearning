import pandas as pd
import copy
import json
from typing import Any, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
import os
from io import BytesIO
from PIL import Image
import torch
from torch.utils.data import DataLoader
import ast

class Muitimodal_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # Extract the bytes from the 'image' dictionary
            image_data = row['image'].get('bytes')  # Access the image bytes

            # Convert the image bytes to a PIL Image
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue
            python_dict = ast.literal_eval(row['MM_QA'])
            json_str = json.dumps(python_dict, indent=4)
            QAs = json.loads(json_str)
            questions = QAs['question']
            answers = QAs['answer']
            for k in questions.keys():
                flattened_data.append({
                    "image": image,
                    "question":questions[k],
                    "answer": answers[k]
                })  


        return flattened_data
    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


def train_collate_fn_llava_muitimodal(examples, processor):
    texts = []
    images = []

    for example in examples:
        image = example.get('image')
        question = example.get('question')
        answer = example.get('answer')

        # 1. 对齐官方的 Chat Template 格式
        # 注意：这里我们手动把答案拼在后面，因为 apply_chat_template 通常只生成 prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]
        # 获取标准 Prompt
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        # 将答案拼接到 Prompt 之后，形成完整的训练序列
        full_text = f"{prompt}{answer}{processor.tokenizer.eos_token}"

        texts.append(full_text)
        images.append(image)

    # 2. 统一处理 batch
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 3. 对齐标签：只对 "ASSISTANT" 的回答部分计算 Loss
    labels = batch["input_ids"].clone()

    # 遮蔽 Padding 部分
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 【进阶对齐】遮蔽 Prompt 部分（可选但推荐）
    # 如果你想让模型只学回答，需要找到 prompt 的长度并把 labels 对应位置设为 -100
    for i, text in enumerate(texts):
        # 重新编码 prompt 部分以获得其长度
        prompt_ids = processor.tokenizer(
            processor.apply_chat_template(conversation, add_generation_prompt=True),
            add_special_tokens=False
        ).input_ids
        prompt_len = len(prompt_ids)
        # 将 labels 中属于 prompt 的部分设为 -100
        labels[i, :prompt_len] = -100

    batch["labels"] = labels

    # 4. 返回字典，方便 model(**batch) 调用
    return batch


class Unimodal_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # QAs = json.loads(row['UM_QA'])
            python_dict = ast.literal_eval(row['UM_QA'])
            json_str = json.dumps(python_dict, indent=4)
            QAs = json.loads(json_str)
            questions = QAs['question']
            answers = QAs['answer']
            for k in questions.keys():
                flattened_data.append({
                    "image": None,
                    "question":questions[k],
                    "answer": answers[k]
                })  
        return flattened_data

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        # image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": None,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


def train_collate_fn_llava_unimodal(examples, processor):
    texts = []

    for example in examples:
        question = example.get('question')
        answer = example.get('answer')

        # 1. 构造纯文本对话格式 (不包含 {"type": "image"})
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                ],
            },
        ]

        # 2. 生成标准 Prompt 并拼接 Answer + EOS
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        full_text = f"{prompt}{answer}{processor.tokenizer.eos_token}"
        texts.append(full_text)

    if len(texts) == 0:
        raise ValueError("Empty batch.")

    # 3. 文本编码
    batch = processor(
        text=texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 4. 标签遮蔽逻辑
    labels = batch["input_ids"].clone()

    # 屏蔽 Padding
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 屏蔽 Prompt (问题部分)，只让模型预测答案
    for i, text in enumerate(texts):
        # 重新编码该样本的 prompt 部分以获取长度
        # 注意：add_special_tokens=False 避免二次添加 BOS token
        prompt_ids = processor.tokenizer(
            processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": examples[i].get('question')}]}],
                add_generation_prompt=True
            ),
            add_special_tokens=False
        ).input_ids

        prompt_len = len(prompt_ids)
        # 将 labels 中对应的 prompt 区域设为 -100
        labels[i, :prompt_len] = -100

    batch["labels"] = labels

    # 5. 返回字典格式，并在图像位补 None
    # 注意：LLaVA 模型在没有 pixel_values 时，会自动走纯文本分支
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "pixel_values": None,
        "labels": batch["labels"]
    }



def train_collate_fn_llava_hybrid(examples, processor):

    texts = []
    images = []

    for example in examples:
        question = example['question']
        answer = example['answer']
        image = example.get('image')  # Muitimodal_Dataset 返回 PIL Image, Unimodal_Dataset 返回 None

        # 1. 构造对话模板
        if image is not None:
            # 多模态样本：必须包含 <image> 占位符
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                    ],
                },
            ]
            images.append(image)
        else:
            # 单模态样本：纯文本格式
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                },
            ]
            # 关键点：LLaVA 1.5 即使是纯文本训练，processor 通常也期望有像素输入
            # 我们传一张全黑的 dummy image (336x336 是 LLaVA 1.5 默认大小)
            images.append(Image.new('RGB', (336, 336), (0, 0, 0)))

        # 2. 使用官方 chat_template 生成 Prompt
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        # 3. 拼接答案并添加 EOS (End of Sentence)
        # 这是为了让模型学会什么时候停止回答
        full_text = f"{prompt}{answer}{processor.tokenizer.eos_token}"
        texts.append(full_text)

    # 4. 调用 processor 转化为 Tensor
    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 5. 构造 Labels 并进行 Masking
    labels = batch["input_ids"].clone()

    # 遮蔽 Padding 部分
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # 遮蔽 Prompt 部分，只对 Answer 计算 Loss
    for i, full_str in enumerate(texts):
        # 重新根据当时的 conversation 拿到 prompt 部分的长度
        # 注意：这里需要再次调用 apply_chat_template 确保长度精确一致
        current_image = examples[i].get('image')
        if current_image is not None:
            temp_conv = [
                {"role": "user", "content": [{"type": "text", "text": examples[i]['question']}, {"type": "image"}]}]
        else:
            temp_conv = [{"role": "user", "content": [{"type": "text", "text": examples[i]['question']}]}]

        temp_prompt = processor.apply_chat_template(temp_conv, add_generation_prompt=True)

        # 编码 prompt 部分，不添加特殊 token
        prompt_token_ids = processor.tokenizer(temp_prompt, add_special_tokens=False).input_ids
        prompt_len = len(prompt_token_ids)

        # 将 labels 中对应的 prompt 区域设为 -100
        labels[i, :prompt_len] = -100

    batch["labels"] = labels

    return batch


if __name__ == "__main__":
    df = pd.read_parquet("/Users/zhc/Downloads/UMU-Bench/full_data/train-00000-of-00001.parquet")
    multimodel_dataset = Muitimodal_Dataset(df=df)
    unimodel_dataset = Unimodal_Dataset(df=df)

    # multimodal_datum = multimodel_dataset[0]
    # unimodal_datum = unimodel_dataset[0]

    model_id = "llava-hf/llava-1.5-7b-hf"

    processor = AutoProcessor.from_pretrained(model_id)

    train_dataloader_multimodal = DataLoader(
        multimodel_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_muitimodal(x, processor)
    )

    train_dataloader_unimodal = DataLoader(
        unimodel_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda x: train_collate_fn_llava_unimodal(x, processor)
    )

    multimodel_batch = next(iter(train_dataloader_multimodal))
    unimodel_batch = next(iter(train_dataloader_unimodal))

