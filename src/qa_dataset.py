# qa_dataset.py
# 构建Dataset数据集

# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
import torch
import json
import numpy as np

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # 兼容 train.py 传入的已切片数据列表 (list) 或文件路径 (str)
        if isinstance(data_path, list):
            # 新逻辑：如果传入的是列表，直接使用它作为数据集
            self.data = data_path
        elif isinstance(data_path, str) and data_path:
            # 原始逻辑：如果传入的是字符串，视为文件路径，按行读取 JSONL 格式
            with open(data_path, "r", encoding='utf-8') as f:
                for line in f:
                    if not line or line.strip() == "":
                        continue
                    try:
                        json_line = json.loads(line)
                        self.data.append(json_line)
                    except json.JSONDecodeError:
                        continue # 忽略格式错误的行

        print("data load ， size：", len(self.data))


    def preprocess(self, input, output):
        messages = [{"role": "system", "content": "你是一个心理医疗专家, 可以帮助用户解决心理咨询问题。"},
                    {"role": "user", "content": input}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        instruction = self.tokenizer(prompt, add_special_tokens=False)
        response = self.tokenizer(
            output, add_special_tokens=False, max_length=self.max_length, truncation=True
        )
        input_ids = instruction["input_ids"] + response["input_ids"] + [self.tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [self.tokenizer.eos_token_id]
        return input_ids, attention_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, attention_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(attention_mask)),
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data)