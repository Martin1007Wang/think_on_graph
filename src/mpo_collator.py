import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, PreTrainedModel, TrainingArguments,PreTrainedTokenizer
from transformers.data.data_collator import DataCollatorMixin
from typing import Dict, Any, Tuple, Optional, Union, List
from torch.utils.data import DataLoader, Dataset
from contextlib import contextmanager
from dataclasses import dataclass
import copy

@dataclass
class DataCollatorForMPO(DataCollatorMixin):
    """
    为多偏好优化（MPO）专门设计的数据整理器。
    它将原始文本数据（包含一个prompt，多个chosen响应和多个rejected响应）
    处理成MPOTrainer所需的批次张量。

    设计思想 (TRL Alignment):
    - 封装性: 将复杂的批处理逻辑（如动态组合、padding、标签创建）封装在此类中，
      使得Trainer本身的代码更清晰，专注于算法。
    """
    tokenizer: PreTrainedTokenizer
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    max_length: int = 4096
    label_pad_token_id: int = -100 # DPO/TRL中常用的标签填充ID

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将一系列样本（字典）处理成一个批次的张量字典。
        """
        # 1. 扁平化数据结构：将每个prompt的所有chosen和rejected响应收集到同一个列表中
        prompts_text = [feature["prompt"] for feature in features]
        
        all_responses_text = []
        prompt_indices = []
        is_chosen_flags = []

        for i, feature in enumerate(features):
            # 确保即使没有"chosen"或"rejected"键，代码也能正常运行
            chosen_responses = feature.get("chosen", [])
            rejected_responses = feature.get("rejected", [])
            
            for response in chosen_responses:
                all_responses_text.append(response)
                prompt_indices.append(i)
                is_chosen_flags.append(True)

            for response in rejected_responses:
                all_responses_text.append(response)
                prompt_indices.append(i)
                is_chosen_flags.append(False)

        # 如果批次为空，返回空字典
        if not all_responses_text:
            return {}

        # 2. 分别Tokenize，以正确构建labels
        # 注意：此处不进行padding，因为每个prompt-response对的长度都不同
        tokenized_prompts = self.tokenizer(prompts_text, truncation=True, max_length=self.max_length, add_special_tokens=False)
        tokenized_responses = self.tokenizer(all_responses_text, truncation=True, max_length=self.max_length, add_special_tokens=False)

        # 3. 组合prompt和response，并创建labels
        batch_input_ids = []
        batch_labels = []

        for i in range(len(all_responses_text)):
            prompt_idx = prompt_indices[i]
            
            # 使用 .get() 方法安全地访问可能不存在的 tokenized prompts
            prompt_ids = tokenized_prompts.get('input_ids')[prompt_idx]
            response_ids = tokenized_responses.get('input_ids')[i]

            # 为prompt和response添加起始和结束token
            # 这是确保模型理解序列开始和结束的关键步骤
            input_ids = [self.tokenizer.bos_token_id] + prompt_ids + response_ids + [self.tokenizer.eos_token_id]
            labels = ([self.label_pad_token_id] * (1 + len(prompt_ids))) + response_ids + [self.tokenizer.eos_token_id]
            
            # 截断到最大长度
            batch_input_ids.append(input_ids[:self.max_length])
            batch_labels.append(labels[:self.max_length])

        # 4. 对整个批次进行Padding
        padded_batch = self.tokenizer.pad(
            {"input_ids": batch_input_ids},
            padding='longest',
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # 对labels进行同样的padding
        padded_labels_batch = self.tokenizer.pad(
             {"input_ids": batch_labels},
             padding='longest',
             max_length=self.max_length,
             pad_to_multiple_of=self.pad_to_multiple_of,
             return_tensors=self.return_tensors,
        )
        
        # 将label中的padding token替换为-100，使其在损失计算中被忽略
        labels = padded_labels_batch["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = self.label_pad_token_id
        padded_batch['labels'] = labels

        # 5. 返回最终的批次数据
        return {
            "input_ids": padded_batch['input_ids'],
            "attention_mask": padded_batch['attention_mask'],
            "labels": padded_batch['labels'],
            "prompt_indices": torch.tensor(prompt_indices, dtype=torch.long),
            "is_chosen_flags": torch.tensor(is_chosen_flags, dtype=torch.bool),
        }