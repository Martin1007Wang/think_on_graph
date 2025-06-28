import copy
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import DataCollatorMixin

from trl.models import create_reference_model
from trl.trainer.utils import disable_dropout_in_model

if_peft_available = True  # Assuming PEFT is available as in the source
if if_peft_available:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


# =================================================================================
#  Data Collator for MPO (Multi-Preference Optimization)
# =================================================================================
@dataclass
class DataCollatorForMPO(DataCollatorMixin):
    """
    Data Collator for Multi-Preference Optimization (MPO).
    
    MPO算法设计思路：
    1. 对于每个prompt，我们有多个chosen和rejected responses
    2. 将问题建模为二分类：给定(prompt, response)对，预测是chosen还是rejected
    3. 使用对比学习的思想：chosen responses应该有更高的reward，rejected responses应该有更低的reward
    4. 损失函数结合了分类损失和对比损失
    """

    pad_token_id: int
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        处理MPO样本为批量数据
        数据结构：每个feature包含一个prompt和多个chosen/rejected responses
        """
        all_input_ids, all_labels, prompt_indices, preference_labels = [], [], [], []

        for i, feature in enumerate(features):
            prompt_ids = feature["prompt_input_ids"]
            prompt_len = len(prompt_ids)

            # 处理chosen responses (标签为1)
            for chosen_ids in feature["chosen_input_ids"]:
                all_input_ids.append(torch.tensor(prompt_ids + chosen_ids, dtype=torch.long))
                all_labels.append(torch.tensor(([self.label_pad_token_id] * prompt_len) + chosen_ids, dtype=torch.long))
                prompt_indices.append(i)
                preference_labels.append(1)  # chosen = 1
            
            # 处理rejected responses (标签为0)
            for rejected_ids in feature["rejected_input_ids"]:
                all_input_ids.append(torch.tensor(prompt_ids + rejected_ids, dtype=torch.long))
                all_labels.append(torch.tensor(([self.label_pad_token_id] * prompt_len) + rejected_ids, dtype=torch.long))
                prompt_indices.append(i)
                preference_labels.append(0)  # rejected = 0

        if not all_input_ids:
            return {}

        batch = {}
        batch["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        batch["labels"] = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=self.label_pad_token_id
        )
        batch["attention_mask"] = (batch["input_ids"] != self.pad_token_id).long()
        
        batch["prompt_indices"] = torch.tensor(prompt_indices, dtype=torch.long)
        batch["preference_labels"] = torch.tensor(preference_labels, dtype=torch.float)  # 0 for rejected, 1 for chosen
        
        return batch


# =================================================================================
#  MPO Trainer - Redesigned as Multi-Preference Classification + Contrastive Learning
# =================================================================================
class MPOTrainer(Trainer):
    """
    Multi-Preference Optimization Trainer
    
    算法设计理念：
    1. **二分类视角**：将每个(prompt, response)对视为二分类问题，预测chosen(1)或rejected(0)
    2. **对比学习**：在同一个prompt下，chosen responses的reward应该高于rejected responses
    3. **多响应建模**：充分利用每个prompt的所有正负样本，而不是简单的pairwise比较
    4. **集成损失**：结合分类损失和对比损失，平衡准确性和偏好一致性
    
    数学公式：
    - Reward: r(x,y) = β * log(π_θ(y|x) / π_ref(y|x))
    - 分类损失: L_cls = -Σ[p_i * log(σ(r_i)) + (1-p_i) * log(1-σ(r_i))]
    - 对比损失: L_con = -Σ log(σ(r_chosen - r_rejected))
    - 总损失: L = λ_cls * L_cls + λ_con * L_con + λ_sft * L_sft
    """
    
    _tag_names = ["trl", "mpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        beta: float = 0.1,
        sft_loss_weight: float = 0.0,
        classification_loss_weight: float = 1.0,  # 分类损失权重
        contrastive_loss_weight: float = 1.0,    # 对比损失权重
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        peft_config: Optional[Dict] = None,
        **kwargs,
    ):
        if not isinstance(args, TrainingArguments):
            raise ValueError("args must be a TrainingArguments object")
        
        if tokenizer is None:
            raise ValueError("MPOTrainer requires a tokenizer.")
        
        model_init_kwargs = kwargs.pop("model_init_kwargs", {})
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        ref_model_init_kwargs = kwargs.pop("ref_model_init_kwargs", {})
        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        self.is_peft_model = if_peft_available and isinstance(model, PeftModel)
        if self.is_peft_model and ref_model is not None:
             raise ValueError("You passed both a ref_model and a peft_config. For training PEFT adapters with MPO, "
                              "pass `ref_model=None` to use the base model as the reference.")
        
        if if_peft_available and peft_config is not None:
            if not self.is_peft_model:
                if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
                model = get_peft_model(model, peft_config)
                self.is_peft_model = True

        if ref_model:
            self.ref_model = ref_model
        elif self.is_peft_model:
            self.ref_model = None
        else:
            self.ref_model = create_reference_model(model)

        # MPO特定参数
        self.beta = beta
        self.sft_loss_weight = sft_loss_weight
        self.classification_loss_weight = classification_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.max_length = getattr(args, 'max_length', 4096)
        self.max_prompt_length = getattr(args, 'max_prompt_length', 1024)

        padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        data_collator = DataCollatorForMPO(pad_token_id=padding_value)

        if train_dataset is not None:
            train_dataset = train_dataset.map(
                self.tokenize_row,
                fn_kwargs={"tokenizer": tokenizer, "max_prompt_length": self.max_prompt_length, "max_length": self.max_length},
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False
            )
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.tokenize_row,
                fn_kwargs={"tokenizer": tokenizer, "max_prompt_length": self.max_prompt_length, "max_length": self.max_length},
                remove_columns=eval_dataset.column_names,
            )
        
        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            **kwargs,
        )

    @staticmethod
    def tokenize_row(feature, tokenizer, max_prompt_length, max_length):
        """ Tokenize a single row of a dataset for MPO. """
        prompt_str = feature["prompt"]
        prompt_ids = tokenizer(prompt_str, truncation=True, max_length=max_prompt_length, add_special_tokens=False)["input_ids"]
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
        
        chosen_responses = feature.get("chosen", [])
        rejected_responses = feature.get("rejected", [])
        
        chosen_input_ids = []
        for response in chosen_responses:
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            response_ids.append(tokenizer.eos_token_id)
            if len(prompt_ids) + len(response_ids) > max_length:
                response_ids = response_ids[:max_length - len(prompt_ids)]
            chosen_input_ids.append(response_ids)

        rejected_input_ids = []
        for response in rejected_responses:
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            response_ids.append(tokenizer.eos_token_id)
            if len(prompt_ids) + len(response_ids) > max_length:
                response_ids = response_ids[:max_length - len(prompt_ids)]
            rejected_input_ids.append(response_ids)
        
        result = { "prompt_input_ids": prompt_ids, "chosen_input_ids": chosen_input_ids, "rejected_input_ids": rejected_input_ids }
        return result

    @contextmanager
    def null_ref_context(self):
        """Context manager for handling the reference model logic, especially for PEFT models."""
        if self.ref_model is None and self.is_peft_model:
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                yield
        else:
            yield

    def _get_batch_logps(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """计算批量log概率"""
        original_training_state = model.training
        try:
            # 在训练时不使用no_grad，在评估时使用
            if not original_training_state:
                model.eval()
                
            outputs = model(
                input_ids=inputs["input_ids"], 
                attention_mask=inputs["attention_mask"], 
                use_cache=False
            )
            logits = outputs.logits
            
            shifted_logits = logits[..., :-1, :].contiguous()
            labels = inputs["labels"][..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            per_token_loss = loss_fct(
                shifted_logits.view(-1, shifted_logits.size(-1)), 
                labels.view(-1)
            )
            per_token_loss = per_token_loss.view(labels.size())
            
            loss_mask = labels != self.data_collator.label_pad_token_id
            sequence_logps = -(per_token_loss * loss_mask).sum(dim=1)
            
            return sequence_logps
                
        finally:
            model.train(original_training_state)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        # 计算策略模型和参考模型的log概率
        policy_logps = self._get_batch_logps(model, inputs)
        
        with self.null_ref_context():
            reference_model = self.ref_model if self.ref_model is not None else self.model
            with torch.no_grad():
                ref_logps = self._get_batch_logps(reference_model, inputs)
        
        # 计算rewards (DPO风格)
        rewards = self.beta * (policy_logps - ref_logps.detach())
        
        # 计算MPO损失
        classification_loss, contrastive_loss, metrics_dict = self.mpo_loss(
            rewards, inputs["prompt_indices"], inputs["preference_labels"]
        )
        
        # 计算SFT损失
        sft_loss = torch.tensor(0.0, device=rewards.device)
        if self.sft_loss_weight > 0:
            chosen_mask = inputs["preference_labels"] == 1
            if torch.any(chosen_mask):
                chosen_logps = policy_logps[chosen_mask]
                sft_loss = -chosen_logps.mean()
        
        # 组合总损失
        total_loss = (
            self.classification_loss_weight * classification_loss +
            self.contrastive_loss_weight * contrastive_loss +
            self.sft_loss_weight * sft_loss
        )
        
        # 记录指标
        if self.is_world_process_zero():
            log_metrics = {
                "loss/total": total_loss.detach().cpu().item(),
                "loss/classification": classification_loss.detach().cpu().item(),
                "loss/contrastive": contrastive_loss.detach().cpu().item(),
            }
            if self.sft_loss_weight > 0:
                log_metrics["loss/sft"] = sft_loss.detach().cpu().item()
            
            log_metrics.update(metrics_dict)
            self.log(log_metrics)
        
        if return_outputs:
            return total_loss, log_metrics
        return total_loss

    def mpo_loss(
        self,
        rewards: torch.Tensor,
        prompt_indices: torch.Tensor,
        preference_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        MPO损失计算：分类损失 + 对比损失
        
        分类损失：将每个response分类为chosen(1)或rejected(0)
        对比损失：同一prompt下，chosen responses的reward应该高于rejected responses
        """
        device = rewards.device
        
        # 1. 分类损失：二分类交叉熵
        classification_probs = torch.sigmoid(rewards)
        classification_loss = F.binary_cross_entropy(
            classification_probs, 
            preference_labels, 
            reduction='mean'
        )
        
        # 2. 对比损失：同一prompt内的chosen vs rejected对比
        num_prompts = int(prompt_indices.max().item()) + 1 if len(prompt_indices) > 0 else 0
        contrastive_losses = []
        
        # 指标收集
        all_chosen_rewards, all_rejected_rewards, all_accuracies = [], [], []
        
        for i in range(num_prompts):
            prompt_mask = prompt_indices == i
            if not torch.any(prompt_mask):
                continue
                
            current_rewards = rewards[prompt_mask]
            current_labels = preference_labels[prompt_mask]
            
            chosen_rewards = current_rewards[current_labels == 1]
            rejected_rewards = current_rewards[current_labels == 0]
            
            if len(chosen_rewards) == 0 or len(rejected_rewards) == 0:
                continue
            
            # 收集指标
            all_chosen_rewards.append(chosen_rewards)
            all_rejected_rewards.append(rejected_rewards)
            
            # 计算准确率（chosen的平均reward是否高于rejected）
            accuracy = (chosen_rewards.mean() > rejected_rewards.mean()).float()
            all_accuracies.append(accuracy)
            
            # 对比损失：所有chosen vs rejected对的损失
            # 使用广播计算所有可能的chosen-rejected对
            reward_margins = chosen_rewards.unsqueeze(1) - rejected_rewards.unsqueeze(0)  # [n_chosen, n_rejected]
            contrastive_loss = -F.logsigmoid(reward_margins).mean()
            contrastive_losses.append(contrastive_loss)
        
        # 计算最终的对比损失
        if len(contrastive_losses) > 0:
            final_contrastive_loss = torch.stack(contrastive_losses).mean()
        else:
            final_contrastive_loss = torch.tensor(0.0, device=device)
        
        # 整合指标
        metrics = {}
        if all_chosen_rewards:
            chosen_rewards_all = torch.cat(all_chosen_rewards)
            metrics["rewards/chosen"] = chosen_rewards_all.mean().cpu().item()
        
        if all_rejected_rewards:
            rejected_rewards_all = torch.cat(all_rejected_rewards)
            metrics["rewards/rejected"] = rejected_rewards_all.mean().cpu().item()
        
        if all_accuracies:
            metrics["rewards/accuracy"] = torch.stack(all_accuracies).mean().cpu().item()
        
        if all_chosen_rewards and all_rejected_rewards:
            metrics["rewards/margin"] = (
                torch.cat(all_chosen_rewards).mean() - 
                torch.cat(all_rejected_rewards).mean()
            ).cpu().item()
        
        # 分类准确率
        classification_acc = ((classification_probs > 0.5).float() == preference_labels).float().mean()
        metrics["classification/accuracy"] = classification_acc.cpu().item()
        
        return classification_loss, final_contrastive_loss, metrics
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """ Perform an evaluation step on `model` using `inputs`. """
        with torch.no_grad():
            loss, metrics = self.compute_loss(model, inputs, return_outputs=True)

        if prediction_loss_only: 
            return (loss, None, None)
            
        # 构造dummy logits和labels用于兼容性
        logits = torch.tensor([
            metrics.get("rewards/chosen", 0), 
            metrics.get("rewards/rejected", 0),
            metrics.get("rewards/accuracy", 0)
        ], device=model.device, dtype=torch.float)
        
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.float)
        return (loss, logits, labels)