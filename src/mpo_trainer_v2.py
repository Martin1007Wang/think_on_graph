import copy
import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import math

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

# V3.0 FIX: Safe PEFT import
try:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT not available, some features will be disabled.")

logger = logging.getLogger(__name__)

# V3.0 ENHANCEMENT: Use constants for magic numbers
MPO_EPSILON = 1e-8
MPO_NEGATIVE_INF = -1e9

@dataclass
class MPOConfig:
    """Configuration class for MPOTrainer."""
    beta: float = 0.1
    positive_loss_weight: float = 0.1
    max_length: int = 2048
    
@dataclass
class DataCollatorForMPO(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}

        # The structure is largely fine, keeping it as is.
        # Dynamic padding is handled correctly here.
        batch_size = len(features)
        max_seq_len = max(len(f["input_ids"]) for f in features)
        max_candidates = max(f["num_candidates"] for f in features)

        batch_input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        batch_attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        batch_labels = torch.zeros((batch_size, max_candidates), dtype=torch.float)

        for i, feature in enumerate(features):
            seq_len = len(feature["input_ids"])
            num_cands = feature["num_candidates"]
            
            batch_input_ids[i, :seq_len] = torch.tensor(feature["input_ids"], dtype=torch.long)
            batch_attention_mask[i, :seq_len] = torch.tensor(feature["attention_mask"], dtype=torch.long)
            batch_labels[i, :num_cands] = torch.tensor(feature["positive_labels"], dtype=torch.float)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "num_candidates": torch.tensor([f["num_candidates"] for f in features], dtype=torch.long),
            "candidates": [f["candidates"] for f in features]
        }


class MPOTrainer(Trainer):
    """
    Multi-Preference Optimization (MPO) Trainer - V3.0 (Refactored for Performance & Robustness)
    """
    _tag_names = ["trl", "mpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        config: Optional[MPOConfig] = None,
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
        
        if config is None:
            logger.info("MPOConfig not provided. Using default values.")
            config = MPOConfig()
        
        self.config = config
        self.beta = self.config.beta
        self.positive_loss_weight = self.config.positive_loss_weight
        self.max_length = self.config.max_length

        model_init_kwargs = kwargs.pop("model_init_kwargs", {})
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        if PEFT_AVAILABLE and peft_config is not None:
             model = get_peft_model(model, peft_config)

        if ref_model is None:
            logger.info("Creating reference model from main model")
            ref_model = create_reference_model(model)
        elif isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **model_init_kwargs)
        
        disable_dropout_in_model(ref_model)
        ref_model.eval()
        self.ref_model = ref_model
        
        self.tokenizer = tokenizer
        
        padding_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        data_collator = DataCollatorForMPO(pad_token_id=padding_value)

        # Dataset processing remains the same, it's efficient enough.
        if train_dataset is not None:
            train_dataset = train_dataset.map(
                self.tokenize_navigation_sample,
                fn_kwargs={"tokenizer": tokenizer, "max_length": self.max_length},
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False, # Consider setting to True for large datasets
                desc="Tokenizing training data (V3.0)"
            )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.tokenize_navigation_sample,
                fn_kwargs={"tokenizer": tokenizer, "max_length": self.max_length},
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=False,
                desc="Tokenizing evaluation data (V3.0)"
            )

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
    def tokenize_navigation_sample(
        feature: Dict[str, Any], 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> Dict[str, Any]:
        """Tokenize navigation sample (V3.0 - with enhanced validation)."""
        prompt_text = feature.get("prompt")
        if not prompt_text or not isinstance(prompt_text, str):
             raise ValueError("Feature must contain a non-empty 'prompt' string.")

        chosen_relations = feature.get("chosen", [])
        if not chosen_relations or not all(isinstance(r, str) for r in chosen_relations):
            raise ValueError("Feature must contain a 'chosen' key with a list of non-empty strings.")

        rejected_relations = feature.get("rejected", [])
        weights = feature.get("weights", {})
        
        # Deduplicate candidates while preserving order
        seen = set()
        all_candidates = []
        for rel in chosen_relations + rejected_relations:
            if rel not in seen:
                all_candidates.append(rel)
                seen.add(rel)
        
        positive_labels = [weights.get(c, 1.0) if c in chosen_relations else 0.0 for c in all_candidates]

        encoded = tokenizer(
            prompt_text, 
            truncation=True, 
            max_length=max_length, 
            add_special_tokens=True,
            return_tensors=None
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "positive_labels": positive_labels,
            "num_candidates": len(all_candidates),
            "candidates": all_candidates
        }

    # V3.0 FIX: Major performance optimization. Replaced the old method with a fully batched version.
    def get_relation_logits_batch_optimized(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidates: List[List[str]],
    ) -> torch.Tensor:
        """
        Calculates relation logits in a fully batched manner.
        This method creates a "super-batch" of all (prompt + candidate) pairs
        and processes them in a single model forward pass.
        """
        device = model.device
        batch_size_samples = input_ids.size(0)
        max_candidates = max(len(c) for c in candidates) if candidates else 0
        if max_candidates == 0:
            return torch.full((batch_size_samples, 0), MPO_NEGATIVE_INF, device=device)

        super_batch_inputs = []
        super_batch_metadata = []

        for sample_idx, sample_candidates in enumerate(candidates):
            context_len = attention_mask[sample_idx].sum().item()
            context_tokens = input_ids[sample_idx, :context_len]
            
            for cand_idx, relation_str in enumerate(sample_candidates):
                try:
                    # Using `encode` is fine here as we're building a list before tensoring
                    relation_tokens = self.tokenizer.encode(relation_str, add_special_tokens=False)
                    if not relation_tokens:
                        logger.warning(f"Empty tokenization for relation: '{relation_str}'")
                        continue
                except Exception as e:
                    logger.error(f"Tokenization failed for relation: '{relation_str}', error: {e}")
                    continue

                full_input = torch.cat([
                    context_tokens, 
                    torch.tensor(relation_tokens, device=device, dtype=torch.long)
                ])
                super_batch_inputs.append(full_input)
                super_batch_metadata.append({
                    "sample_idx": sample_idx,
                    "cand_idx": cand_idx,
                    "context_len": context_len,
                    "relation_len": len(relation_tokens)
                })

        if not super_batch_inputs:
            return torch.full((batch_size_samples, max_candidates), MPO_NEGATIVE_INF, device=device)

        padded_inputs = torch.nn.utils.rnn.pad_sequence(
            super_batch_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id
        ).to(device)
        padded_attention = (padded_inputs != self.tokenizer.pad_token_id).long()
        
        # Single forward pass for all candidates in the batch
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = model(input_ids=padded_inputs, attention_mask=padded_attention, use_cache=False, return_dict=True)
            logits = outputs.logits
        
        log_probs = F.log_softmax(logits, dim=-1)

        relation_scores = torch.full((batch_size_samples, max_candidates), MPO_NEGATIVE_INF, device=device)
        
        for i, meta in enumerate(super_batch_metadata):
            ctx_len, rel_len = meta["context_len"], meta["relation_len"]
            start_pos, end_pos = ctx_len - 1, ctx_len + rel_len - 1

            if end_pos >= padded_inputs.size(1): continue

            target_logits = log_probs[i, start_pos:end_pos]
            target_tokens = padded_inputs[i, ctx_len : end_pos + 1]
            
            if target_logits.size(0) != target_tokens.size(0): continue

            token_scores = target_logits.gather(dim=1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
            
            if token_scores.numel() > 0:
                relation_scores[meta["sample_idx"], meta["cand_idx"]] = token_scores.mean()
        
        return relation_scores

    def compute_kl_penalty(self, policy_logits: torch.Tensor, ref_logits: torch.Tensor, num_candidates: torch.Tensor) -> torch.Tensor:
        candidate_mask = torch.arange(policy_logits.size(1), device=policy_logits.device)[None, :] < num_candidates[:, None]
        
        policy_probs = torch.sigmoid(policy_logits)
        ref_probs = torch.sigmoid(ref_logits.detach())

        policy_probs = policy_probs.clamp(min=MPO_EPSILON, max=1.0 - MPO_EPSILON)
        ref_probs = ref_probs.clamp(min=MPO_EPSILON, max=1.0 - MPO_EPSILON)

        kl_div = (
            policy_probs * (policy_probs.log() - ref_probs.log()) +
            (1 - policy_probs) * ((1 - policy_probs).log() - (1 - ref_probs).log())
        )
        masked_kl = kl_div * candidate_mask.float()
        return (masked_kl.sum(dim=1) / num_candidates.clamp(min=1).float()).mean()

    def compute_mpo_loss(self, policy_logits: torch.Tensor, ref_logits: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        candidate_mask = torch.arange(policy_logits.size(1), device=policy_logits.device)[None, :] < num_candidates[:, None]
        
        bce_loss = F.binary_cross_entropy_with_logits(policy_logits, labels, reduction='none')
        masked_bce = bce_loss * candidate_mask.float()
        classification_loss = (masked_bce.sum(dim=1) / num_candidates.clamp(min=1).float()).mean()
        
        kl_penalty = self.compute_kl_penalty(policy_logits, ref_logits, num_candidates)
        
        weighted_positive_mask = labels * candidate_mask.float()
        num_positives = weighted_positive_mask.sum().clamp(min=MPO_EPSILON)
        positive_log_probs = F.logsigmoid(policy_logits)
        positive_loss = -(positive_log_probs * weighted_positive_mask).sum() / num_positives
        
        total_loss = classification_loss + self.beta * kl_penalty + self.positive_loss_weight * positive_loss
        
        metrics = {
            "loss": total_loss.item(),
            "classification_loss": classification_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "positive_loss": positive_loss.item()
        }
        return total_loss, metrics

    # V3.0 FIX: Optimized and clarified evaluation metrics, especially MRR.
    def compute_evaluation_metrics(self, policy_logits: torch.Tensor, labels: torch.Tensor, num_candidates: torch.Tensor) -> Dict[str, float]:
        candidate_mask = torch.arange(policy_logits.size(1), device=policy_logits.device)[None, :] < num_candidates[:, None]
        policy_probs = torch.sigmoid(policy_logits)
        
        predictions = (policy_probs > 0.5).float() * candidate_mask.float()
        binary_labels = (labels > 0).float() * candidate_mask.float()
        
        # F1, Precision, Recall
        tp = (predictions * binary_labels).sum(dim=1)
        fp = (predictions * (1 - binary_labels)).sum(dim=1)
        fn = ((1 - predictions) * binary_labels).sum(dim=1)
        
        precision = tp / (tp + fp).clamp(min=MPO_EPSILON)
        recall = tp / (tp + fn).clamp(min=MPO_EPSILON)
        f1 = 2 * (precision * recall) / (precision + recall).clamp(min=MPO_EPSILON)
        
        exact_match = ((predictions == binary_labels) | ~candidate_mask).all(dim=1).float().mean()
        
        # MRR (Mean Reciprocal Rank) - Robust Calculation
        reciprocal_ranks = []
        for i in range(policy_probs.size(0)):
            if not candidate_mask[i].any() or not binary_labels[i].any():
                continue

            sample_probs = policy_probs[i][candidate_mask[i]]
            sample_labels = binary_labels[i][candidate_mask[i]]
            
            # Sort probabilities to get ranks
            sorted_indices = torch.argsort(sample_probs, descending=True)
            
            # Find the rank of the first correct item
            # `(sample_labels[sorted_indices] > 0).nonzero()` gives the positions of true items in the sorted list
            ranks = (sample_labels[sorted_indices] > 0).nonzero(as_tuple=True)[0]
            if ranks.numel() > 0:
                first_correct_rank = ranks[0].item() + 1  # Ranks are 1-based
                reciprocal_ranks.append(1.0 / first_correct_rank)

        mrr = torch.tensor(reciprocal_ranks).mean() if reciprocal_ranks else torch.tensor(0.0)
        
        return {
            "exact_match": exact_match.item(),
            "f1_score": f1.mean().item(),
            "precision": precision.mean().item(),
            "recall": recall.mean().item(),
            "mrr": mrr.item()
        }

    def compute_loss(self, model: nn.Module, inputs: Dict[str, Any], return_outputs=False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        policy_logits = self.get_relation_logits_batch_optimized(
            model, inputs["input_ids"], inputs["attention_mask"], inputs["candidates"]
        )
        with torch.no_grad():
            ref_logits = self.get_relation_logits_batch_optimized(
                self.ref_model, inputs["input_ids"], inputs["attention_mask"], inputs["candidates"]
            )
        
        total_loss, train_metrics = self.compute_mpo_loss(
            policy_logits, ref_logits, inputs["labels"], inputs["num_candidates"]
        )
        
        if self.is_training:
            self.log(train_metrics)
        
        return (total_loss, policy_logits) if return_outputs else total_loss

    def prediction_step(
        self, model: nn.Module, inputs: Dict[str, Any], prediction_loss_only: bool, ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model_inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, policy_logits = self.compute_loss(model, model_inputs, return_outputs=True)
            loss = loss.mean().detach()

        if prediction_loss_only:
            return (loss, None, None)

        eval_metrics = self.compute_evaluation_metrics(
            policy_logits, inputs["labels"], inputs["num_candidates"]
        )
        self.log(eval_metrics)
        return (loss, policy_logits, inputs["labels"])

# Example usage function, adapted for V3.0
def train_mpo_model(
    model_name: str,
    train_data: List[Dict],
    eval_data: Optional[List[Dict]] = None,
    ref_model_name: Optional[str] = None,
    output_dir: str = "./mpo_model_v3_0",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    positive_loss_weight: float = 0.1,
    **kwargs
):
    from transformers import AutoTokenizer, TrainingArguments
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size * 2,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=200,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=200 if eval_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_f1_score",
        greater_is_better=True,
        remove_unused_columns=False, # Important for our custom collator
        fp16=True,
        gradient_checkpointing=True,
        **kwargs
    )
    
    mpo_config = MPOConfig(
        beta=beta,
        positive_loss_weight=positive_loss_weight
    )
    
    trainer = MPOTrainer(
        model=model_name,
        ref_model=ref_model_name, # Can be None
        config=mpo_config,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer