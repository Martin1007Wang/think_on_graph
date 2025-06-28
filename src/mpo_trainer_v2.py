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

# PEFT imports
if_peft_available = True
if if_peft_available:
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

logger = logging.getLogger(__name__)


@dataclass
class DataCollatorForMPO(DataCollatorMixin):
    """Memory-optimized data collator for MPO training"""
    pad_token_id: int
    return_tensors: str = "pt"
    
    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            return {}
        
        # Pre-allocate tensors to avoid repeated memory allocation
        batch_size = len(features)
        max_seq_len = max(len(f["input_ids"]) for f in features)
        max_candidates = max(f["num_candidates"] for f in features)
        
        # Efficient batching with pre-allocation
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
    Multi-Preference Optimization (MPO) Trainer
    
    Implements memory-efficient training for knowledge graph navigation
    using multi-label classification with reference model comparison.
    """
    _tag_names = ["trl", "mpo"]
    
    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: TrainingArguments = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        peft_config: Optional[Dict] = None,
        beta: float = 0.1,
        **kwargs,
    ):
        if not isinstance(args, TrainingArguments):
            raise ValueError("args must be a TrainingArguments object")
        
        if processing_class is None:
            raise ValueError("MPOTrainer requires a processing_class (tokenizer).")

        # Initialize main model
        model_init_kwargs = kwargs.pop("model_init_kwargs", {})
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        
        # Initialize reference model
        if ref_model is None:
            logger.info("Creating reference model from main model")
            ref_model = create_reference_model(model)
        elif isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **model_init_kwargs)
        
        # Disable dropout in reference model and set to eval mode
        disable_dropout_in_model(ref_model)
        ref_model.eval()
        self.ref_model = ref_model
        
        # Training hyperparameters
        self.beta = beta
        self.max_length = getattr(args, 'max_length', 2048)
        
        # Setup data collator
        padding_value = processing_class.pad_token_id if processing_class.pad_token_id is not None else processing_class.eos_token_id
        data_collator = DataCollatorForMPO(pad_token_id=padding_value)

        # Process datasets
        if train_dataset is not None:
            train_dataset = train_dataset.map(
                self.tokenize_navigation_sample,
                fn_kwargs={
                    "tokenizer": processing_class, 
                    "max_length": self.max_length
                },
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False,
                desc="Tokenizing training data"
            )
        
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(
                self.tokenize_navigation_sample,
                fn_kwargs={
                    "tokenizer": processing_class, 
                    "max_length": self.max_length
                },
                remove_columns=eval_dataset.column_names,
                desc="Tokenizing evaluation data"
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            **kwargs,
        )

    @staticmethod
    def tokenize_navigation_sample(
        feature: Dict[str, Any], 
        tokenizer: PreTrainedTokenizerBase, 
        max_length: int
    ) -> Dict[str, Any]:
        """Tokenize navigation sample with proper error handling"""
        prompt_text = feature["prompt"]
        chosen_relations = feature.get("chosen", [])
        if not chosen_relations:
            raise ValueError("Feature must contain a 'chosen' key with at least one relation.")
        
        negative_relations = feature.get("rejected", [])
        all_candidates = list(set(chosen_relations + negative_relations))  # Remove duplicates
        all_candidates.sort()  # Ensure consistent ordering

        # Create binary labels
        positive_labels = [1.0 if candidate in chosen_relations else 0.0 for candidate in all_candidates]

        # Tokenize prompt
        encoded = tokenizer(
            prompt_text, 
            truncation=True, 
            max_length=max_length, 
            add_special_tokens=True,
            return_tensors=None  # Return lists, not tensors
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "positive_labels": positive_labels,
            "num_candidates": len(all_candidates),
            "candidates": all_candidates
        }

    def get_relation_logits_batch(
        self, 
        model: nn.Module, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidates: List[List[str]],
        batch_size: int = 32
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size_samples = input_ids.size(0)
        max_candidates = max(len(cands) for cands in candidates) if candidates else 0
        
        if max_candidates == 0:
            return torch.full((batch_size_samples, 0), -1e9, device=device, dtype=torch.float)
        
        # Initialize result tensor
        relation_scores = torch.full(
            (batch_size_samples, max_candidates), 
            -1e9, 
            device=device, 
            dtype=torch.float
        )
        
        # Process each sample
        for sample_idx, sample_candidates in enumerate(candidates):
            if not sample_candidates:
                continue
                
            context_len = attention_mask[sample_idx].sum().item()
            context_tokens = input_ids[sample_idx, :context_len]
            
            # Process candidates in batches to prevent memory buildup
            num_candidates = len(sample_candidates)
            for start_idx in range(0, num_candidates, batch_size):
                end_idx = min(start_idx + batch_size, num_candidates)
                batch_candidates = sample_candidates[start_idx:end_idx]
                
                # Prepare batch inputs
                batch_inputs = []
                batch_metadata = []
                
                for cand_idx, relation_str in enumerate(batch_candidates):
                    relation_tokens = self.processing_class.encode(
                        relation_str, 
                        add_special_tokens=False
                    )
                    if not relation_tokens:
                        continue
                        
                    full_input = torch.cat([
                        context_tokens,
                        torch.tensor(relation_tokens, device=device, dtype=torch.long)
                    ], dim=0)
                    
                    batch_inputs.append(full_input)
                    batch_metadata.append((start_idx + cand_idx, len(relation_tokens), context_len))
                
                if not batch_inputs:
                    continue
                
                # Pad and process batch
                padded_inputs = torch.nn.utils.rnn.pad_sequence(
                    batch_inputs, 
                    batch_first=True, 
                    padding_value=self.processing_class.pad_token_id
                )
                padded_attention = (padded_inputs != self.processing_class.pad_token_id).long()
                
                # Forward pass with memory management
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = model(
                        input_ids=padded_inputs,
                        attention_mask=padded_attention,
                        use_cache=False,
                        return_dict=True
                    )
                    logits = outputs.logits
                
                # Compute scores for this batch
                log_probs = F.log_softmax(logits, dim=-1)
                
                for i, (global_cand_idx, relation_len, ctx_len) in enumerate(batch_metadata):
                    if i >= log_probs.size(0):
                        continue
                    
                    # Extract target tokens and compute average log probability
                    start_pos = ctx_len - 1
                    end_pos = ctx_len + relation_len - 1
                    
                    if end_pos >= padded_inputs.size(1):
                        continue
                    
                    target_logits = log_probs[i, start_pos:end_pos]
                    target_tokens = padded_inputs[i, ctx_len:end_pos + 1]
                    token_scores = target_logits.gather(
                        dim=1, 
                        index=target_tokens.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    avg_score = token_scores.mean()
                    relation_scores[sample_idx, global_cand_idx] = avg_score
        
        return relation_scores

    def compute_kl_penalty(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        num_candidates: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL penalty between policy and reference model"""
        # Create mask for valid candidates
        candidate_mask = torch.arange(
            policy_logits.size(1), 
            device=policy_logits.device
        )[None, :] < num_candidates[:, None]
        
        # Convert logits to probabilities
        policy_probs = torch.softmax(policy_logits, dim=-1)
        ref_probs = torch.softmax(ref_logits.detach(), dim=-1)
        
        # Compute KL divergence
        kl_div = F.kl_div(
            F.log_softmax(policy_logits, dim=-1),
            ref_probs,
            reduction='none'
        )
        
        # Apply mask and compute mean
        masked_kl = kl_div * candidate_mask.float()
        kl_penalty = masked_kl.sum(dim=1) / num_candidates.clamp(min=1).float()
        
        return kl_penalty.mean()

    def compute_multilabel_classification_loss(
        self,
        policy_logits: torch.Tensor,
        ref_logits: torch.Tensor,
        labels: torch.Tensor,
        num_candidates: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MPO loss with reference model comparison
        
        The loss combines:
        1. Multi-label classification loss (BCE)
        2. KL penalty to prevent deviation from reference model
        """
        # Create candidate mask
        candidate_mask = torch.arange(
            policy_logits.size(1), 
            device=policy_logits.device
        )[None, :] < num_candidates[:, None]
        
        # Multi-label classification loss
        bce_loss = F.binary_cross_entropy_with_logits(
            policy_logits, 
            labels, 
            reduction='none'
        )
        masked_bce = bce_loss * candidate_mask.float()
        classification_loss = (masked_bce.sum(dim=1) / num_candidates.clamp(min=1).float()).mean()
        
        # KL penalty
        kl_penalty = self.compute_kl_penalty(policy_logits, ref_logits, num_candidates)
        
        # Total loss
        total_loss = classification_loss + self.beta * kl_penalty
        
        # Compute metrics (detached to prevent memory leaks)
        with torch.no_grad():
            policy_probs = torch.sigmoid(policy_logits)
            predictions = (policy_probs > 0.5).float() * candidate_mask.float()
            
            # Exact match accuracy
            label_mask = labels * candidate_mask.float()
            pred_mask = predictions * candidate_mask.float()
            exact_matches = (label_mask == pred_mask).all(dim=1)
            exact_match_acc = exact_matches.float().mean()
            
            # Positive confidence
            positive_mask = (labels == 1.0) * candidate_mask.float()
            positive_confidence = (policy_probs * positive_mask).sum() / positive_mask.sum().clamp(min=1)
        
        # Ensure all metrics are Python scalars to prevent memory leaks
        metrics = {
            "mpo/total_loss": total_loss.detach().cpu().item(),
            "mpo/classification_loss": classification_loss.detach().cpu().item(),
            "mpo/kl_penalty": kl_penalty.detach().cpu().item(),
            "mpo/exact_match_accuracy": exact_match_acc.detach().cpu().item(),
            "mpo/positive_confidence": positive_confidence.detach().cpu().item(),
        }
        
        return total_loss, metrics

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute MPO loss with proper memory management
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        num_candidates = inputs["num_candidates"]
        candidates = inputs["candidates"]

        # Get policy model logits
        policy_logits = self.get_relation_logits_batch(
            model, input_ids, attention_mask, candidates
        )
        
        # Get reference model logits
        with torch.no_grad():
            ref_logits = self.get_relation_logits_batch(
                self.ref_model, input_ids, attention_mask, candidates
            )
        
        # Compute loss and metrics
        total_loss, metrics = self.compute_multilabel_classification_loss(
            policy_logits, ref_logits, labels, num_candidates
        )
        
        # Log metrics
        if self.is_world_process_zero():
            self.log(metrics)
        
        if return_outputs:
            return total_loss, metrics
        return total_loss
    
    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Evaluation step with proper memory management"""
        with torch.no_grad():
            loss, metrics = self.compute_loss(model, inputs, return_outputs=True)
            
        if prediction_loss_only:
            return (loss, None, None)
        
        # Return detached labels to prevent memory leaks
        return (loss, None, inputs["labels"].detach())


# Enhanced utility functions
def create_navigation_dataset(navigation_data: List[Dict]) -> Dataset:
    """
    Create navigation training dataset with validation
    
    Args:
        navigation_data: Navigation decision data with format:
        [
            {
                "prompt": "Navigate from France to find its capital",
                "chosen": ["capital"],
                "rejected": ["population", "language", "area"]
            },
            ...
        ]
    """
    # Validate data format
    required_keys = {"prompt", "chosen", "rejected"}
    for i, item in enumerate(navigation_data):
        if not all(key in item for key in required_keys):
            raise ValueError(f"Item {i} missing required keys: {required_keys}")
        if not item["chosen"]:
            raise ValueError(f"Item {i} has empty 'chosen' list")
    
    return Dataset.from_list(navigation_data)


def train_mpo_model(
    model_name: str,
    train_data: List[Dict],
    eval_data: Optional[List[Dict]] = None,
    ref_model_name: Optional[str] = None,
    output_dir: str = "./mpo_model",
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,  # Reduced for memory efficiency
    learning_rate: float = 5e-5,
    beta: float = 0.1,
    **kwargs
):
    """
    Train MPO model with proper memory management
    
    Args:
        model_name: Name or path of the model to train
        train_data: Training data in MPO format
        eval_data: Evaluation data (optional)
        ref_model_name: Reference model name (defaults to model_name)
        output_dir: Output directory for trained model
        num_train_epochs: Number of training epochs
        per_device_train_batch_size: Batch size per device
        learning_rate: Learning rate
        beta: KL penalty coefficient
        **kwargs: Additional training arguments
    """
    from transformers import AutoTokenizer, TrainingArguments
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = create_navigation_dataset(train_data)
    eval_dataset = create_navigation_dataset(eval_data) if eval_data else None
    
    # Enhanced training arguments for memory efficiency
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=kwargs.pop("gradient_accumulation_steps", 2),
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        dataloader_drop_last=False,
        remove_unused_columns=True,
        # Memory optimization
        dataloader_pin_memory=False,
        fp16=True,  # Enable mixed precision
        gradient_checkpointing=True,
        **kwargs
    )
    
    # Create trainer
    trainer = MPOTrainer(
        model=model_name,
        ref_model=ref_model_name or model_name,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        beta=beta,
    )
    
    # Train model
    trainer.train()
    
    # Save model and tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    return trainer