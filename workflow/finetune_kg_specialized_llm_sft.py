import os
import torch
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import torch.utils.data
import datasets
import dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
)
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
import transformers
import trl

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
except ValueError:
    N_CPUS = 4
    logger.warning(f"Could not determine CPU count, defaulting to {N_CPUS} workers")

def is_bf16_supported():
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception: # Changed from general except
        return False

class Constants:
    HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ScriptArguments:
    data_path_list: List[str] = field(metadata={"help": "Path(s) to the SFT training data directories (Hugging Face datasets format). Each dataset should contain 'prompt' and 'completion' columns."})
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the SFT evaluation data directory (Hugging Face datasets format). Should also contain 'prompt' and 'completion' columns."}
    )
    # Removed response_template_with_prompt as formatting_func is removed
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat-hf")
    use_peft: bool = field(default=False)
    save_merged: bool = field(default=False)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    bnb_4bit_quant_type: str = field(default="nf4")
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_use_double_quant: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")

@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default="saved_models/llama2_sft")
    optim: str = field(default="adamw_torch")
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)
    dataloader_pin_memory: bool = field(default=True)
    remove_unused_columns: bool = field(default=False) # SFTTrainer default is True. If False, ensure all columns are handled or needed.
                                                      # For prompt/completion, SFTTrainer should handle them.
    per_device_train_batch_size: int = field(default=8)
    overwrite_output_dir: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=5e-5)
    num_train_epochs: int = field(default=3)
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=50)
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=2)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.03)

# Removed global _RESPONSE_TEMPLATE_WITH_PROMPT
# Removed formatting_func

def train():
    logger.info("Starting SFT training pipeline")
    parser = HfArgumentParser((ScriptArguments, TrainingConfig))
    args, training_args = parser.parse_args_into_dataclasses()

    logger.info("Environment information:")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device_count = torch.cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    logger.info(f"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, TRL: {trl.__version__}")

    is_distributed = device_count > 1 and training_args.local_rank != -1
    logger.info(f"Distributed training: {is_distributed}")
    
    # Quantization setup
    compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype) if args.bnb_4bit_compute_dtype else None
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )
    elif args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        token=Constants.HF_TOKEN
    )
    special_tokens_dict = {}
    added_tokens = 0

    if tokenizer.pad_token is None:
        logger.info("Tokenizer does not have a pad token. Adding <PAD> as pad token.")
        special_tokens_dict["pad_token"] = "<PAD>" 
        added_tokens += 1
    
    if tokenizer.pad_token_id is None and tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    if added_tokens > 0:
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} special tokens.")
        if tokenizer.pad_token_id is None and "pad_token" in special_tokens_dict: 
             tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(special_tokens_dict["pad_token"])

    model_kwargs = {
        "trust_remote_code": True,
        "token": Constants.HF_TOKEN,
        "torch_dtype": torch.bfloat16 if training_args.bf16 and is_bf16_supported() else 
                       (torch.float16 if training_args.fp16 else None),
    }
    
    if args.attn_implementation == "flash_attention_2":
        try:
            import flash_attn 
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8: 
                logger.warning("Flash Attention 2 requires CUDA capability >= 8.0. Falling back to 'sdpa' attention.")
                model_kwargs["attn_implementation"] = "sdpa" 
            else:
                model_kwargs["attn_implementation"] = args.attn_implementation
                logger.info("Using Flash Attention 2.")
        except ImportError:
            logger.warning("Flash Attention 2 requested but not installed. Falling back to 'sdpa' attention.")
            model_kwargs["attn_implementation"] = "sdpa"
    elif args.attn_implementation: 
        model_kwargs["attn_implementation"] = args.attn_implementation
        logger.info(f"Using specified attention implementation: {args.attn_implementation}")

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        if is_distributed and (args.load_in_4bit or args.load_in_8bit):
            model_kwargs["device_map"] = {"": training_args.local_rank}
            logger.info(f"Setting device_map to {{'': {training_args.local_rank}}} for DDP with quantization.")
        else:
            model_kwargs["device_map"] = "auto"
            logger.info("Setting device_map to 'auto'.")
    else: 
        if is_distributed:
             logger.info("Distributed training without quantization. device_map not explicitly set for DDP, relying on DDP handling.")
        else: 
            model_kwargs["device_map"] = "auto" 
            logger.info("Single device training. Setting device_map to 'auto'.")

    logger.info(f"Loading model from {args.model_name_or_path} with kwargs: {model_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    peft_config = None
    if args.use_peft:
        target_modules_list = args.target_modules.split(",") if args.target_modules else ["q_proj", "v_proj", "k_proj", "o_proj"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules_list,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, target_modules={target_modules_list}")

    logger.info("Loading SFT training dataset...")
    datasets.enable_progress_bar()
    try:
        if isinstance(args.data_path_list, str):
            data_paths = [args.data_path_list]
        else:
            data_paths = args.data_path_list

        if not data_paths:
            raise ValueError("No training data paths provided in data_path_list.")

        logger.info(f"Loading SFT training datasets from paths: {data_paths}")
        data_list = [datasets.load_from_disk(path) for path in tqdm(data_paths, desc="Loading SFT training datasets")]
        train_dataset = datasets.concatenate_datasets(data_list)
        logger.info(f"Successfully loaded and concatenated SFT training datasets. Total examples: {len(train_dataset)}")
    except Exception as e:
        logger.error(f"Failed to load SFT training dataset from {args.data_path_list}: {e}", exc_info=True)
        raise
    datasets.disable_progress_bar()

    # Verify columns for SFT (prompt and completion are expected by SFTTrainer for instruction format)
    required_columns_for_instruction = ['prompt', 'completion'] 
    if not all(col in train_dataset.column_names for col in required_columns_for_instruction):
        missing_cols = [col for col in required_columns_for_instruction if col not in train_dataset.column_names]
        logger.error(f"SFT Training dataset missing required columns for instruction format: {missing_cols}. Found columns: {train_dataset.column_names}")
        raise ValueError(f"SFT Training dataset missing required columns for instruction format: {missing_cols}")
    logger.info(f"SFT Training dataset columns for instruction format verified. All columns: {train_dataset.column_names}")

    eval_dataset = None
    if args.eval_dataset_path:
        logger.info(f"Loading SFT evaluation dataset from {args.eval_dataset_path}")
        try:
            eval_dataset = datasets.load_from_disk(args.eval_dataset_path)
            logger.info(f"Successfully loaded SFT evaluation dataset. Total examples: {len(eval_dataset)}")
            if not all(col in eval_dataset.column_names for col in required_columns_for_instruction):
                missing_cols = [col for col in required_columns_for_instruction if col not in eval_dataset.column_names]
                logger.error(f"SFT Evaluation dataset missing required columns for instruction format: {missing_cols}. Found columns: {eval_dataset.column_names}")
                raise ValueError(f"SFT Evaluation dataset missing required columns for instruction format: {missing_cols}")
            logger.info(f"SFT Evaluation dataset columns for instruction format verified.")
        except Exception as e:
            logger.error(f"Failed to load SFT evaluation dataset from {args.eval_dataset_path}: {e}", exc_info=True)
            eval_dataset = None 
            logger.warning(f"Proceeding without SFT evaluation dataset due to loading error.")
    
    global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps 
    if is_distributed:
        global_batch_size *= device_count
    
    logger.info(f"Global batch size for SFT: {global_batch_size}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        # formatting_func=None, # Explicitly None, or simply omit as it's the default
        # dataset_text_field=None, # SFTTrainer will look for 'prompt'/'completion' or use chat template
        # packing=True, # Consider adding SFTConfig for this if desired, or set max_seq_length
    )
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            is_actual_checkpoint_dir = any(
                os.path.isdir(os.path.join(training_args.output_dir, item)) and 
                item.startswith("checkpoint-") for item in os.listdir(training_args.output_dir)
            )
            if not is_actual_checkpoint_dir:
                logger.warning(
                    f"Output directory {training_args.output_dir} exists, is not empty, "
                    f"but contains no valid checkpoint subdirectories. Will proceed as if it's a new training run, "
                    f"potentially overwriting files if not careful. Consider using --overwrite_output_dir or cleaning the directory."
                )
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint found, resuming SFT from {last_checkpoint}")

    checkpoint_to_resume = training_args.resume_from_checkpoint or last_checkpoint

    logger.info("Starting SFT training")
    if is_distributed:
        torch.distributed.barrier() 
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    if training_args.local_rank <= 0: 
        logger.info("Saving final SFT model")
        trainer.save_model(training_args.output_dir) 
        tokenizer.save_pretrained(training_args.output_dir) 
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if args.use_peft and args.save_merged:
        if is_distributed:
            torch.distributed.barrier() 
            
        if training_args.local_rank <= 0: 
            logger.info("Merging LoRA adapters for SFT model")
            merged_model_path = os.path.join(training_args.output_dir, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            
            try:
                base_model_dtype = torch.bfloat16 if training_args.bf16 and is_bf16_supported() else \
                                   (torch.float16 if training_args.fp16 else torch.float32)
                
                logger.info(f"Reloading base model ({args.model_name_or_path}) for merging with dtype: {base_model_dtype}")
                base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=base_model_dtype,
                    trust_remote_code=True,
                    token=Constants.HF_TOKEN,
                    device_map="cpu", 
                )
                
                if added_tokens > 0: 
                    base_model_for_merge.resize_token_embeddings(len(tokenizer))
                
                logger.info(f"Loading PEFT model from {training_args.output_dir} to merge.")
                merged_model = PeftModel.from_pretrained(base_model_for_merge, training_args.output_dir, is_trainable=False)
                
                logger.info("Merging LoRA layers...")
                merged_model = merged_model.merge_and_unload() 
                
                logger.info(f"Saving merged SFT model to {merged_model_path}")
                merged_model.save_pretrained(merged_model_path)
                tokenizer.save_pretrained(merged_model_path)
                logger.info(f"Merged SFT model saved successfully to {merged_model_path}")
            except Exception as e:
                logger.error(f"Error merging SFT model: {e}", exc_info=True)

    logger.info("SFT Training completed")

if __name__ == "__main__":
    train()
