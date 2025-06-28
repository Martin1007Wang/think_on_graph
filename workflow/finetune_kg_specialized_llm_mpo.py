import os
import torch
import unsloth
from unsloth import FastLanguageModel
import logging
from dataclasses import dataclass, field
from typing import List, Optional
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
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
import random
import transformers
import peft
import trl

# Import our custom MPOTrainer
from src.mpo_trainer_v2 import MPOTrainer # Assuming the file is named mpo_trainer_from_dpo.py

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
    except:
        return False

class Constants:
    HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ScriptArguments:
    """ Arguments for the MPO training script. """
    data_path_list: List[str] = field(metadata={"help": "Path(s) to the **preprocessed** training data directories (Hugging Face datasets format)."})
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the **preprocessed** evaluation data directory (Hugging Face datasets format)."}
    )
    model_name_or_path: str = field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    use_peft: bool = field(default=False)
    save_merged: bool = field(default=False)
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    bnb_4bit_quant_type: str = field(default="nf4") 
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_use_double_quant: bool = field(default=True)
    attn_implementation: str = field(default="flash_attention_2")
    
    # MPO-specific arguments
    beta: float = field(default=0.1, metadata={"help": "The beta factor in MPO loss. Controls the KL divergence penalty."})
    sft_loss_weight: float = field(default=0.05, metadata={"help": "Weight for the SFT loss, for IPO-style regularization."})
    max_length: int = field(default=4096, metadata={"help": "The maximum sequence length for the model."})
    max_prompt_length: int = field(default=1024, metadata={"help": "The maximum prompt length to truncate to."})


@dataclass
class TrainingConfig(TrainingArguments):
    """ TrainingArguments customized for this script. """
    output_dir: str = field(default="saved_models/mpo_model")
    optim: str = field(default="adamw_torch")
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)
    dataloader_pin_memory: bool = field(default=True)
    remove_unused_columns: bool = field(default=False) # Important: Set to False to keep our custom columns
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
    deepspeed: Optional[str] = field(default=None)
    

def train():
    logger.info("Starting MPO training pipeline")
    parser = HfArgumentParser((ScriptArguments, TrainingConfig))
    args, training_args = parser.parse_args_into_dataclasses()

    logger.info("Environment information:")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device_count = torch.cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    logger.info(f"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, TRL: {trl.__version__}, PEFT: {peft.__version__}")

    is_distributed = device_count > 1 and training_args.local_rank != -1
    logger.info(f"Distributed training: {is_distributed}")
    
    dtype = None 
    if training_args.bf16 and is_bf16_supported(): dtype = torch.bfloat16
    elif training_args.fp16: dtype = torch.float16

    logger.info(f"Loading model '{args.model_name_or_path}' with Unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name_or_path,
        max_seq_length = args.max_length,
        dtype = dtype,
        load_in_4bit = args.load_in_4bit,
        token = Constants.HF_TOKEN,
        attn_implementation = args.attn_implementation,
    )
    logger.info("Model and tokenizer loaded successfully via Unsloth.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # For decoder-only models, left-padding is recommended for batch generation
    tokenizer.padding_side = 'left' 

    if args.use_peft:
        logger.info("Applying PEFT config using Unsloth's optimized method...")
        target_modules = args.target_modules.split(",") if args.target_modules else ["q_proj", "v_proj", "k_proj", "o_proj"]
        model = FastLanguageModel.get_peft_model(
            model,
            r = args.lora_r,
            lora_alpha = args.lora_alpha,
            lora_dropout = args.lora_dropout,
            target_modules = target_modules,
            bias = "none",
            use_gradient_checkpointing = training_args.gradient_checkpointing,
            random_state = training_args.seed
        )
        logger.info(f"Unsloth PEFT model created. LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}")

    # Dataset loading
    logger.info("Loading preprocessed training dataset...")
    datasets.enable_progress_bar()
    try:
        data_paths = args.data_path_list if isinstance(args.data_path_list, list) else [args.data_path_list]
        if not data_paths:
            raise ValueError("No training data paths provided in data_path_list.")

        logger.info(f"Loading from paths: {data_paths}")
        data_list = [datasets.load_from_disk(path) for path in tqdm(data_paths, desc="Loading training datasets")]
        train_dataset = datasets.concatenate_datasets(data_list)
        # train_dataset = train_dataset.select(range(10))
        logger.info(f"Successfully loaded and concatenated training datasets. Total examples: {len(train_dataset)}")
    
    except Exception as e:
        logger.error(f"Failed to load training dataset from {args.data_path_list}: {e}")
        raise
    datasets.disable_progress_bar()

    required_columns = ['prompt', 'chosen', 'rejected']
    if not all(col in train_dataset.column_names for col in required_columns):
        missing_cols = [col for col in required_columns if col not in train_dataset.column_names]
        raise ValueError(f"Training dataset missing required columns: {missing_cols}")
    logger.info(f"Training dataset columns verified: {train_dataset.column_names}")

    # Evaluation dataset
    eval_dataset = None
    if args.eval_dataset_path:
        logger.info(f"Loading preprocessed evaluation dataset from {args.eval_dataset_path}")
        try:
            eval_dataset = datasets.load_from_disk(args.eval_dataset_path)
            if not all(col in eval_dataset.column_names for col in required_columns):
                raise ValueError(f"Evaluation dataset missing required columns.")
            logger.info(f"Evaluation dataset loaded and verified. Total examples: {len(eval_dataset)}")
        except Exception as e:
            logger.error(f"Failed to load evaluation dataset: {e}. Proceeding without evaluation.")
            eval_dataset = None

    # Log training info
    global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps 
    if is_distributed:
        global_batch_size *= device_count
    
    logger.info(f"Global batch size: {global_batch_size}, MPO beta: {args.beta}, SFT loss weight: {args.sft_loss_weight}")

    logger.info("Initializing MPOTrainer...")
    trainer = MPOTrainer(
        model=model,
        args=training_args,
        beta=args.beta,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    
    # Checkpoint handling
    last_checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if last_checkpoint:
        logger.info(f"Checkpoint found, resuming from {last_checkpoint}")

    # Training
    logger.info("Starting training...")
    if is_distributed: torch.distributed.barrier()
    
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save final model and metrics
    if training_args.local_rank <= 0:
        logger.info("Saving final model and metrics...")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("Final model and metrics saved.")

    # Merge and save LoRA adapters if requested
    if args.use_peft and args.save_merged and training_args.local_rank <= 0:
        logger.info("Merging LoRA adapters...")
        try:
            # Use the trainer's already prepared model
            merged_model = trainer.model.merge_and_unload()
            merged_model_path = os.path.join(training_args.output_dir, "merged_model")
            merged_model.save_pretrained(merged_model_path)
            tokenizer.save_pretrained(merged_model_path)
            logger.info(f"Merged model saved to {merged_model_path}")
        except Exception as e:
            logger.error(f"Error merging model: {e}")

    logger.info("Training completed successfully!")

if __name__ == "__main__":
    train()

