import os
import torch
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
from transformers import Trainer
from peft import LoraConfig, get_peft_model, PeftModel
from src.template import KnowledgeGraphTemplates
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
import transformers
import trl
from trl import SFTTrainer

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
    PATH_START_TOKEN = "<PATH>"
    PATH_END_TOKEN = "</PATH>"
    HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ScriptArguments:
    data_path_list: List[str] = field(metadata={"help": "Path to the raw training data directories"})
    preprocessed_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to save/load preprocessed dataset"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store processed dataset cache"}
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether to use cached preprocessed datasets"}
    )
    force_preprocess: bool = field(
        default=False,
        metadata={"help": "Force dataset preprocessing even if cache exists"}
    )
    response_template: str = field(default="", metadata={"help": "Response template"})
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
    batch_size: int = field(default=4)
    max_prompt_length: int = field(default=512)  # 增加提示长度
    eval_dataset_path: str = field(default="")

@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default="saved_models/llama2_align")
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=2048, 
        metadata={"help": "Maximum sequence length for tokenization."}
    )
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=N_CPUS)
    dataloader_pin_memory: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
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

def get_cache_path(args, data_paths, dataset_type, cache_dir, is_eval):
    if cache_dir is None:
        if is_eval and args.preprocessed_data_path:
            cache_path = args.preprocessed_data_path + "_eval_sft"
        elif args.preprocessed_data_path:
            cache_path = args.preprocessed_data_path + "_sft"
        else:
            os.makedirs("cache", exist_ok=True)
            paths_hash = hash(str(sorted(data_paths)))
            cache_path = f"cache/sft_dataset_{dataset_type}_{paths_hash}_processed"
    else:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"sft_dataset_{dataset_type}_processed")
    return cache_path

def save_dataset_cache(dataset, cache_path, all_prompts, all_completions):
    try:
        logger.info(f"Saving processed SFT dataset to cache: {cache_path}")
        dataset.save_to_disk(cache_path)
        json_path = cache_path + ".json"
        try:
            import json
            max_samples = min(100, len(all_prompts))
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    [{"prompt": all_prompts[i], "completion": all_completions[i]} 
                     for i in range(max_samples)],
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.info(f"Saved sample of SFT dataset as JSON to: {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save SFT dataset sample as JSON: {e}")
    except Exception as e:
        logger.error(f"Failed to save dataset cache: {e}")

def prepare_sft_dataset(data_paths, tokenizer, args, is_eval=False, cache_dir=None, use_cache=True, force_preprocess=False):
    dataset_type = "evaluation" if is_eval else "training"
    if use_cache:
        cache_path = get_cache_path(args, data_paths, dataset_type, cache_dir, is_eval)
        if os.path.exists(cache_path) and not force_preprocess:
            logger.info(f"Loading processed {dataset_type} SFT dataset from cache: {cache_path}")
            return datasets.load_from_disk(cache_path)
    
    logger.info(f"Preparing SFT {dataset_type} dataset from {len(data_paths)} sources")
    datasets.enable_progress_bar()
    data_list = []
    for path in tqdm(data_paths, desc=f"Loading {dataset_type} datasets"):
        try:
            data_list.append(datasets.load_from_disk(path))
        except Exception as e:
            logger.error(f"Failed to load dataset from {path}: {e}")
    
    if not data_list:
        raise ValueError(f"No valid datasets could be loaded from the provided paths")
    
    dataset = datasets.concatenate_datasets(data_list)
    
    # Pre-compile templates for better performance
    prompt_template = KnowledgeGraphTemplates.ZERO_SHOT_PROMPT
    positive_template = KnowledgeGraphTemplates.POSITIVE_PATH_TEMPLATE
    
    # Optimized batch processing function
    def process_batch(examples):
        batch_size = len(examples["question"])
        sft_pairs_list = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            question = examples.get("question", [""])[i]
            q_entity = examples.get("q_entity", [""])[i]
            a_entity = examples.get("a_entity", [""])[i]
            prompt = prompt_template.format(question=question, entity=q_entity)
            
            positive_paths = examples.get("positive_paths", [[]])[i]
            if not isinstance(positive_paths, list):
                positive_paths = []
                
            for path in positive_paths:
                if not path or not path.strip():
                    continue
                    
                formatted_path = f"{Constants.PATH_START_TOKEN}{path.strip()}{Constants.PATH_END_TOKEN}"
                response = positive_template.format(
                    reasoning_path=formatted_path,
                    answer=a_entity
                )
                sft_pairs_list[i].append({"prompt": prompt, "response": response})
        
        return {"sft_pairs": sft_pairs_list}
    
    # Process in batches for better efficiency
    batch_size = 32  # Adjust based on memory constraints
    processed_data = dataset.map(
        process_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=min(8, N_CPUS),
        remove_columns=dataset.column_names,
        desc=f"Processing {dataset_type} examples"
    )
    
    # Flatten the dataset efficiently
    all_prompts = []
    all_completions = []
    
    for item in processed_data:
        for pair in item["sft_pairs"]:
            all_prompts.append(pair["prompt"])
            all_completions.append(pair["response"])
    
    total_samples = len(all_prompts)
    logger.info(f"Generated {total_samples} SFT pairs from {len(dataset)} examples")
    
    if total_samples == 0:
        logger.warning(f"No valid samples generated for {dataset_type}")
        return datasets.Dataset.from_dict({"prompt": [], "completion": []})
    
    # Create dataset directly from lists for better memory efficiency
    sft_dataset = datasets.Dataset.from_dict({
        "prompt": all_prompts,
        "completion": all_completions
    })
    
    # Save cache if needed
    if use_cache:
        save_dataset_cache(sft_dataset, cache_path, all_prompts, all_completions)
    
    datasets.disable_progress_bar()
    return sft_dataset


def train():
    logger.info("Starting training pipeline")
    parser = HfArgumentParser((ScriptArguments, TrainingConfig))
    args, training_args = parser.parse_args_into_dataclasses()

    # --- Input Validation ---
    if not args.data_path_list:
        raise ValueError("data_path_list cannot be empty. Please provide at least one path to training data.")
    # --- End Validation ---

    logger.info("Environment information:")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device_count = torch.cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    logger.info(f"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, TRL: {trl.__version__}")

    is_distributed = device_count > 1 and training_args.local_rank != -1
    logger.info(f"Distributed training: {is_distributed}")
    
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        token=Constants.HF_TOKEN
    )
    tokenizer.padding_side = "right"
    
    special_tokens_dict = {}
    added_tokens = 0
    
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<PAD>"
        added_tokens += 1
        
    current_special_tokens = set(tokenizer.all_special_tokens)
    additional_tokens = []
    
    if Constants.PATH_START_TOKEN not in current_special_tokens:
        additional_tokens.append(Constants.PATH_START_TOKEN)
    if Constants.PATH_END_TOKEN not in current_special_tokens:
        additional_tokens.append(Constants.PATH_END_TOKEN)
        
    if additional_tokens:
        special_tokens_dict['additional_special_tokens'] = additional_tokens
        added_tokens += len(additional_tokens)
        
    if added_tokens > 0:
        tokenizer.add_special_tokens(special_tokens_dict)

    model_kwargs = {
        "trust_remote_code": True,
        "token": Constants.HF_TOKEN,
        "torch_dtype": torch.bfloat16 if training_args.bf16 and is_bf16_supported() else 
                      (torch.float16 if training_args.fp16 else None),
        "device_map": "auto"
    }
    
    if args.attn_implementation == "flash_attention_2":
        try:
            import flash_attn
            if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 7:
                logger.warning("Flash Attention 2 requires CUDA capability >= 7.0. Falling back to default.")
                model_kwargs["attn_implementation"] = None
            else:
                model_kwargs["attn_implementation"] = args.attn_implementation
        except ImportError:
            logger.warning("Flash Attention 2 requested but not installed. Falling back to default.")
            model_kwargs["attn_implementation"] = None
    
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["device_map"] = {"": torch.cuda.current_device()} if is_distributed else "auto"

    logger.info(f"Loading model from {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    
    if added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    
    model.config.use_cache = False

    peft_config = None
    if args.use_peft:
        target_modules = args.target_modules.split(",") if args.target_modules else ["q_proj", "v_proj", "k_proj", "o_proj"]
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA configuration: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")

    sft_dataset = None
    if args.preprocessed_data_path and os.path.exists(args.preprocessed_data_path):
        logger.info(f"Loading preprocessed SFT training dataset from {args.preprocessed_data_path}")
        sft_dataset = datasets.load_from_disk(args.preprocessed_data_path)
    else:
        logger.info("Generating SFT training dataset...")
        cache_dir = os.path.dirname(args.preprocessed_data_path) if args.preprocessed_data_path else "cache"
        sft_dataset = prepare_sft_dataset(
            args.data_path_list, 
            tokenizer, 
            args, 
            is_eval=False,
            cache_dir=cache_dir,
            use_cache=True,
            force_preprocess=training_args.overwrite_output_dir
        )
        if args.preprocessed_data_path and training_args.local_rank <= 0 and not os.path.exists(args.preprocessed_data_path):
            try:
                sft_dataset.save_to_disk(args.preprocessed_data_path)
                logger.info(f"Saved preprocessed SFT dataset to {args.preprocessed_data_path}")
            except Exception as e:
                logger.error(f"Failed to save preprocessed SFT data: {e}")

    eval_dataset = None
    if args.eval_dataset_path:
        eval_preprocessed_path = args.preprocessed_data_path + "_eval" if args.preprocessed_data_path else None
        if eval_preprocessed_path and os.path.exists(eval_preprocessed_path):
            eval_dataset = datasets.load_from_disk(eval_preprocessed_path)
        else:
            cache_dir = os.path.dirname(eval_preprocessed_path) if eval_preprocessed_path else "cache"
            eval_dataset = prepare_sft_dataset(
                [args.eval_dataset_path], 
                tokenizer, 
                args, 
                is_eval=True,
                cache_dir=cache_dir,
                use_cache=True,
                force_preprocess=training_args.overwrite_output_dir
            )
            if eval_preprocessed_path and training_args.local_rank <= 0 and not os.path.exists(eval_preprocessed_path):
                try:
                    eval_dataset.save_to_disk(eval_preprocessed_path)
                except Exception as e:
                    logger.error(f"Failed to save preprocessed eval data: {e}")
    global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    if is_distributed:
        global_batch_size *= device_count
    logger.info(f"Global SFT batch size: {global_batch_size}")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config if args.use_peft else None,
    )
    
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            if not any(os.path.isdir(os.path.join(training_args.output_dir, item)) and 
                       item.startswith("checkpoint-") for item in os.listdir(training_args.output_dir)):
                logger.warning(f"Output directory exists without checkpoints. Continuing.")
            else:
                raise ValueError(f"Output directory exists and is not empty. Use --overwrite_output_dir to override.")
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint found, resuming from {last_checkpoint}")

    checkpoint = training_args.resume_from_checkpoint or last_checkpoint

    # Training
    logger.info("Starting SFT training")
    if is_distributed:
        torch.distributed.barrier()
    
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Save model and metrics
    if training_args.local_rank <= 0:
        logger.info("Saving final model")
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Merge and save LoRA adapters if requested
    if args.use_peft and args.save_merged:
        if is_distributed:
            torch.distributed.barrier()
            
        if training_args.local_rank <= 0:
            logger.info("Merging LoRA adapters")
            merged_model_path = os.path.join(training_args.output_dir, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            
            try:
                # Reload base model for merging
                base_model_dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
                base_model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=base_model_dtype,
                    trust_remote_code=True,
                    token=Constants.HF_TOKEN,
                    device_map="cpu",
                )
                
                if added_tokens > 0:
                    base_model.resize_token_embeddings(len(tokenizer))
                
                # Load and merge adapter
                merged_model = PeftModel.from_pretrained(base_model, training_args.output_dir)
                merged_model = merged_model.merge_and_unload()
                
                # Save merged model
                merged_model.save_pretrained(merged_model_path)
                tokenizer.save_pretrained(merged_model_path)
                logger.info(f"Merged model saved to {merged_model_path}")
            except Exception as e:
                logger.error(f"Error merging model: {e}")

    logger.info("SFT Training completed")

if __name__ == "__main__":
    train()