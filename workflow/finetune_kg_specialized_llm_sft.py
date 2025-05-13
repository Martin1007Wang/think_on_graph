import os
import torch
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict # Python 3.8需要从typing导入Dict
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
from peft import LoraConfig, PeftModel
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint
import transformers
import trl # type: ignore
from trl import SFTTrainer # type: ignore
import json 
import importlib.util 
import importlib.metadata 

dotenv.load_dotenv()
datasets.disable_progress_bar()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    if N_CPUS == 0: 
        N_CPUS = 1
        logger.warning("os.cpu_count() returned 0 or invalid value, defaulting to 1 worker.")
except (ValueError, TypeError): 
    N_CPUS = 4 
    logger.warning(f"Could not reliably determine CPU count, defaulting to {N_CPUS} workers.")

def is_bf16_supported():
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    except Exception as e: # Catch any other unexpected errors
        logger.warning(f"Error checking bf16 support: {e}")
        return False

class Constants:
    # PATH_START_TOKEN = "<PATH>" # Uncomment if used in 'content' fields
    # PATH_END_TOKEN = "</PATH>"   # Uncomment if used in 'content' fields
    HF_TOKEN = os.getenv("HF_TOKEN")

@dataclass
class ScriptArguments:
    data_path_list: List[str] = field(
        default_factory=list, 
        metadata={"help": "Path(s) to the SFT training data directories (Arrow format, expected to contain list-of-dicts for prompt/completion)."}
    )
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the SFT evaluation data directory (Arrow format, list-of-dicts structure)."}
    )
    load_preprocessed_train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to load a fully preprocessed training dataset (Hugging Face disk format, 'text' field ready)."}
    )
    load_preprocessed_eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to load a fully preprocessed evaluation dataset (Hugging Face disk format, 'text' field ready)."}
    )
    save_processed_train_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to save the training dataset after formatting to 'text' field (Hugging Face disk format)."}
    )
    save_processed_eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to save the eval dataset after formatting to 'text' field (Hugging Face disk format)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store Hugging Face datasets library cache files (e.g., for .map())."}
    )
    force_data_processing: bool = field(
        default=False,
        metadata={"help": "Force dataset loading and processing even if `save_processed_..._path` or user preprocessed path exists."}
    )
    prompt_field: str = field(default="prompt", metadata={"help": "Field name in raw data for prompt messages (list of dicts: [{'role': 'user', 'content': ...}])."})
    completion_field: str = field(default="completion", metadata={"help": "Field name in raw data for completion messages (list of dicts: [{'role': 'assistant', 'content': ...}])."})
    # response_template: str = field(default="", metadata={"help": "Response template"}) # Removed, will use tokenizer.apply_chat_template
    model_name_or_path: str = field(default="meta-llama/Llama-3.1-8B-Instruct") # Default to Llama 3
    use_peft: bool = field(default=True, metadata={"help": "Enable PEFT/LoRA fine-tuning."}) # Default to True
    save_merged: bool = field(default=False, metadata={"help": "Merge LoRA adapters and save the full model after training."})
    lora_alpha: float = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_r: int = field(default=8)
    target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj", metadata={"help": "Comma-separated list of LoRA target modules (Llama 3 default)."})
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)
    bnb_4bit_quant_type: str = field(default="nf4") 
    bnb_4bit_compute_dtype: str = field(default="bfloat16")
    bnb_4bit_use_double_quant: bool = field(default=True)
    attn_implementation: Optional[str] = field(default="flash_attention_2", metadata={"help": "Attention implementation to request (e.g., 'flash_attention_2', 'sdpa', 'eager', or None for default)."})
    # batch_size: int = field(default=4) # Removed, use per_device_train_batch_size from TrainingConfig

@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default="saved_models/sft_finetuned_model")
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(default=2048, metadata={"help": "Maximum sequence length for tokenization. Texts will be truncated or padded by SFTTrainer."})
    # ddp_find_unused_parameters is inherited
    dataloader_num_workers: int = field(default=N_CPUS) # Use detected N_CPUS as default
    # dataloader_pin_memory is inherited
    remove_unused_columns: bool = field(default=False) # SFTTrainer can handle this if dataset_text_field is set
    # per_device_train_batch_size, overwrite_output_dir, gradient_accumulation_steps, learning_rate, num_train_epochs are inherited
    # fp16, bf16, logging_steps, save_steps, save_total_limit, lr_scheduler_type, warmup_ratio are inherited

def prepare_dataset(
    data_paths: List[str],
    args: ScriptArguments,
    training_args: TrainingConfig,
    tokenizer: AutoTokenizer, # Tokenizer is needed for apply_chat_template
    is_eval: bool = False
) -> Optional[datasets.Dataset]:
    dataset_type = "evaluation" if is_eval else "training"
    logger.info(f"--- Preparing {dataset_type} dataset ---")

    load_user_preprocessed_path = args.load_preprocessed_eval_dataset_path if is_eval else args.load_preprocessed_train_dataset_path
    if load_user_preprocessed_path and not args.force_data_processing:
        if os.path.exists(load_user_preprocessed_path):
            logger.info(f"Attempting to load user-provided preprocessed {dataset_type} dataset from: {load_user_preprocessed_path}")
            try:
                # This assumes the user-preprocessed data already has the final "text" field
                return datasets.load_from_disk(load_user_preprocessed_path)
            except Exception as e:
                logger.error(f"Failed to load from user-provided preprocessed path {load_user_preprocessed_path}: {e}. Will attempt to process from raw data.", exc_info=True)
        else:
            logger.warning(f"User-provided preprocessed {dataset_type} dataset path not found: {load_user_preprocessed_path}. Will process from raw data.")

    save_script_processed_path = args.save_processed_eval_dataset_path if is_eval else args.save_processed_train_dataset_path
    if save_script_processed_path and os.path.exists(save_script_processed_path) and not args.force_data_processing:
        logger.info(f"Attempting to load script-processed {dataset_type} dataset from cache: {save_script_processed_path}")
        try:
            return datasets.load_from_disk(save_script_processed_path)
        except Exception as e:
            logger.warning(f"Failed to load from script's cache {save_script_processed_path}: {e}. Reprocessing from raw data.", exc_info=True)

    if not data_paths:
        if is_eval:
            logger.info(f"No raw data paths provided for {dataset_type} dataset. Skipping.")
            return None
        raise ValueError(f"No raw data paths specified for {dataset_type} dataset.")

    datasets.enable_progress_bar()
    loaded_data_segments = []
    logger.info(f"Loading raw {dataset_type} data from paths: {data_paths}")
    for path_item in tqdm(data_paths, desc=f"Loading raw {dataset_type} files"):
        if not path_item or not os.path.exists(path_item):
            logger.warning(f"Path '{path_item}' is invalid or does not exist. Skipping.")
            continue
        try:
            if os.path.isdir(path_item):
                segment = datasets.load_from_disk(path_item)
            elif path_item.endswith(('.json', '.jsonl')):
                segment = datasets.load_dataset('json', data_files=path_item, cache_dir=args.dataset_cache_dir, trust_remote_code=True)['train']
            else:
                logger.warning(f"Unsupported file type or invalid path for raw data: {path_item}. Skipping.")
                continue
            
            # Verify that prompt_field and completion_field exist and are lists (of dicts)
            # This basic check can be expanded if needed
            if args.prompt_field not in segment.column_names or args.completion_field not in segment.column_names:
                logger.error(f"Dataset at {path_item} is missing '{args.prompt_field}' or '{args.completion_field}'. Skipping.")
                continue
            # Add more sophisticated checks if needed, e.g., that they are lists of dicts with 'role' and 'content'
            loaded_data_segments.append(segment)
        except Exception as e:
            logger.error(f"Failed to load or process data from {path_item}: {e}", exc_info=True)
    datasets.disable_progress_bar()

    if not loaded_data_segments:
        if is_eval: return None
        raise ValueError(f"No {dataset_type} data could be successfully loaded/processed from paths: {data_paths}")

    raw_dataset = datasets.concatenate_datasets(loaded_data_segments) if len(loaded_data_segments) > 1 else loaded_data_segments[0]
    logger.info(f"Concatenated {len(raw_dataset)} raw samples for {dataset_type}.")

    # No need to save raw examples here as the format is now list-of-dicts, less human-readable directly

    def _formatting_map_function(examples):
        texts = []
        # examples[args.prompt_field] is a list of lists of dicts because of batched=True
        # examples[args.completion_field] is also a list of lists of dicts
        
        for i in range(len(examples[args.prompt_field])):
            prompt_messages = examples[args.prompt_field][i]
            completion_messages = examples[args.completion_field][i]
            
            if not isinstance(prompt_messages, list) or not isinstance(completion_messages, list):
                logger.error(f"Skipping example due to unexpected format. Prompt or completion is not a list. Prompt: {type(prompt_messages)}, Completion: {type(completion_messages)}")
                texts.append(None) # Or handle error appropriately
                continue

            # Combine prompt and completion messages to form the full conversation turn for SFT
            full_conversation_messages = prompt_messages + completion_messages
            
            try:
                formatted_text = tokenizer.apply_chat_template(
                    full_conversation_messages, 
                    tokenize=False, 
                    add_generation_prompt=False # Important for SFT: we want the full turn including assistant's part
                )
                texts.append(formatted_text)
            except Exception as e:
                logger.error(f"Error applying chat template to messages: {full_conversation_messages}. Error: {e}", exc_info=True)
                texts.append(None) # Or handle error

        # Filter out None entries if any errors occurred
        return {"text": [t for t in texts if t is not None]}

    columns_to_remove = [col for col in raw_dataset.column_names if col != args.prompt_field and col != args.completion_field]
    # We will create a new "text" column and then can remove the original prompt/completion fields
    # However, SFTTrainer might be able to directly use a column of message lists if dataset_kwargs is set.
    # For simplicity here, we create the "text" field.
    
    map_cache_file_name = None
    if args.dataset_cache_dir:
        os.makedirs(args.dataset_cache_dir, exist_ok=True)
        path_hash = hash(tuple(sorted(data_paths)))
        # Using tokenizer name/path in hash as chat template can vary
        tokenizer_name_hash = hash(tokenizer.name_or_path)
        relevant_args_str = f"{args.prompt_field}-{args.completion_field}-{path_hash}-{tokenizer_name_hash}"
        cache_hash = hash(relevant_args_str)
        map_cache_file_name = os.path.join(args.dataset_cache_dir, f"sft_formatted_{dataset_type}_{cache_hash}.arrow")
        logger.info(f"Using map cache file for SFT formatting: {map_cache_file_name}")

    processed_dataset = raw_dataset.map(
        _formatting_map_function,
        batched=True, # Process in batches
        remove_columns=[args.prompt_field, args.completion_field] + columns_to_remove, # Remove old fields
        desc=f"Formatting SFT {dataset_type} dataset to 'text' field using chat template",
        cache_file_name=map_cache_file_name,
        load_from_cache_file=not args.force_data_processing if map_cache_file_name else True,
    )
    
    # Filter out examples where formatting might have failed and resulted in None
    original_len = len(processed_dataset)
    processed_dataset = processed_dataset.filter(lambda example: example["text"] is not None)
    if len(processed_dataset) < original_len:
        logger.warning(f"Filtered out {original_len - len(processed_dataset)} examples due to formatting errors.")

    if len(processed_dataset) > 0:
        logger.info(f"Formatted SFT {dataset_type} dataset. Example 'text' entry: {processed_dataset[0]['text'][:500]}...") # Log a snippet
    else:
        logger.warning(f"Formatted SFT {dataset_type} dataset is empty. Check data or formatting logic.")
        return None


    if save_script_processed_path and training_args.local_rank <= 0: # type: ignore
        logger.info(f"Saving processed SFT {dataset_type} dataset to script's cache: {save_script_processed_path}")
        try:
            processed_dataset.save_to_disk(save_script_processed_path)
        except Exception as e:
            logger.error(f"Failed to save processed SFT {dataset_type} dataset to {save_script_processed_path}: {e}", exc_info=True)
    return processed_dataset

def train():
    logger.info("==================================================")
    logger.info("           STARTING SFT TRAINING PIPELINE          ")
    logger.info("==================================================")

    parser = HfArgumentParser((ScriptArguments, TrainingConfig))
    args, training_args = parser.parse_args_into_dataclasses()

    if not args.data_path_list and not args.load_preprocessed_train_dataset_path:
        raise ValueError("FATAL: No training data provided. Specify `data_path_list` (for raw data) or `load_preprocessed_train_dataset_path`.")
    
    logger.info("--- Environment Information ---")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"TRL version: {trl.__version__}")
    logger.info(f"Datasets version: {datasets.__version__}")
    try:
        logger.info(f"PEFT version: {importlib.metadata.version('peft')}")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("PEFT version: Not found (required if use_peft=True)")
    try:
        logger.info(f"Accelerate version: {importlib.metadata.version('accelerate')}")
    except importlib.metadata.PackageNotFoundError:
        logger.warning("Accelerate version: Not found (required by Transformers Trainer)")

    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")
        for i in range(device_count):
            logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"BF16 supported: {is_bf16_supported()}")
    logger.info(f"Dataloader num_workers (from TrainingArguments): {training_args.dataloader_num_workers}")
    
    logger.info(f"Distributed training: {training_args.world_size > 1}")
    if training_args.world_size > 1:
         logger.info(f"Local rank: {training_args.local_rank}, World size: {training_args.world_size}")

    logger.info("--- Configuration ---")
    logger.info(f"Script Arguments (parsed): {args}")
    logger.info(f"Training Arguments (parsed): {training_args}")

    quantization_config = None
    bnb_compute_dtype = None
    if args.bnb_4bit_compute_dtype:
        try:
            bnb_compute_dtype = getattr(torch, args.bnb_4bit_compute_dtype)
        except AttributeError:
            logger.warning(f"Invalid bnb_4bit_compute_dtype: {args.bnb_4bit_compute_dtype}. Defaulting.")
            bnb_compute_dtype = torch.bfloat16 if is_bf16_supported() else torch.float16
    
    if args.load_in_4bit:
        logger.info(f"Loading model in 4-bit quantization with compute_dtype: {bnb_compute_dtype}")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant
        )
    elif args.load_in_8bit:
        logger.info("Loading model in 8-bit quantization")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    
    logger.info(f"--- Loading Tokenizer: {args.model_name_or_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
        token=Constants.HF_TOKEN
    )
    
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer missing <PAD> token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None: # Should be set when pad_token is assigned
            tokenizer.pad_token_id = tokenizer.eos_token_id 
            if tokenizer.pad_token_id is None: # Still None
                 raise ValueError("tokenizer.pad_token_id is still None after setting pad_token to eos_token.")
    
    tokenizer.padding_side = "right" # SFTTrainer default, or "left" if preferred for generation models
    logger.info(f"Tokenizer pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}, padding_side: {tokenizer.padding_side}")

    # Custom special tokens (e.g. <PATH>) are assumed not to be in the 'content' of list-of-dicts data
    # If they are, they should be added here. For now, this logic is simplified.
    added_tokens_count = 0 
    # Example: if you had custom tokens in Constants and needed to add them:
    # custom_tokens_to_add = []
    # if hasattr(Constants, 'PATH_START_TOKEN') and Constants.PATH_START_TOKEN not in tokenizer.get_vocab():
    #     custom_tokens_to_add.append(Constants.PATH_START_TOKEN)
    # if custom_tokens_to_add:
    #     added_tokens_count = tokenizer.add_special_tokens({"additional_special_tokens": custom_tokens_to_add})
    #     logger.info(f"Added {added_tokens_count} custom special token(s). New vocab size: {len(tokenizer)}")


    logger.info(f"--- Loading Model: {args.model_name_or_path} ---")
    model_torch_dtype = None
    if training_args.bf16:
        if is_bf16_supported(): model_torch_dtype = torch.bfloat16
        else: logger.warning("BF16 requested but not supported by hardware. Will use FP32 or FP16 if enabled.")
    if model_torch_dtype is None and training_args.fp16:
        model_torch_dtype = torch.float16
    logger.info(f"Model torch_dtype for loading: {model_torch_dtype or 'torch.float32 (default)'}")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True, "token": Constants.HF_TOKEN, "torch_dtype": model_torch_dtype,
        "device_map": "auto"
    }
    if args.attn_implementation and args.attn_implementation.lower() != "eager":
        model_kwargs["attn_implementation"] = args.attn_implementation
        logger.info(f"Requesting attention implementation: {args.attn_implementation}")
    else:
        logger.info(f"Using model's default attention implementation or 'eager'.")

    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
        logger.info(f"Applying quantization. Device map is '{model_kwargs['device_map']}'.")

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)

    if added_tokens_count > 0:
        logger.info(f"Resizing token embeddings of the model to {len(tokenizer)}.")
        model.resize_token_embeddings(len(tokenizer))
    
    if model.config.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        if tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
            logger.info(f"Set model.config.pad_token_id to tokenizer's: {model.config.pad_token_id}")
        else:
            raise ValueError("tokenizer.pad_token_id is None, critical for model config.")


    if hasattr(model.config, "use_cache") and model.config.use_cache:
        model.config.use_cache = False
        logger.info("Set model.config.use_cache to False for training.")

    peft_config = None
    if args.use_peft:
        logger.info("--- Setting up PEFT (LoRA) ---")
        target_modules_list = [mod.strip() for mod in args.target_modules.split(",") if mod.strip()]
        if not target_modules_list:
             logger.warning(f"target_modules string '{args.target_modules}' resulted in empty list. Using Llama 3 default for LoRA.")
             target_modules_list = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        logger.info(f"LoRA target modules: {target_modules_list}")
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=target_modules_list, bias="none", task_type="CAUSAL_LM",
        )
        logger.info(f"PEFT (LoRA) config: {peft_config}")

    train_dataset = prepare_dataset(
        data_paths=args.data_path_list, args=args, training_args=training_args, tokenizer=tokenizer, is_eval=False
    )
    if train_dataset is None or len(train_dataset) == 0: # Check if dataset is empty after processing
        raise RuntimeError("SFT Training dataset preparation failed or resulted in an empty dataset. Cannot proceed.")
    logger.info(f"SFT Training dataset prepared with {len(train_dataset)} samples.")

    eval_dataset = None
    eval_paths_to_load = []
    if args.eval_dataset_path and os.path.exists(args.eval_dataset_path):
        eval_paths_to_load.append(args.eval_dataset_path)
    elif args.eval_dataset_path:
        logger.warning(f"Raw evaluation data path specified but not found: {args.eval_dataset_path}")

    if args.load_preprocessed_eval_dataset_path or eval_paths_to_load:
         eval_dataset = prepare_dataset(
            data_paths=eval_paths_to_load, args=args, training_args=training_args, tokenizer=tokenizer, is_eval=True
        )
    
    if eval_dataset and len(eval_dataset) > 0: # Check if eval_dataset is not None and not empty
        logger.info(f"SFT Evaluation dataset prepared with {len(eval_dataset)} samples.")
        training_args.do_eval = True
    else:
        if args.eval_dataset_path or args.load_preprocessed_eval_dataset_path: # Log if paths were given but resulted in no data
            logger.warning("Evaluation dataset specified but resulted in an empty or None dataset after processing. Disabling evaluation.")
        else:
            logger.info("No evaluation dataset path provided. Skipping evaluation.")
        training_args.do_eval = False


    logger.info("--- Initializing SFTTrainer ---")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer, # Changed from processing_class
        peft_config=peft_config if args.use_peft else None,
        dataset_text_field="text", # `prepare_dataset` creates this field using chat_template
        max_seq_length=training_args.max_seq_length,
        # packing=True, # Consider if you have many short sequences
    )

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    effective_train_batch_size = (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size)
    logger.info(f"Effective global training batch size: {effective_train_batch_size}")
    trainer_max_seq_len = getattr(trainer, 'max_seq_length', training_args.max_seq_length)
    logger.info(f"Max sequence length for SFTTrainer: {trainer_max_seq_len}")

    logger.info("--- Handling Checkpoints ---")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and any(os.path.isdir(os.path.join(training_args.output_dir, item)) for item in os.listdir(training_args.output_dir)):
             if any(item.startswith("checkpoint-") for item in os.listdir(training_args.output_dir)):
                logger.warning(f"Output dir ({training_args.output_dir}) exists, not empty, has checkpoint-like folders but get_last_checkpoint failed. Check dir structure or use --overwrite_output_dir.")
             else:
                logger.info(f"Output dir ('{training_args.output_dir}') exists, not empty, no checkpoints. Overwriting possible if script continues.")
        elif last_checkpoint is not None:
            logger.info(f"Checkpoint detected! Resuming training from: {last_checkpoint}")
    checkpoint_to_resume = training_args.resume_from_checkpoint or last_checkpoint

    logger.info("--- Starting SFT Training ---")
    if training_args.world_size > 1 and checkpoint_to_resume is None: # type: ignore
        logger.info("Distributed training: Barrier before initial training start.")
        torch.distributed.barrier() # type: ignore

    train_result = trainer.train(resume_from_checkpoint=checkpoint_to_resume)

    if trainer.is_world_process_zero():
        logger.info("--- Saving Final Model and Tokenizer ---")
        trainer.save_model(training_args.output_dir) 
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Model/adapters and tokenizer saved to {training_args.output_dir}")

        logger.info("--- Logging and Saving Metrics ---")
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset) if train_dataset else 0
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("Training metrics and state saved.")

    if args.use_peft and args.save_merged:
        if training_args.world_size > 1: # type: ignore
            logger.info("Distributed training: Barrier before merging PEFT adapters.")
            torch.distributed.barrier() # type: ignore
        if trainer.is_world_process_zero():
            logger.info("--- Merging LoRA Adapters and Saving Full Model ---")
            merged_model_path = os.path.join(training_args.output_dir, "merged_model")
            os.makedirs(merged_model_path, exist_ok=True)
            try:
                logger.info(f"Reloading base model '{args.model_name_or_path}' (unquantized) for merging.")
                base_model_dtype_for_merge = torch.bfloat16 if is_bf16_supported() and training_args.bf16 else \
                                             (torch.float16 if training_args.fp16 else torch.float32)
                
                base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=base_model_dtype_for_merge,
                    trust_remote_code=True, token=Constants.HF_TOKEN, device_map="cpu",
                    quantization_config=None # Ensure no quantization for merge base
                )
                if added_tokens_count > 0:
                    base_model_for_merge.resize_token_embeddings(len(tokenizer))

                logger.info(f"Loading trained PEFT adapters from {training_args.output_dir}.")
                merged_model = PeftModel.from_pretrained(base_model_for_merge, training_args.output_dir, is_trainable=False)
                logger.info("Merging adapters into base model...")
                merged_model = merged_model.merge_and_unload()
                
                logger.info(f"Saving merged model to {merged_model_path}")
                merged_model.save_pretrained(merged_model_path)
                tokenizer.save_pretrained(merged_model_path)
                logger.info(f"Successfully saved merged model and tokenizer to {merged_model_path}")
            except Exception as e:
                logger.error(f"Error during model merging and saving: {e}", exc_info=True)

    logger.info("==================================================")
    logger.info("        SFT TRAINING PIPELINE COMPLETED           ")
    logger.info("==================================================")

if __name__ == "__main__":
    train()
