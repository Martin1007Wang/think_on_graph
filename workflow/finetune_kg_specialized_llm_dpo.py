import os
import torch
import logging
from dataclasses import dataclass, field
from typing import List
import torch.utils.data
import datasets
import dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DPOTrainer, DPOConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from src.template import KnowledgeGraphTemplates
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint

dotenv.load_dotenv()
datasets.disable_progress_bar()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Constants:
    PATH_START_TOKEN = "<PATH>"
    PATH_END_TOKEN = "</PATH>"
    HF_TOKEN = os.getenv("HF_TOKEN")
    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

@dataclass
class ScriptArguments:
    data_path_list: List[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: str = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "The model name"})
    use_peft: bool = field(default=False,metadata={"help": "Whether to use PEFT or not to train adapters"},)
    save_merged: bool = field(default=False, metadata={"help": "Whether to save merged model"})
    lora_alpha: float = field(default=16,metadata={"help": "The lora alpha parameter"})
    lora_dropout: float = field(default=0.05, metadata={"help": "The lora dropout parameter"})
    lora_r: int = field(default=8, metadata={"help": "The lora r parameter"})
    max_neg_paths: int = field(default=3,metadata={"help": "Maximum number of negative paths to consider per example"})
    load_in_4bit: bool = field(default=False, metadata={"help": "Load model in 4bit"})
    load_in_8bit: bool = field(default=False,metadata={"help": "Load model in 8bit"})
    attn_implementation: str = field(default="eager",metadata={"help": "Attention implementation (eager, flash_attention_2)"})
    beta: float = field(default=0.1,metadata={"help": "The beta parameter for DPO"})
    loss_type: str = field(default="sigmoid",metadata={"help": "Loss type for DPO (sigmoid, hinge, ipo, simpo)"})
    dpo_alpha: float = field(default=0.1,metadata={"help": "The alpha parameter for DPO-SimPO"})
    batch_size: int = field(default=4, metadata={"help": "Batch size for training"})
    response_template: str = field(default="", metadata={"help": "Response template"})
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    max_prompt_length: int = field(default=256, metadata={"help": "Maximum prompt length"})
    reference_free: bool = field(default=False, metadata={"help": "Use reference-free mode for DPO training"})
    label_smoothing: float = field(default=0.0, metadata={"help": "Label smoothing factor for DPO training"})
    eval_dataset_path: str = field(default="", metadata={"help": "Path to evaluation dataset (optional)"})
    precompute_ref_log_probs: bool = field(default=False, metadata={"help": "Precompute reference model log probs"})
    generate_during_eval: bool = field(default=False, metadata={"help": "Generate during evaluation"})
    
@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default="saved_models/llama2_align",metadata={"help": "The output directory"},)
    optim: str = field(default="adamw_torch")
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=Constants.N_CPUS)
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"})
    remove_unused_columns: bool = field(default=False)
    per_device_train_batch_size: int = field(default=4)
    overwrite_output_dir: bool = field(default=True)
    gradient_accumulation_steps: int = field(default=16)
    learning_rate: float = field(default=5e-5)
    num_train_epochs: int = field(default=3)
    fp16: bool = field(default=True)
    logging_steps: int = field(default=10)
    lr_scheduler_type: str = field(default="cosine")
    warmup_ratio: float = field(default=0.05)

def prepare_preference_dataset(data_paths, tokenizer, args):
    logger.info(f"Preparing DPO dataset from {len(data_paths)} sources")
    data_list = [datasets.load_from_disk(path) for path in tqdm(data_paths, desc="Loading datasets")]
    dataset = datasets.concatenate_datasets(data_list)
    prompt_template = KnowledgeGraphTemplates.ZERO_SHOT_PROMPT
    semantic_template = KnowledgeGraphTemplates.SEMANTIC_PATH_TEMPLATE
    shortest_template = KnowledgeGraphTemplates.SHORTEST_PATH_TEMPLATE
    negative_template = KnowledgeGraphTemplates.NEGATIVE_PATH_TEMPLATE
    all_samples = []
    def input_formatter(example):
        question = example.get("question", "")
        q_entity = example.get("q_entity", "")
        a_entity = example.get("a_entity", "")
        prompt = prompt_template.format(question=question, entity=q_entity)
        paths = {
            'shortest': f"{Constants.PATH_START_TOKEN}{example['golden_path']}{Constants.PATH_END_TOKEN}",
            'semantic': f"{Constants.PATH_START_TOKEN}{example['positive_path']}{Constants.PATH_END_TOKEN}",
            'negative': [f"{Constants.PATH_START_TOKEN}{path}{Constants.PATH_END_TOKEN}" for path in example.get("negative_paths", [])]
        }
        responses = {
            'semantic': semantic_template.format(reasoning_path=paths['semantic'], answer=a_entity),
            'shortest': shortest_template.format(reasoning_path=paths['shortest'], answer=a_entity)
        }
        chat_templates = {
            'semantic': [{"role": "user", "content": prompt}, {"role": "assistant", "content": responses['semantic']}],
            'shortest': [{"role": "user", "content": prompt}, {"role": "assistant", "content": responses['shortest']}]
        }
        texts = {
            'semantic': tokenizer.apply_chat_template(chat_templates['semantic'], tokenize=False, add_generation_prompt=False),
            'shortest': tokenizer.apply_chat_template(chat_templates['shortest'], tokenize=False, add_generation_prompt=False)
        }
        samples = []
        for neg_path in paths['negative']:
            neg_response = negative_template.format(reasoning_path=neg_path, answer="Cannot determine")
            neg_chat = [{"role": "user", "content": prompt}, {"role": "assistant", "content": neg_response}]
            neg_text = tokenizer.apply_chat_template(neg_chat, tokenize=False, add_generation_prompt=False)
            samples.append({
                "prompt": prompt, 
                "chosen": texts['semantic'], 
                "rejected": neg_text
            })
            samples.append({
                "prompt": prompt, 
                "chosen": texts['shortest'], 
                "rejected": neg_text
            })
        return samples
    for i, example in enumerate(tqdm(dataset, desc="Processing samples")):
        samples = input_formatter(example)
        all_samples.extend(samples)
    preference_dataset = datasets.Dataset.from_dict({
        "prompt": [sample["prompt"] for sample in all_samples],
        "chosen": [sample["chosen"] for sample in all_samples],
        "rejected": [sample["rejected"] for sample in all_samples]
    })
    return preference_dataset

def train():
    logger.info(f"Starting training pipeline")
    parser = HfArgumentParser((ScriptArguments, TrainingConfig))
    args, training_args = parser.parse_args_into_dataclasses()
    
    # 添加更详细的GPU检测和环境变量检查
    logger.info("Environment information:")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    device_count = torch.cuda.device_count()
    logger.info(f"CUDA devices: {device_count}")
    for i in range(device_count):
        logger.info(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    logger.info(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'Not set')}")
    logger.info(f"RANK: {os.environ.get('RANK', 'Not set')}")
    logger.info(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'Not set')}")
    
    # 如果只有一个GPU但尝试使用分布式训练，发出警告并调整设置
    if device_count == 1 and training_args.local_rank != -1:
        logger.warning("Only one GPU detected but attempting distributed training. "
                       "Falling back to non-distributed mode.")
        # 禁用分布式功能
        os.environ["LOCAL_RANK"] = "-1"
        training_args.local_rank = -1
        training_args.ddp_backend = None
    
    # 确保分布式训练只在有足够GPU的情况下进行
    if training_args.local_rank != -1:
        if device_count < 2:
            logger.warning("Not enough GPUs for distributed training. Falling back to single GPU.")
            training_args.local_rank = -1
        else:
            # 确保正确分配设备
            local_rank = training_args.local_rank
            gpu_to_use = local_rank % device_count
            torch.cuda.set_device(gpu_to_use)
            logger.info(f"Process with rank {local_rank} using GPU: {gpu_to_use}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "token": Constants.HF_TOKEN,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": args.attn_implementation,
        "load_in_4bit": args.load_in_4bit,
        "load_in_8bit": args.load_in_8bit,
    }
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if args.attn_implementation == "flash_attention_2":
        model = model.to(training_args.device)
    model.config.use_cache = False
    if args.use_peft:
        logger.info(f"Initializing LoRA adapter")
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none", task_type="CAUSAL_LM",
        )
        logger.info(f"LoRA adapter initialized with rank {args.lora_r}")
    else:
        peft_config = None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False, token=Constants.HF_TOKEN)
    tokenizer.padding_side = "right"
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "<PAD>"
    special_tokens_dict['additional_special_tokens'] = [Constants.PATH_START_TOKEN, Constants.PATH_END_TOKEN]
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    DPO_dataset = prepare_preference_dataset(args.data_path_list, tokenizer, args)
    
    # 修改分布式同步代码
    if training_args.local_rank != -1 and device_count > 1:
        try:
            logger.info(f"Process {training_args.local_rank} waiting at barrier")
            torch.distributed.barrier()
            logger.info(f"Process {training_args.local_rank} passed data preparation barrier")
        except Exception as e:
            logger.warning(f"Error in distributed barrier: {e}. Continuing without barrier.")
    
    # Add reference model for DPO
    ref_model = None
    if not args.reference_free:
        logger.info(f"Loading reference model from {args.model_name_or_path}")
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            **model_kwargs
        )
        ref_model.config.use_cache = False
    
    # Load evaluation dataset if provided
    eval_dataset = None
    if args.eval_dataset_path:
        logger.info(f"Loading evaluation dataset from {args.eval_dataset_path}")
        eval_dataset = datasets.load_from_disk(args.eval_dataset_path)
        eval_dataset = prepare_preference_dataset([args.eval_dataset_path], tokenizer, args)
    
    # Use response template for data collator if provided
    data_collator = None
    if args.response_template:
        data_collator = DataCollatorForCompletionOnlyLM(
            args.response_template, tokenizer=tokenizer, mlm=False
        )
    trainer_config = DPOConfig(
        **training_args.to_dict(),
        loss_type=args.loss_type,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        padding_value=tokenizer.pad_token_id,
        reference_free=args.reference_free,
        label_smoothing=args.label_smoothing,
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        generate_during_eval=args.generate_during_eval,
    )
    
    # 类似地修改第二个barrier
    if training_args.local_rank != -1 and device_count > 1:
        try:
            torch.distributed.barrier()
            logger.info(f"Process {training_args.local_rank} ready to start training")
        except Exception as e:
            logger.warning(f"Error in second distributed barrier: {e}. Continuing without barrier.")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=trainer_config,
        train_dataset=DPO_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        # data_collator=data_collator,
    )
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                            "Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logging.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                        "the `--output_dir` or add `--overwrite_output_dir` to train from scratch.")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    
    # Add post-training merge and save for PEFT models
    if args.use_peft and args.save_merged:
        logger.info("Saving merged model...")
        trainer.model.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))
        tokenizer.save_pretrained(os.path.join(training_args.output_dir, "merged_model"))
    
    return trainer.model, tokenizer  # Return model and tokenizer for potential use

if __name__ == "__main__":
    train()