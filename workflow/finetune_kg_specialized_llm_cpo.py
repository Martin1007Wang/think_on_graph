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
from trl import CPOTrainer, CPOConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from src.template import Template
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
    beta: float = field(default=0.1,metadata={"help": "The beta parameter for CPO"})
    loss_type: str = field(default="simpo",metadata={"help": "Loss type for CPO (sigmoid, hinge, ipo, simpo)"})
    cpo_alpha: float = field(default=0.1,metadata={"help": "The alpha parameter for CPO-SimPO"})
    batch_size: int = field(default=4, metadata={"help": "Batch size for training"})
    response_template: str = field(default="", metadata={"help": "Response template"})
    max_length: int = field(default=512, metadata={"help": "Maximum sequence length"})
    max_prompt_length: int = field(default=256, metadata={"help": "Maximum prompt length"})
    
@dataclass
class TrainingConfig(TrainingArguments):
    output_dir: str = field(default="saved_models/llama2_align",metadata={"help": "The output directory"},)
    optim: str = field(default="adamw_torch")
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=Constants.N_CPUS)
    local_rank: int = field(default=-1, metadata={"help": "Local rank for distributed training"})
    remove_unused_columns: bool = field(default=False)
    per_device_train_batch_size: int = field(default=4)

def prepare_cpo_dataset(data_paths, tokenizer, args):
    logger.info(f"Preparing CPO dataset from {len(data_paths)} sources")
    data_list = [datasets.load_from_disk(path) for path in tqdm(data_paths, desc="Loading datasets")]
    dataset = datasets.concatenate_datasets(data_list)
    prompt_template = Template.ZERO_SHOT_PROMPT
    semantic_template = Template.SEMANTIC_PATH_TEMPLATE
    shortest_template = Template.SHORTEST_PATH_TEMPLATE
    negative_template = Template.NEGATIVE_PATH_TEMPLATE
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
    cpo_dataset = prepare_cpo_dataset(args.data_path_list, tokenizer, args)
    if training_args.local_rank != -1:
        torch.distributed.barrier()
        logger.info(f"Process {training_args.local_rank} passed data preparation barrier")
    trainer_config = CPOConfig(
        **training_args.to_dict(),
        loss_type=args.loss_type,
        cpo_alpha=args.cpo_alpha,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        padding_value=tokenizer.pad_token_id,
    )
    if training_args.local_rank != -1:
        torch.distributed.barrier()
        logger.info(f"Process {training_args.local_rank} passed trainer config barrier")
    # data_collator = DataCollatorForCompletionOnlyLM(
    #     args.response_template, tokenizer=tokenizer, mlm=False
    # )
    trainer = CPOTrainer(
        model=model,
        args=trainer_config,
        train_dataset=cpo_dataset,
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
    if training_args.local_rank != -1:
        torch.distributed.barrier()
        logger.info(f"Process {training_args.local_rank} ready to start training")
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    
if __name__ == "__main__":
    train()