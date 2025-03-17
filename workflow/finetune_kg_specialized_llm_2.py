import os
import torch
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import torch.utils.data
import datasets
import dotenv
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer, 
    HfArgumentParser,
    TrainingArguments,
    trainer_utils
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
from peft import LoraConfig
from src import utils
from src.knowledge_graph import KnowledgeGraph
from src.template import Template
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
from tqdm import tqdm

# 配置
dotenv.load_dotenv()
datasets.disable_progress_bar()

# 常量定义
class Constants:
    PATH_START_TOKEN = "<PATH>"
    PATH_END_TOKEN = "</PATH>"
    HF_TOKEN = os.getenv("HF_TOKEN")
    N_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))

def input_formatter(example, tokenizer):
    question = example["question"]
    q_entity = example["q_entity"]
    a_entity = example["a_entity"]
    golden_path = example["golden_path"]
    positive_path = example["positive_path"]
    negative_paths = example["negative_paths"]

    if not question.endswith("?"):
        question += "?"
    raw_input = Template.ZERO_SHOT_PROMPT.format(question=question, entity=q_entity)
    
    # 将处理后的路径用标签包装
    golden_path_string = " ".join([f"{Constants.PATH_START_TOKEN}{golden_path}{Constants.PATH_END_TOKEN}"])
    positive_path_string = " ".join([f"{Constants.PATH_START_TOKEN}{positive_path}{Constants.PATH_END_TOKEN}"])
    negative_paths_string = [" ".join([f"{Constants.PATH_START_TOKEN}{path}{Constants.PATH_END_TOKEN}" for path in negative_paths])]
    return {"text": raw_input, "golden_path": golden_path_string, "positive_path": positive_path_string, "negative_paths": negative_paths_string}

@dataclass
class ScriptArguments:  
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", 
        metadata={"help": "the model name"}
    )
    encode_model_name: Optional[str] = field(
        default="msmarco-distilbert-base-tas-b",
        metadata={"help": "the encode model name"}
    )
    neo4j_uri: Optional[str] = field(
        default="bolt://localhost:7687",
        metadata={"help": "Neo4j URI"}
    )
    neo4j_user: Optional[str] = field(
        default="neo4j",
        metadata={"help": "Neo4j user"}
    )
    neo4j_password: Optional[str] = field(
        default="Martin1007Wang",
        metadata={"help": "Neo4j password"}
    )
    use_peft: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to use PEFT or not to train adapters"},
    )
    save_merged: Optional[bool] = field(
        default=False, 
        metadata={"help": "Wether to save merged model"}
    )
    lora_alpha: Optional[float] = field(
        default=16,
        metadata={"help": "the lora alpha parameter"}
    )
    lora_dropout: Optional[float] = field(
        default=0.05, 
        metadata={"help": "the lora dropout parameter"}
    )
    lora_r: Optional[int] = field(
        default=8, 
        metadata={"help": "the lora r parameter"}
    )
    similarity_threshold: Optional[float] = field(
        default=0.5,
        metadata={"help": "Threshold for relation similarity scores"}
    )
    max_relations_per_entity: Optional[int] = field(
        default=5,
        metadata={"help": "Maximum number of relations to consider per entity"}
    )
    load_in_4bit: bool = field(
        default=False, 
        metadata={"help": "Load model in 4bit"}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load model in 8bit"}
    )
    attn_implementation: Optional[str] = field(
        default="flash_attention_2",
        metadata={"help": "attn implementation"}
    )
    response_template: Optional[str] = field(
        default="[/INST]",
        metadata={"help": "Response template"}
    )

@dataclass
class ScriptTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default="saved_models/llama2_align",
        metadata={"help": "The output directory"},
    )
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ddp_find_unused_parameters: bool = field(default=False)
    dataloader_num_workers: int = field(default=Constants.N_CPUS)

class Trainer:
    @staticmethod
    def train():
        """训练主函数"""
        parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
        script_args, training_args = parser.parse_args_into_dataclasses()
        kg = KnowledgeGraph(
            uri=script_args.neo4j_uri,
            user=script_args.neo4j_user,
            password=script_args.neo4j_password,
            model_name=script_args.encode_model_name
        )
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            trust_remote_code=True,
            token=Constants.HF_TOKEN,
            torch_dtype=torch.bfloat16,
            attn_implementation=script_args.attn_implementation,
            load_in_4bit=script_args.load_in_4bit,
            load_in_8bit=script_args.load_in_8bit
        )
        model.config.use_cache = False

        peft_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ) if script_args.use_peft else None
        
        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
            token=Constants.HF_TOKEN,
        )
        tokenizer.padding_side = "right"
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = '<PAD>'
        special_tokens_dict['additional_special_tokens'] = [Constants.PATH_START_TOKEN, Constants.PATH_END_TOKEN]
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        
        data_list = [
            datasets.load_from_disk(data_path) for data_path in script_args.data_path_list
        ]
        dataset = datasets.concatenate_datasets(data_list)
        dataset = dataset.map(
            input_formatter,
            fn_kwargs={"tokenizer": tokenizer},
            remove_columns=["question", "q_entity", "a_entity", "golden_path","positive_path","negative_paths"]
        )

    @staticmethod
    def _get_checkpoint(training_args):
        """获取checkpoint"""
        checkpoint = None
        if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
            last_checkpoint = trainer_utils.get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logging.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )
                checkpoint = last_checkpoint
        
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
            
        return checkpoint

if __name__ == "__main__":
    Trainer.train()
