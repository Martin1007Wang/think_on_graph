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
from workflow.build_knowledge_graph import KnowledgeGraph
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

# 提示模板
class Templates:
    ZERO_SHOT_PROMPT = """Given a question and topic entities, you need to select the most relevant relations from the available relations to find the answer. Consider which relations would lead you to the answer entity.
                        # Question: 
                        {question}
                        # Topic entities: 
                        {entities}
                        # Available Relations: 
                        {relations}"""

    RELATION_SELECTION_TEMPLATE = """# Selected Relations (ordered by relevance):
                        {selected_relations}
                        # Explanation:
                        These relations are most relevant because they {explanation}"""

    PATH_TEMPLATE = """# Reasoning Path:
                        {reasoning_path}
                        # Answer:
                        {answer}"""

@dataclass
class ScriptArguments:
    data_path_list: list[str] = field(metadata={"help": "Path to the training data."})
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf", 
        metadata={"help": "the model name"}
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

class ModelManager:
    @staticmethod
    def setup_model_and_tokenizer(script_args):
        """设置模型和分词器"""
        model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            trust_remote_code=True,
            token=Constants.HF_TOKEN,
            torch_dtype=torch.bfloat16,
            attn_implementation=script_args.attn_implementation,
            load_in_4bit=script_args.load_in_4bit,
            load_in_8bit=script_args.load_in_8bit,
        )
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(
            script_args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
            token=Constants.HF_TOKEN,
        )
        
        # 配置tokenizer
        tokenizer.padding_side = "right"
        special_tokens_dict = {}
        if tokenizer.pad_token is None:
            special_tokens_dict['pad_token'] = '<PAD>'
        special_tokens_dict['additional_special_tokens'] = [Constants.PATH_START_TOKEN, Constants.PATH_END_TOKEN]
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        
        return model, tokenizer

class DatasetManager:
    @staticmethod
    def prepare_dataset(data_path_list, tokenizer, kg: KnowledgeGraph):
        """准备数据集"""
        # 初始化sentence transformer模型（只初始化一次）
        similarity_model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')
        
        @lru_cache(maxsize=10000)
        def get_text_embedding(text: str) -> torch.Tensor:
            """缓存文本嵌入"""
            return similarity_model.encode(text, convert_to_tensor=True)
        
        def get_relation_similarity_scores(question: str, relations: List[str]) -> List[Tuple[str, float]]:
            """计算关系与问题的相似度分数"""
            if not relations:
                return []
            
            # 预处理关系并获取嵌入
            processed_relations = [rel.replace('_', ' ').lower() for rel in relations]
            question_embedding = get_text_embedding(question)
            relation_embeddings = torch.stack([get_text_embedding(rel) for rel in processed_relations])
            
            # 计算余弦相似度
            cosine_scores = util.pytorch_cos_sim(question_embedding, relation_embeddings)[0]
            
            # 返回关系和分数对
            return [(rel, score.item()) for rel, score in zip(relations, cosine_scores)]

        def input_formatter(example):
            chunks = []
            
            for i in tqdm(range(len(example["q_entity"])), desc="Processing examples", leave=False):
                question = example["question"][i]
                start_node = example["q_entity"][i]
                ground_paths = example["ground_truth_paths"][i]
                
                if not question.endswith("?"):
                    question += "?"
                
                # 获取所有可用关系
                connected_relations = []
                for entity in start_node:
                    relations = kg.get_connected_relations(entity)
                    connected_relations.extend(relations)
                connected_relations = list(set(connected_relations))  # 去重
                
                # 计算关系相似度分数
                relation_scores = get_relation_similarity_scores(question, connected_relations)
                relation_scores.sort(key=lambda x: x[1], reverse=True)  # 按相似度降序排序
                
                # 准备训练样本
                relations_str = ", ".join(connected_relations)
                raw_input = Templates.ZERO_SHOT_PROMPT.format(
                    question=question, 
                    entities=",".join(start_node),
                    relations=relations_str
                )
                
                if len(ground_paths) > 0:
                    for path in ground_paths:
                        if len(path) == 0:
                            continue
                            
                        # 获取路径中使用的关系
                        path_relations = [triple[1] for triple in path]
                        
                        # 找出这些关系在相似度排序中的位置
                        top_similar_relations = [rel for rel, _ in relation_scores[:5]]
                        matching_relations = [rel for rel in path_relations if rel in top_similar_relations]
                        
                        # 生成关系选择响应
                        selected_relations = "\n".join([f"{i+1}. {rel} (used in correct path)" 
                                                      if rel in path_relations 
                                                      else f"{i+1}. {rel}"
                                                      for i, rel in enumerate(top_similar_relations)])
                        
                        explanation = "directly connect to the target entity" if len(path) == 1 else \
                                    "form a path to the target entity through intermediate nodes"
                        
                        relation_response = Templates.RELATION_SELECTION_TEMPLATE.format(
                            selected_relations=selected_relations,
                            explanation=explanation
                        )
                        
                        # 生成路径响应
                        ground_path_string = f"{Constants.PATH_START_TOKEN}{utils.path_to_string(path)}{Constants.PATH_END_TOKEN}"
                        path_answer = path[-1][-1].strip()
                        path_response = Templates.PATH_TEMPLATE.format(
                            reasoning_path=ground_path_string,
                            answer=path_answer
                        )
                        
                        # 组合完整对话
                        chat = [
                            {"role": "user", "content": raw_input},
                            {"role": "assistant", "content": relation_response + "\n" + path_response},
                        ]
                        
                        final_input = tokenizer.apply_chat_template(
                            chat,
                            tokenize=False,
                            add_generation_prompt=False
                        )
                        chunks.append(final_input)
                        
            return {"text": chunks}

        # 加载并处理数据集
        print("Loading datasets...")
        data_list = [datasets.load_from_disk(path) for path in data_path_list]
        dataset = datasets.concatenate_datasets(data_list)
        
        print("Processing dataset...")
        train_dataset = dataset.map(
            input_formatter,
            batched=True,
            batch_size=32,  # 设置合适的批处理大小
            remove_columns=dataset.column_names,
            num_proc=Constants.N_CPUS,
            desc="Processing dataset"
        )
        
        return train_dataset

class Trainer:
    @staticmethod
    def train():
        """训练主函数"""
        parser = HfArgumentParser((ScriptArguments, ScriptTrainingArguments))
        script_args, training_args = parser.parse_args_into_dataclasses()
        
        # 初始化 KnowledgeGraph
        kg = KnowledgeGraph(
            uri=script_args.neo4j_uri,
            user=script_args.neo4j_user,
            password=script_args.neo4j_password
        )
        
        try:
            # 设置模型和tokenizer
            model, tokenizer = ModelManager.setup_model_and_tokenizer(script_args)
            
            # 准备数据集，传入 kg 实例
            train_dataset = DatasetManager.prepare_dataset(
                script_args.data_path_list, 
                tokenizer,
                kg
            )
            print(train_dataset[0])
            
            # 配置PEFT
            peft_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ) if script_args.use_peft else None

            # 配置训练器
            data_collator = DataCollatorForCompletionOnlyLM(
                script_args.response_template, 
                tokenizer=tokenizer, 
                mlm=False
            )
            
            sft_cfg = SFTConfig(
                **training_args.to_dict(),
                dataset_text_field="text",
                packing=False,
                dataset_kwargs={"add_special_tokens": False},
            )
            
            trainer = SFTTrainer(
                model=model,
                train_dataset=train_dataset,
                peft_config=peft_config,
                tokenizer=tokenizer,
                args=sft_cfg,
                data_collator=data_collator,
            )

            # 检查checkpoint
            checkpoint = Trainer._get_checkpoint(training_args)
            
            # 开始训练
            trainer.train(resume_from_checkpoint=checkpoint)
            trainer.save_model(training_args.output_dir)

        finally:
            # 确保在训练结束后关闭数据库连接
            kg.close()

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
