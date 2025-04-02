import os
import argparse
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import asyncio
from tqdm import tqdm
import json
import logging
from functools import partial, lru_cache
from multiprocessing import Pool
from datasets import load_dataset
import time

from src.llms import get_registed_model
from src.utils.qa_utils import eval_path_result_w_ans
from src.knowledge_graph import KnowledgeGraph
from src.knowledge_explorer import KnowledgeExplorer
# 设置环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 日志设置
def setup_logging(name: str = "kg_qa", level: int = logging.INFO) -> logging.Logger:
    """设置日志记录器。

    Args:
        name: 日志记录器名称
        level: 日志级别

    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # 防止重复添加处理器
        return logger
        
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logging()

# 提示模板集中管理
class PromptTemplates:
    """管理系统中使用的所有提示模板。"""
    
    RELATION_SELECTION = """
You are a knowledge graph reasoning expert. Given a question and a topic entity, your task is to select the most relevant relations to explore from the provided list.

# Question: 
{question}

# Topic entity: 
{entity}

# Available relations from this entity (select only from these):
{relations}

Select exactly {relation_k} relations that are most relevant to answering the question. Your response must follow this exact format, with no additional text outside the numbered list:
1. [relation_name] - [brief explanation of relevance]
2. [relation_name] - [brief explanation of relevance]
...
{relation_k}. [relation_name] - [brief explanation of relevance]

- Only choose relations from the provided list.
- If fewer than {relation_k} relations are relevant, repeat the most relevant relation to fill the list.
- Do not include any introductory text, conclusions, or extra lines beyond the {relation_k} numbered items.
"""

    ENTITY_RANKING = """
    You are a knowledge graph reasoning expert. Given a question and a set of already explored entities, 
    rank the candidate entities by their relevance to answering the question.
    
    # Question: 
    {question}
    
    # Already explored entities: 
    {explored}
    
    # Candidate entities to evaluate:
    {candidates}
    
    For each candidate entity, assign a relevance score from 1-10 (10 being most relevant) based on:
    1. Direct relevance to the question
    2. Potential to connect to relevant information
    3. Uniqueness compared to already explored entities
    
    Format your response as:
    [Entity]: [Score] - [Brief justification]
    """
    
    RELATION_SELECTION_WITH_CONTEXT = """
    You are a knowledge graph reasoning expert. Given a question, a topic entity, and the exploration history so far,
    select the most promising relations to explore next.
    
    # Question: 
    {question}
    
    # Current entity to expand: 
    {entity}
    
    # Exploration history so far:
    {history}
    
    # Available relations from this entity (select only from these):
    {relations}
    
    Select exactly {relation_k} relations that are most likely to lead to the answer. Your response must follow this exact format:
    1. [relation_name] - [brief explanation of relevance]
    2. [relation_name] - [brief explanation of relevance]
    ...
    
    Consider:
    - Which relations might connect to information needed to answer the question
    - Avoid relations that would lead to already explored paths
    - Prioritize relations that fill gaps in the current knowledge
    """
    
    REASONING = """
    You are a knowledge graph reasoning expert. Given a question and information gathered from a knowledge graph, determine if you can answer the question.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Knowledge graph exploration (over {num_rounds} rounds):
    {exploration_history}

    Based on the information above, can you answer the question? Respond in this exact format, with no additional text outside the specified sections:

    [Decision: Yes/No]
    [Answer: your answer if Yes, otherwise leave blank]
    [Reasoning path: specify the exact path of relations and entities if Yes, otherwise leave blank]
    [Missing information: specify what additional relations or entities are needed if No, otherwise leave blank]
    """

    FINAL_ANSWER = """
    You are a knowledge graph reasoning expert. Based on the exploration of the knowledge graph, provide a final answer to the question.

    # Question: 
    {question}

    # Starting entities: 
    {entity}

    # Complete exploration:
    {full_exploration}

    Provide your final answer in this exact format, with no additional text outside the specified sections:
    [Final Answer: your concise answer to the question]
    [Reasoning: brief explanation of how the answer was derived from the exploration]
    """
    
class FileHandler:
    """处理文件读写操作。"""
    
    @staticmethod
    def get_output_file(path: str, force: bool = False) -> Tuple[Any, List[str]]:
        """获取输出文件和已处理的问题ID列表。

        Args:
            path: 输出文件路径
            force: 是否强制覆盖现有文件

        Returns:
            文件对象和已处理的问题ID列表
        """
        if not os.path.exists(path) or force:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fout = open(path, "w")
            return fout, []
        else:
            processed_results = []
            try:
                with open(path, "r") as f:
                    for line in f:
                        try:
                            results = json.loads(line)
                            processed_results.append(results["id"])
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line in {path}")
                            continue
            except Exception as e:
                logger.error(f"Error reading file {path}: {str(e)}")
                return open(path, "w"), []
                
            return open(path, "a"), processed_results

    @staticmethod
    def save_args(output_dir: str, args: argparse.Namespace) -> None:
        """保存参数到文件。

        Args:
            output_dir: 输出目录
            args: 参数命名空间
        """
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
            
def main(args: argparse.Namespace) -> None:
    """主函数，协调整个执行流程。

    Args:
        args: 命令行参数
    """
    # 初始化知识图谱
    logger.info("Initializing knowledge graph...")
    kg = KnowledgeGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        model_name=args.embedding_model
    )
    
    try:
        # 加载数据集
        logger.info(f"Loading dataset from {args.data_path}/{args.data_name}...")
        input_file = os.path.join(args.data_path, args.data_name)
        dataset = load_dataset(input_file, split=args.split)
        
        # 设置输出目录
        post_fix = f"{args.prefix}iterative-rounds{args.max_rounds}-topk{args.top_k_relations}"
        data_name = args.data_name + "_undirected" if args.undirected else args.data_name
        output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
        
        logger.info(f"Results will be saved to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化语言模型
        logger.info(f"Initializing language model: {args.model_name}")
        LLM = get_registed_model(args.model_name)
        model = LLM(args)
        model.prepare_for_inference()
        
        # 保存参数
        FileHandler.save_args(output_dir, args)
        
        # 获取输出文件和已处理列表
        predictions_path = os.path.join(output_dir, 'predictions.jsonl')
        fout, processed_list = FileHandler.get_output_file(predictions_path, force=args.force)
        
        # 初始化知识探索器
        explorer = KnowledgeExplorer(
            kg=kg, 
            model=model, 
            max_rounds=args.max_rounds, 
            relation_k=args.top_k_relations,
            use_cache=True
        )
        
        for data in tqdm(dataset):
            res = explorer.process_question(data, processed_list)
            if res is not None:
                if args.debug:
                    print(json.dumps(res))
                fout.write(json.dumps(res) + "\n")
                fout.flush()
            else:
                logger.warning(f"None result for: {data.get('id', 'unknown')}")
                    
        # 关闭输出文件
        fout.close()
        
        # 评估结果
        logger.info("Evaluating results...")
        eval_path_result_w_ans(predictions_path)
        
    finally:
        # 确保关闭知识图谱连接
        kg.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering via Iterative Reasoning")
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='rmanluo', help="Path to the dataset directory")
    parser.add_argument('--data_name', type=str, default='RoG-webqsp', help="Dataset name")
    parser.add_argument('--split', type=str, default='test[:100]', help="Dataset split to use")
    
    # 输出参数
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning', help="Path to save prediction results")
    parser.add_argument('--force', action='store_true', help="Force overwrite existing results")
    parser.add_argument('--debug', action='store_true', help="Print debug information")
    parser.add_argument('--prefix', type=str, default="", help="Prefix for result directory")
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='gcr-Llama-2-7b-chat-hf', help="Name of LLM to use")
    parser.add_argument('--model_path', type=str, default=None, help="Path to model weights")
    
    # 推理参数
    parser.add_argument('--max_rounds', type=int, default=3, help="Maximum number of exploration rounds")
    parser.add_argument('--top_k_relations', type=int, default=5, help="Number of relations to select per entity")
    parser.add_argument('--n', type=int, default=10, help="Number of parallel processes")
    
    # 知识图谱参数
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help="Neo4j database URI")
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help="Neo4j username")
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang', help="Neo4j password")
    parser.add_argument('--embedding_model', type=str, default='msmarco-distilbert-base-tas-b', help="Embedding model for entity search")
    parser.add_argument('--undirected', type=lambda x: (str(x).lower() == 'true'), default=False, help="Whether to treat the graph as undirected")
    
    # 解析参数
    args, _ = parser.parse_known_args()
    
    # 添加模型特定参数
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()
    
    main(args)