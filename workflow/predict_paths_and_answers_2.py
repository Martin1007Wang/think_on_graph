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
            
def ensure_ground_truth(predictions_path: str, dataset: Any) -> None:
    """检查预测文件中是否包含ground_truth字段，如果没有则从数据集中添加。
    
    Args:
        predictions_path: 预测结果文件路径
        dataset: 原始数据集
    """
    logger.info("Checking for ground_truth field in predictions...")
    
    # 创建id到answer的映射
    dataset_answers = {}
    for item in dataset:
        if "id" in item and "answer" in item:
            dataset_answers[item["id"]] = item["answer"]
    
    # 读取预测文件
    with open(predictions_path, "r") as f:
        predictions = [json.loads(line) for line in f]
    
    # 检查并添加ground_truth
    needs_update = False
    for pred in predictions:
        if "id" in pred and "ground_truth" not in pred:
            if pred["id"] in dataset_answers:
                pred["ground_truth"] = dataset_answers[pred["id"]]
                needs_update = True
    
    # 如果需要更新，重写文件
    if needs_update:
        logger.info("Adding missing ground_truth fields to predictions...")
        with open(predictions_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
    else:
        logger.info("All predictions already have ground_truth field.")

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
        
        # target_id = "WebQTest-1161"
        # found = False

        # for data in dataset:
        #     if data.get("id") == target_id:
        #         found = True
        #         logger.info(f"Processing only the target item (id: {target_id})")
        #         res = explorer.process_question(data, processed_list)
        #         if res is not None:
        #             if args.debug:
        #                 print(json.dumps(res))
        #             fout.write(json.dumps(res) + "\n")
        #             fout.flush()
        #         else:
        #             logger.warning(f"None result for target id: {target_id}")
        #         break

        # if not found:
        #     logger.warning(f"Target item with id '{target_id}' not found in dataset")
        
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
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning_v2', help="Path to save prediction results")
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