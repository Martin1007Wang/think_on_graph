import torch
import json
import argparse
import os
import gc
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging
import psutil

# 基本日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """数据处理配置"""
    max_pairs: int = 5
    max_negatives_per_pair: int = 5

def format_path_for_json(path: List[Tuple[str, str, str]]) -> str:
    """将路径转换为JSON友好的字符串格式"""
    return " ; ".join(f"{src}-[{rel}]->{tgt}" for src, rel, tgt in path) if path else ""

class CachingPathGenerator:
    """带缓存的路径生成器，避免重复计算"""
    
    def __init__(self, kg, max_path_length, top_k_relations):
        """初始化路径生成器和缓存"""
        from src.path_generator import PathGenerator
        self.generator = PathGenerator(kg=kg, max_path_length=max_path_length, top_k_relations=top_k_relations)
        # 缓存
        self.golden_path_cache = {}
        self.positive_path_cache = {}
        self.negative_paths_cache = {}
    
    def get_golden_path(self, q_entity: str, a_entity: str):
        """获取带缓存的黄金路径"""
        cache_key = (q_entity, a_entity)
        if cache_key not in self.golden_path_cache:
            self.golden_path_cache[cache_key] = self.generator.get_golden_path(q_entity, a_entity)
        return self.golden_path_cache[cache_key]
    
    def get_positive_path(self, q_entity: str, a_entity: str, question: str):
        """获取带缓存的正向路径"""
        cache_key = (q_entity, a_entity, question)
        if cache_key not in self.positive_path_cache:
            self.positive_path_cache[cache_key] = self.generator.get_positive_path(q_entity, a_entity, question)
        return self.positive_path_cache[cache_key]
    
    def get_negative_paths(self, positive_path, question: str, a_entity: str, max_negatives: int):
        """获取带缓存的负向路径"""
        # 使用路径字符串表示作为缓存键
        path_str = str(positive_path)
        cache_key = (path_str, question, a_entity, max_negatives)
        
        if cache_key not in self.negative_paths_cache:
            self.negative_paths_cache[cache_key] = self.generator.get_negative_paths(
                positive_path, question, a_entity, max_negatives
            )[:max_negatives]
        
        return self.negative_paths_cache[cache_key]

class OptimizedKnowledgeGraph:
    """优化的知识图谱类，封装原始KnowledgeGraph并添加优化"""
    
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_name):
        """初始化知识图谱"""
        from src.knowledge_graph import KnowledgeGraph
        self.kg = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password, model_name)
        
    def initialize_embeddings(self, dataset, split):
        """初始化嵌入向量"""
        logger.info("初始化实体和关系嵌入...")
        self.kg.initialize_embeddings(dataset=dataset, split=split)
        logger.info("嵌入初始化完成")
        
        # 确保嵌入已加载到GPU（如果可用）
        if torch.cuda.is_available():
            self.kg.entity_embeddings = self.kg.entity_embeddings.cuda()
            self.kg.relation_embeddings = self.kg.relation_embeddings.cuda()
            logger.info("嵌入已转移到GPU")

def process_entity_pair(q_entity: str, a_entity: str, question: str, path_generator, max_negatives_per_pair: int) -> Dict[str, Any]:
    """处理单个实体对"""
    # 跳过空实体
    if not q_entity or not a_entity:
        return {"golden_path": "", "positive_path": "", "negative_paths": []}
    
    # 获取黄金路径
    golden_path = path_generator.get_golden_path(q_entity, a_entity)
    
    # 获取正向路径
    positive_path, _ = path_generator.get_positive_path(q_entity, a_entity, question)
    
    # 获取负向路径
    negative_paths = []
    if positive_path:
        negative_paths = path_generator.get_negative_paths(
            positive_path, question, a_entity, max_negatives_per_pair
        )[:max_negatives_per_pair]
    
    return {
        "golden_path": format_path_for_json(golden_path),
        "positive_path": format_path_for_json(positive_path),
        "negative_paths": [format_path_for_json(np) for np in negative_paths]
    }

def process_sample(sample: Dict, config: ProcessingConfig, path_generator: CachingPathGenerator) -> List[Dict]:
    q_entities = sample['q_entity'] if isinstance(sample['q_entity'], list) else [sample['q_entity']]
    a_entities = sample.get('a_entity', []) if isinstance(sample.get('a_entity', []), list) else [sample.get('a_entity', '')]
    question = sample['question']
    sample_id = sample.get('id', 'unknown')
    
    all_results = []
    for q_entity in q_entities:
        pairs = [(q_entity, a) for a in a_entities if a][:config.max_pairs]
        for q_entity, a_entity in pairs:
            pair_result = process_entity_pair(
                q_entity, a_entity, question, path_generator, config.max_negatives_per_pair
            )
            has_valid_path = (
                pair_result["golden_path"].strip() != "" or
                pair_result["positive_path"].strip() != "" or
                any(path.strip() != "" for path in pair_result["negative_paths"])
            )
            if has_valid_path:
                result_item = {
                    "id": sample_id,
                    "question": question,
                    "q_entity": q_entity,
                    "a_entity": a_entity,
                    "golden_path": pair_result["golden_path"],
                    "positive_path": pair_result["positive_path"],
                    "negative_paths": pair_result["negative_paths"]
                }
                all_results.append(result_item)
    return all_results

def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    logger.info(f"当前内存使用: {memory_mb:.2f} MB")

def prepare_dataset(args):
    logger.info(f"加载数据集: {args.data_path}")
    dataset = load_dataset(args.data_path, split=args.split)
    dataset_list = list(dataset)
    if args.num_samples > 0:
        dataset_list = dataset_list[:args.num_samples]
    total_samples = len(dataset_list)
    logger.info(f"加载了 {total_samples} 个样本")
    logger.info("初始化知识图谱...")
    kg = OptimizedKnowledgeGraph(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.model_name
    )
    kg.initialize_embeddings(dataset=args.dataset_name, split=args.split)
    
    logger.info("初始化路径生成器...")
    path_generator = CachingPathGenerator(
        kg=kg.kg,  # 使用原始KG实例
        max_path_length=args.max_path_length,
        top_k_relations=args.top_k_relations
    )
    
    # 处理配置
    config = ProcessingConfig(
        max_pairs=args.max_pairs,
        max_negatives_per_pair=args.max_negatives_per_pair
    )
    
    logger.info("开始处理样本")
    print_memory_usage()
    
    # 处理所有样本
    all_results = []
    
    try:
        for idx, sample in enumerate(tqdm(dataset_list, desc="处理样本")):
            sample_results = process_sample(sample, config, path_generator)
            all_results.extend(sample_results)
    
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}", exc_info=True)
        if all_results:
            logger.info(f"尝试保存已处理的 {len(all_results)} 个结果...")
    
    finally:
        # 确保处理完毕后清理内存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 保存结果
    if all_results:
        logger.info(f"处理完成。总结果数: {len(all_results)}")
        
        # 创建输出目录
        output_dir = os.path.join(args.output_path, args.output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存为JSON格式
        json_output_path = os.path.join(output_dir, 'data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"数据集已保存为JSON: {json_output_path}")
        
        # 保存为Hugging Face数据集格式
        processed_dataset = Dataset.from_list(all_results)
        processed_dataset.save_to_disk(output_dir)
        logger.info(f"数据集已保存为Dataset格式: {output_dir}")
    else:
        logger.warning("没有收集到任何结果，不保存输出")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset with various path types")
    parser.add_argument('--data_path', type=str, required=True, help='输入数据集路径')
    parser.add_argument('--dataset_name', type=str, default='RoG-webqsp', help='数据集名称')
    parser.add_argument('--split', type=str, default='train', help='数据集分割')
    parser.add_argument('--output_path', type=str, default='data/processed', help='输出目录')
    parser.add_argument('--output_name', type=str, default='path_enhanced_dataset', help='输出数据集名称')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Neo4j用户名')
    parser.add_argument('--neo4j_password', type=str, default='password', help='Neo4j密码')
    parser.add_argument('--max_path_length', type=int, default=3, help='最大路径长度')
    parser.add_argument('--top_k_relations', type=int, default=5, help='考虑的顶部K个关系')
    parser.add_argument('--max_pairs', type=int, default=5, help='每个样本的最大实体对数')
    parser.add_argument('--max_negatives_per_pair', type=int, default=5, help='每对实体的最大负面样本数')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='预训练模型名称')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--num_samples', type=int, default=-1, help='处理的最大样本数，-1表示处理所有样本')
    
    args = parser.parse_args()
    prepare_dataset(args)
