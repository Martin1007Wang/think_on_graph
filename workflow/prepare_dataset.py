import torch
import json
import argparse
import os
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging
from src.path_generator import PathGenerator

# 基本日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    max_pairs: int = 5
    max_negatives_per_pair: int = 5
    max_path_length: int = 3
    top_k_relations: int = 5

def format_path_for_json(path: List[Tuple[str, str, str]]) -> str:
    return " ; ".join(f"{src}-[{rel}]->{tgt}" for src, rel, tgt in path) if path else ""


class OptimizedKnowledgeGraph:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password, model_name):
        from src.knowledge_graph import KnowledgeGraph
        self.kg = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password, model_name)
        
    def initialize_embeddings(self, dataset, split):
        self.kg.initialize_embeddings(dataset=dataset, split=split)
        if torch.cuda.is_available():
            self.kg.entity_embeddings = self.kg.entity_embeddings.cuda()
            self.kg.relation_embeddings = self.kg.relation_embeddings.cuda()

def process_entity_pair(q_entity: str, a_entity: str, question: str, path_generator, max_negatives_per_pair: int) -> Dict[str, Any]:
    if not q_entity or not a_entity:
        return {"shortest_path": [], "semantic_path": [], "negative_paths": [], "positive_paths": []}
    shortest_paths = path_generator.get_shortest_paths(q_entity, a_entity) or []
    semantic_paths, _ = path_generator.get_semantic_path(q_entity, a_entity, question)
    semantic_paths = semantic_paths or []
    def path_to_tuple(path):
        return tuple(path) if path else tuple()
    all_positive = {path_to_tuple(p) for p in shortest_paths if p}
    all_positive.update({path_to_tuple(p) for p in semantic_paths if p})
    negative_paths = []
    for p in all_positive:
        if not p:
            continue
        negs = path_generator.get_negative_paths(p, question, a_entity, max_negatives_per_pair)
        if negs:
            negative_paths.extend(negs[:max_negatives_per_pair])
    return {
        "shortest_paths": [format_path_for_json(p) for p in shortest_paths if p],
        "semantic_paths": [format_path_for_json(p) for p in semantic_paths if p],
        "negative_paths": [format_path_for_json(np) for np in negative_paths if np],
        "positive_paths": [format_path_for_json(list(p)) for p in all_positive if p]
    }


def process_sample(sample: Dict, config: ProcessingConfig, path_generator) -> List[Dict]:
    q_entities = sample['q_entity'] if isinstance(sample['q_entity'], list) else [sample['q_entity']]
    a_entities = sample.get('a_entity', []) if isinstance(sample.get('a_entity', []), list) else [sample.get('a_entity', '')]
    question = sample['question']
    sample_id = sample.get('id', 'unknown')
    
    all_results = []
    for q_entity in q_entities:
        pairs = [(q_entity, a) for a in a_entities if a][:config.max_pairs]
        for q_entity, a_entity in pairs:
            pair_result = process_entity_pair(q_entity, a_entity, question, path_generator, config.max_negatives_per_pair)
            
            # Only include results where both positive and negative paths exist
            if pair_result['positive_paths'] and pair_result['negative_paths']:
                result_item = {
                    "id": sample_id,
                    "question": question,
                    "q_entity": q_entity,
                    "a_entity": a_entity,
                    **pair_result
                }
                all_results.append(result_item)
    return all_results

def prepare_dataset(args):
    logger.info(f"Loading dataset: {args.data_path}")
    dataset = load_dataset(args.data_path, split=args.split)
    dataset_list = list(dataset)
    if args.num_samples > 0:
        dataset_list = dataset_list[:args.num_samples]
    total_samples = len(dataset_list)
    logger.info(f"Loaded {total_samples} samples")
    logger.info("Initializing knowledge graph...")
    kg = OptimizedKnowledgeGraph(
        args.neo4j_uri, args.neo4j_user, args.neo4j_password, args.model_name
    )
    kg.initialize_embeddings(dataset=args.dataset_name, split=args.split)

    logger.info("Initializing path generator...")
    config = ProcessingConfig(
        max_pairs=args.max_pairs,
        max_negatives_per_pair=args.max_negatives_per_pair,
        max_path_length=args.max_path_length,
        top_k_relations=args.top_k_relations
    )
    path_generator = PathGenerator(
        kg=kg.kg,
        max_path_length=config.max_path_length,
        top_k_relations=config.top_k_relations
    )
    logger.info("Starting to process samples")
    all_results = []
    try:
        for sample in tqdm(dataset_list, desc="Processing samples"):
            sample_results = process_sample(sample, config, path_generator)
            all_results.extend(sample_results)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        if all_results:
            logger.info(f"Attempting to save {len(all_results)} processed results...")
    if all_results:
        logger.info(f"Processing complete. Total results: {len(all_results)}")
        output_dir = os.path.join(args.output_path, args.output_name)
        os.makedirs(output_dir, exist_ok=True)
        json_output_path = os.path.join(output_dir, 'data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved to JSON: {json_output_path}")
        processed_dataset = Dataset.from_list(all_results)
        processed_dataset.save_to_disk(output_dir) 
        logger.info(f"Dataset saved to: {output_dir}")
    else:
        logger.warning("No results collected, nothing to save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset with various path types")
    parser.add_argument('--data_path', type=str, required=True, help='Input dataset path')
    parser.add_argument('--dataset_name', type=str, default='RoG-webqsp', help='Dataset name')
    parser.add_argument('--split', type=str, default='train', help='Dataset split')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--output_name', type=str, default='path_enhanced_dataset', help='Output dataset name')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default='password', help='Neo4j password')
    parser.add_argument('--max_path_length', type=int, default=3, help='Maximum path length')
    parser.add_argument('--top_k_relations', type=int, default=5, help='Top K relations to consider')
    parser.add_argument('--max_pairs', type=int, default=5, help='Maximum entity pairs per sample')
    parser.add_argument('--max_negatives_per_pair', type=int, default=5, help='Maximum negative samples per pair')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=-1, help='Maximum samples to process, -1 for all')
    
    args = parser.parse_args()
    prepare_dataset(args)
