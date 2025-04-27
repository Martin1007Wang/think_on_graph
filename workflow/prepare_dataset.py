import torch
import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging
from collections import defaultdict
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
    max_k_relations: int = 5  # Maximum number of relations to select in prompt

def format_path_for_json(path: List[Tuple[str, str, str]]) -> str:
    return " ; ".join(f"{src}-[{rel}]->{tgt}" for src, rel, tgt in path) if path else ""

def extract_relations_from_path(path_str: str) -> List[Tuple[str, str, int]]:
    """Extract relations from a path string with their source entity and hop level."""
    if not path_str:
        return []
    
    relations = []
    path_steps = path_str.split(" ; ")
    
    for i, step in enumerate(path_steps):
        match = re.match(r"([^-]+)-\[([^\]]+)\]->([^-]+)", step)
        if match:
            src, rel, tgt = match.groups()
            relations.append((src.strip(), rel.strip(), i))
    
    return relations

def group_relations_by_hop(paths: List[str]) -> Dict[int, Dict[str, Set[str]]]:
    """Group relations by hop level and source entity."""
    hop_relations = defaultdict(lambda: defaultdict(set))
    
    for path in paths:
        relations = extract_relations_from_path(path)
        for src, rel, hop in relations:
            hop_relations[hop][src].add(rel)
    
    return hop_relations

def create_relation_selection_example(question: str, entity: str, available_relations: List[str], 
                                      chosen_relations: List[str], max_k_relations: int) -> Dict:
    """Create a preference example for relation selection."""
    # Create relation mapping
    relation_dict = {f"REL_{i}": rel for i, rel in enumerate(available_relations)}
    relation_options = "\n".join(f"[REL_{i}] {rel}" for i, rel in enumerate(available_relations))
    
    # Create prompt
    prompt = f"""You are a knowledge graph exploration strategist. Given a question and a topic entity, select relevant relations to explore.

# Question: 
{question}

# Topic entity: 
{entity}

# Available relations from this entity:
{relation_options}

Select up to {max_k_relations} relation IDs that seem most promising or potentially relevant for answering the question. Consider that the answer might require exploring multiple steps.
Your response should ONLY contain the relation IDs (e.g., REL_0, REL_1) from the list above.
Your selection (IDs only, up to {max_k_relations}):"""
    
    # Create chosen and rejected responses
    chosen_ids = [f"REL_{i}" for i, rel in enumerate(available_relations) if rel in chosen_relations]
    # Limit to max_k_relations
    chosen_ids = chosen_ids[:max_k_relations]
    chosen = ", ".join(chosen_ids) if chosen_ids else "None of these relations seem relevant"
    
    # For rejected, we'll use relations that aren't in chosen_relations
    rejected_relations = [rel for rel in available_relations if rel not in chosen_relations]
    rejected_ids = [f"REL_{i}" for i, rel in enumerate(available_relations) if rel in rejected_relations]
    # Limit to max_k_relations and ensure it's different from chosen
    rejected_ids = rejected_ids[:max_k_relations]
    if set(rejected_ids) == set(chosen_ids):
        # If they would be the same, modify rejected to be different
        if len(rejected_ids) > 1:
            rejected_ids = rejected_ids[1:] + [rejected_ids[0]]
        elif available_relations:
            # If only one relation, just pick a different format
            rejected = "I need more context to select relations"
        else:
            rejected = "None of these relations seem relevant"
    else:
        rejected = ", ".join(rejected_ids) if rejected_ids else "None of these relations seem relevant"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }


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
    preference_examples = []
    
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
                
                # Create preference examples for relation selection
                positive_hop_relations = group_relations_by_hop(pair_result['positive_paths'])
                negative_hop_relations = group_relations_by_hop(pair_result['negative_paths'])
                
                # Process each hop level
                for hop, src_relations in positive_hop_relations.items():
                    for src_entity, pos_relations in src_relations.items():
                        # Get negative relations for the same source entity and hop if available
                        neg_relations = set()
                        if hop in negative_hop_relations and src_entity in negative_hop_relations[hop]:
                            neg_relations = negative_hop_relations[hop][src_entity]
                        
                        # Combine all available relations
                        all_relations = list(pos_relations.union(neg_relations))
                        if not all_relations:
                            continue
                            
                        # Create preference example
                        example = create_relation_selection_example(
                            question=question,
                            entity=src_entity,
                            available_relations=all_relations,
                            chosen_relations=list(pos_relations),
                            max_k_relations=config.max_k_relations
                        )
                        
                        # Add metadata
                        example["metadata"] = {
                            "id": f"{sample_id}_hop{hop}_{src_entity}",
                            "hop": hop,
                            "source_entity": src_entity,
                            "target_entity": a_entity,
                            "available_relations": all_relations,
                            "positive_relations": list(pos_relations),
                            "negative_relations": list(neg_relations)
                        }
                        
                        preference_examples.append(example)
    
    return all_results, preference_examples

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
        top_k_relations=args.top_k_relations,
        max_k_relations=args.max_k_relations
    )
    path_generator = PathGenerator(
        kg=kg.kg,
        max_path_length=config.max_path_length,
        top_k_relations=config.top_k_relations
    )
    logger.info("Starting to process samples")
    all_results = []
    all_preference_examples = []
    
    try:
        for sample in tqdm(dataset_list, desc="Processing samples"):
            sample_results, preference_examples = process_sample(sample, config, path_generator)
            all_results.extend(sample_results)
            all_preference_examples.extend(preference_examples)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        if all_results or all_preference_examples:
            logger.info(f"Attempting to save processed results...")
    
    # Save path-enhanced dataset
    if all_results:
        logger.info(f"Processing complete. Total path results: {len(all_results)}")
        output_dir = os.path.join(args.output_path, args.output_name)
        os.makedirs(output_dir, exist_ok=True)
        json_output_path = os.path.join(output_dir, 'data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Path dataset saved to JSON: {json_output_path}")
        processed_dataset = Dataset.from_list(all_results)
        processed_dataset.save_to_disk(output_dir) 
        logger.info(f"Path dataset saved to: {output_dir}")
    else:
        logger.warning("No path results collected, nothing to save")
    
    # Save preference dataset
    if all_preference_examples:
        logger.info(f"Total preference examples: {len(all_preference_examples)}")
        preference_output_dir = os.path.join(args.output_path, f"{args.output_name}_preference")
        os.makedirs(preference_output_dir, exist_ok=True)
        
        # Save as JSON
        preference_json_path = os.path.join(preference_output_dir, 'preference_data.json')
        with open(preference_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_preference_examples, f, ensure_ascii=False, indent=2)
        logger.info(f"Preference dataset saved to JSON: {preference_json_path}")
        
        # Save as HF dataset
        preference_dataset = Dataset.from_list(all_preference_examples)
        preference_dataset.save_to_disk(preference_output_dir)
        logger.info(f"Preference dataset saved to: {preference_output_dir}")
    else:
        logger.warning("No preference examples collected, nothing to save")

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
    parser.add_argument('--max_k_relations', type=int, default=5, help='Maximum relations to select in prompt')
    parser.add_argument('--max_pairs', type=int, default=5, help='Maximum entity pairs per sample')
    parser.add_argument('--max_negatives_per_pair', type=int, default=5, help='Maximum negative samples per pair')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='Pretrained model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=-1, help='Maximum samples to process, -1 for all')
    
    args = parser.parse_args()
    prepare_dataset(args)
