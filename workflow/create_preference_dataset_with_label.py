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

# 基本日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    max_k_relations: int = 5  # Maximum number of relations to select in prompt

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
    # Create relation list with actual relation names
    relation_options = "\n".join(f"- {rel}" for rel in available_relations)
    
    # Create prompt
    prompt = f"""You are a knowledge graph exploration strategist. Given a question and a topic entity, select relevant relations to explore.

# Question: 
{question}

# Topic entity: 
{entity}

# Available relations from this entity:
{relation_options}

Select up to {max_k_relations} relations that seem most promising or potentially relevant for answering the question. Consider that the answer might require exploring multiple steps.
Your response should ONLY contain the relation names from the list above, separated by commas.
Your selection (up to {max_k_relations} relations):"""
    
    # Create chosen and rejected responses
    # Limit to max_k_relations
    chosen_relations_limited = chosen_relations[:max_k_relations]
    chosen = ", ".join(chosen_relations_limited) if chosen_relations_limited else "None of these relations seem relevant"
    
    # For rejected, we'll use relations that aren't in chosen_relations
    rejected_relations = [rel for rel in available_relations if rel not in chosen_relations]
    # Limit to max_k_relations and ensure it's different from chosen
    rejected_relations_limited = rejected_relations[:max_k_relations]
    if set(rejected_relations_limited) == set(chosen_relations_limited):
        # If they would be the same, modify rejected to be different
        if len(rejected_relations_limited) > 1:
            rejected_relations_limited = rejected_relations_limited[1:] + [rejected_relations_limited[0]]
        elif available_relations:
            # If only one relation, just pick a different format
            rejected = "I need more context to select relations"
        else:
            rejected = "None of these relations seem relevant"
    else:
        rejected = ", ".join(rejected_relations_limited) if rejected_relations_limited else "None of these relations seem relevant"
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def process_path_item(item: Dict, config: ProcessingConfig) -> List[Dict]:
    """Process a single path item to create preference examples."""
    question = item["question"]
    q_entity = item["q_entity"]
    a_entity = item["a_entity"]
    sample_id = item.get("id", "unknown")
    
    preference_examples = []
    
    # Create preference examples for relation selection
    positive_hop_relations = group_relations_by_hop(item['positive_paths'])
    negative_hop_relations = group_relations_by_hop(item['negative_paths'])
    
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
    
    return preference_examples

def create_preference_dataset(args):
    logger.info(f"Loading path dataset: {args.input_path}")
    
    # Load the path dataset
    if args.input_path.endswith('.json'):
        with open(args.input_path, 'r', encoding='utf-8') as f:
            path_data = json.load(f)
    else:
        # Assume it's a Hugging Face dataset directory
        path_dataset = Dataset.load_from_disk(args.input_path)
        path_data = list(path_dataset)
    
    total_items = len(path_data)
    logger.info(f"Loaded {total_items} path items")
    
    if args.num_samples > 0:
        path_data = path_data[:args.num_samples]
        logger.info(f"Using {len(path_data)} samples")
    
    config = ProcessingConfig(max_k_relations=args.max_k_relations)
    
    logger.info("Starting to process path items")
    all_preference_examples = []
    
    try:
        for item in tqdm(path_data, desc="Processing path items"):
            preference_examples = process_path_item(item, config)
            all_preference_examples.extend(preference_examples)
    except Exception as e:
        logger.error(f"Error during processing: {e}", exc_info=True)
        if all_preference_examples:
            logger.info(f"Attempting to save {len(all_preference_examples)} processed examples...")
    
    # Save preference dataset
    if all_preference_examples:
        logger.info(f"Processing complete. Total preference examples: {len(all_preference_examples)}")
        
        # Create output directory
        output_dir = os.path.join(args.output_path, args.output_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_output_path = os.path.join(output_dir, 'preference_data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_preference_examples, f, ensure_ascii=False, indent=2)
        logger.info(f"Preference dataset saved to JSON: {json_output_path}")
        
        # Save as HF dataset
        preference_dataset = Dataset.from_list(all_preference_examples)
        preference_dataset.save_to_disk(output_dir)
        logger.info(f"Preference dataset saved to: {output_dir}")
    else:
        logger.warning("No preference examples collected, nothing to save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create preference dataset from path data")
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Input path dataset file (.json) or directory (HF dataset)')
    parser.add_argument('--output_path', type=str, default='/mnt/wangjingxiong/think_on_graph/data/processed', 
                        help='Output directory')
    parser.add_argument('--output_name', type=str, default='RoG-webqsp_train_preference_with_label', 
                        help='Output dataset name')
    parser.add_argument('--max_k_relations', type=int, default=5, 
                        help='Maximum relations to select in prompt')
    parser.add_argument('--num_samples', type=int, default=-1, 
                        help='Maximum samples to process, -1 for all')
    
    args = parser.parse_args()
    create_preference_dataset(args)
