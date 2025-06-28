import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from datasets import Dataset # type: ignore
from tqdm import tqdm # type: ignore
import logging
from collections import defaultdict
import random
from enum import Enum

from src.knowledge_graph import KnowledgeGraph
from src.template_v2 import KnowledgeGraphTemplates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pre-compiled regex for performance
PATH_STEP_REGEX = re.compile(r"(.+?)\s*-\[\s*(.+?)\s*\]->\s*(.+)")

class CandidateStrategy(Enum):
    PN_ONLY = "pn_only"
    KG_ALLHOP = "kg_allhop"
    PN_KG_SUPPLEMENT = "pn_kg_supplement"

class PositiveSource(Enum):
    POSITIVE_PATHS = "positive_paths"
    SHORTEST_PATHS = "shortest_paths"

@dataclass
class ProcessingConfig:
    max_selection_count: int = 5
    enable_relation_sampling: bool = False
    relation_sampling_threshold: int = 25
    num_distractors_to_sample: int = 10
    candidate_strategy: CandidateStrategy = CandidateStrategy.PN_KG_SUPPLEMENT
    positive_source_field: PositiveSource = PositiveSource.POSITIVE_PATHS
    max_negatives_per_sample: int = 3

def parse_path_to_segments(path_str: str) -> List[Tuple[str, str, str]]:
    if not path_str:
        return []
    segments: List[Tuple[str, str, str]] = []
    path_steps = path_str.split(" ; ")
    for i, step in enumerate(path_steps):
        match = PATH_STEP_REGEX.match(step)
        if match:
            src, rel, tgt = match.groups()
            segments.append((src.strip(), rel.strip(), tgt.strip()))
        else:
            logger.debug(f"Malformed path step: '{step}' in path '{path_str}'")
    return segments

def build_history_key(history_segments: List[Tuple[str, str, str]]) -> str:
    if not history_segments:
        return ""
    return " ; ".join([f"{s}-[{r}]->{t}" for s, r, t in history_segments])

def create_mpo_preference_example(
    question: str,
    history_tuples: List[Tuple[str, str, str]],
    current_entity: str,
    prompt_available_relations: List[str],
    gold_chosen_relations: List[str],
    gold_negative_relations: List[str],
    max_selection_count: int,
    template_builder: KnowledgeGraphTemplates
) -> Optional[Dict[str, Any]]:
    """
    Creates a single preference data sample in the MPO format, ensuring the
    chosen/rejected values match the tagged format shown in the prompt.
    """
    unique_prompt_available_relations = sorted(list(set(prompt_available_relations)))

    if not unique_prompt_available_relations:
        logger.debug(f"No available relations for entity '{current_entity}'. Skipping.")
        return None

    # ========================================================================
    #  1. 创建从纯关系到带标签关系的映射字典 (Create a map from raw relation to tagged relation)
    # ========================================================================
    # 这个映射是本次修改的核心，它能让我们轻松地找到每个关系对应的完整格式
    tagged_relations_map = {
        rel: f"[REL{i+1}] {rel}"
        for i, rel in enumerate(unique_prompt_available_relations)
    }

    # 使用映射的值来构建将要展示在 prompt 中的字符串
    relation_options_str = "\n".join(tagged_relations_map.values())

    # --- Template formatting logic remains the same ---
    template_args = {
        "question": question,
        "entity": current_entity,
        "history": history_tuples,
        "relations": relation_options_str,
        "max_selection_count": max_selection_count
    }
    # (Assuming you have updated KnowledgeGraphTemplates to handle history_tuples)
    user_prompt_content = template_builder.format_template(
        "relation_selection_with_history" if history_tuples else "relation_selection", **template_args
    )

    # ========================================================================
    #  2. 使用映射来构建 'chosen' 和 'rejected' 列表
    #     (Use the map to construct the 'chosen' and 'rejected' lists)
    # ========================================================================

    # 2.1. Construct the list of 'chosen' relations using the map
    mpo_chosen_list = sorted([
        tagged_relations_map[rel]
        for rel in gold_chosen_relations
        if rel in tagged_relations_map  # Ensure the gold relation is actually in the prompt
    ])

    if not mpo_chosen_list:
        logger.debug(f"No gold chosen relations for entity '{current_entity}' were in the final candidate pool. Skipping.")
        return None

    # 2.2. Construct the list of 'rejected' relations using the map
    # Start with gold negative relations that are in the candidate pool
    rejected_candidates = {
        rel for rel in gold_negative_relations if rel in tagged_relations_map
    }
    # Add other available relations that are not chosen to the rejection pool
    other_distractors = {
        rel for rel in unique_prompt_available_relations if rel not in gold_chosen_relations
    }
    rejected_candidates.update(other_distractors)
    
    mpo_rejected_list = sorted([
        tagged_relations_map[rel]
        for rel in rejected_candidates
        # Final check to ensure we don't accidentally include a chosen one
        if rel in tagged_relations_map and rel not in gold_chosen_relations
    ])


    if not mpo_rejected_list:
        logger.debug(f"Could not form a non-empty rejected list for entity '{current_entity}'. Skipping.")
        return None

    # Final sanity check to avoid identical preference pairs
    if set(mpo_chosen_list) == set(mpo_rejected_list):
        logger.debug(f"Chosen and rejected lists are identical for entity '{current_entity}'. Skipping.")
        return None

    # Return the data in the final, perfectly consistent MPO format
    return {
        "prompt": user_prompt_content,
        "chosen": mpo_chosen_list,
        "rejected": mpo_rejected_list
    }

def process_path_item(
    kg: Optional[KnowledgeGraph],
    item: Dict[str, Any],
    config: ProcessingConfig,
    template_builder: KnowledgeGraphTemplates,
    kg_relations_cache: Dict[str, Set[str]]
) -> List[Dict[str, Any]]:
    question = item.get("question", "")
    sample_id = str(item.get("id", f"unknown_sample_{random.getrandbits(32)}"))
    preference_examples: List[Dict[str, Any]] = []

    positive_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    negative_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    # Aggregate all positive and negative relations at each step
    positive_source_key = config.positive_source_field.value
    for p_path_str in item.get(positive_source_key, []):
        history_segments: List[Tuple[str,str,str]] = []
        for src, rel, tgt in parse_path_to_segments(p_path_str):
            positive_next_relations[build_history_key(history_segments)][src].add(rel)
            history_segments.append((src, rel, tgt))

    for n_path_str in item.get('negative_paths', []):
        history_segments = []
        for src, rel, tgt in parse_path_to_segments(n_path_str):
            negative_next_relations[build_history_key(history_segments)][src].add(rel)
            history_segments.append((src, rel, tgt))

    # Iterate through positive paths to generate examples for each step
    processed_steps: Set[Tuple[str, str]] = set()
    for p_path_str in item.get('positive_paths', []):
        current_history: List[Tuple[str, str, str]] = []
        for i_seg, (src, rel, tgt) in enumerate(parse_path_to_segments(p_path_str)):
            history_key = build_history_key(current_history)
            
            # Avoid re-processing the same decision point (entity + history) within one sample
            if (history_key, src) in processed_steps:
                current_history.append((src, rel, tgt))
                continue
            processed_steps.add((history_key, src))

            gold_chosen_rels = sorted(list(positive_next_relations[history_key].get(src, set())))
            if not gold_chosen_rels:
                current_history.append((src, rel, tgt))
                continue

            full_gold_negative_rels = sorted(list(negative_next_relations[history_key].get(src, set())))
            sampled_negative_rels = full_gold_negative_rels
            if config.max_negatives_per_sample > 0 and len(full_gold_negative_rels) > config.max_negatives_per_sample:
                sampled_negative_rels = random.sample(full_gold_negative_rels, config.max_negatives_per_sample)
                logger.debug(f"Sampled {config.max_negatives_per_sample} negatives from a pool of {len(full_gold_negative_rels)} for entity '{src}'.")
            
            sampled_negative_rels = sorted(sampled_negative_rels)
            # Build the candidate pool for the prompt
            candidate_pool: Set[str] = set()
            relations_from_kg: Set[str] = set()

            if config.candidate_strategy != CandidateStrategy.PN_ONLY:
                if src in kg_relations_cache:
                    relations_from_kg = kg_relations_cache[src]
                elif kg:
                    try:
                        kg_rels_list = kg.get_related_relations(src, "out")
                        if kg_rels_list:
                            relations_from_kg = {r.strip() for r in kg_rels_list if r and r.strip()}
                        kg_relations_cache[src] = relations_from_kg
                    except Exception as e:
                        logger.error(f"Error fetching KG relations for '{src}': {e}")
            
            if config.candidate_strategy == CandidateStrategy.PN_ONLY:
                candidate_pool.update(gold_chosen_rels)
                candidate_pool.update(sampled_negative_rels)
            elif config.candidate_strategy == CandidateStrategy.KG_ALLHOP:
                candidate_pool.update(relations_from_kg)
            elif config.candidate_strategy == CandidateStrategy.PN_KG_SUPPLEMENT:
                candidate_pool.update(gold_chosen_rels)
                candidate_pool.update(sampled_negative_rels)
                supplementary = list(relations_from_kg - candidate_pool)
                random.shuffle(supplementary)
                candidate_pool.update(supplementary[:config.num_distractors_to_sample])

            final_relations_for_prompt = sorted(list(candidate_pool))
            
            # Relation sampling if the pool is too large
            if config.enable_relation_sampling and len(final_relations_for_prompt) > config.relation_sampling_threshold:
                must_include = set(gold_chosen_rels) | set(sampled_negative_rels)
                distractors = [r for r in final_relations_for_prompt if r not in must_include]
                random.shuffle(distractors)
                
                num_distractors_needed = config.relation_sampling_threshold - len(must_include)
                final_relations_for_prompt = sorted(list(must_include) + distractors[:max(0, num_distractors_needed)])

            if not final_relations_for_prompt:
                current_history.append((src, rel, tgt))
                continue

            example = create_mpo_preference_example(
                question=question,
                history_tuples=current_history,
                current_entity=src,
                prompt_available_relations=final_relations_for_prompt,
                gold_chosen_relations=gold_chosen_rels,
                gold_negative_relations=sampled_negative_rels,
                max_selection_count=config.max_selection_count,
                template_builder=template_builder
            )

            if example:
                preference_examples.append(example)

            current_history.append((src, rel, tgt))

    return preference_examples


def load_path_data(input_path: str, num_samples: int) -> List[Dict[str, Any]]:
    logger.info(f"Loading data from: {input_path}")
    # This logic now handles both .json and .jsonl correctly based on our previous debug session
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            if input_path.endswith('.jsonl'):
                data = [json.loads(line) for line in f if line.strip()]
            else: # Assumes standard JSON array
                data = json.load(f)
        
        if not isinstance(data, list):
            logger.error(f"Data in {input_path} is not a list.")
            return []
            
        logger.info(f"Loaded {len(data)} items from {input_path}.")
        if num_samples > 0:
            return data[:num_samples]
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {input_path}: {e}")
        return []

def main(args, template_builder):
    template_builder = KnowledgeGraphTemplates()
    kg_instance = None 
    kg_relations_cache = defaultdict(set)

    config = ProcessingConfig(
        candidate_strategy=CandidateStrategy(args.candidate_strategy),
        positive_source_field=PositiveSource(args.positive_source_field),
    )

    all_mpo_examples = []
    for path in args.input_paths:
        path_data = load_path_data(path, args.num_samples)
        for item in tqdm(path_data, desc=f"Processing {os.path.basename(path)}"):
            all_mpo_examples.extend(
                process_path_item(kg_instance, item, config, template_builder, kg_relations_cache)
            )

    logger.info(f"Total MPO preference examples generated: {len(all_mpo_examples)}")

    # Save the dataset
    if all_mpo_examples:
        output_dir = os.path.join(args.output_path, f"cand_{config.candidate_strategy.value}_pos_{config.positive_source_field.value}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSONL
        jsonl_path = os.path.join(output_dir, "mpo_preference_data.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for ex in all_mpo_examples:
                f.write(json.dumps(ex) + '\n')
        logger.info(f"Saved MPO dataset to JSONL: {jsonl_path}")

        # Save as Hugging Face Dataset
        hf_path = os.path.join(output_dir, "hf_dataset")
        dataset = Dataset.from_list(all_mpo_examples)
        dataset.save_to_disk(hf_path)
        logger.info(f"Saved MPO dataset to Hugging Face format: {hf_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a COMBINED DPO preference dataset from multiple path-enhanced data sources for a GIVEN strategy configuration.")
    parser.add_argument('--input_paths', type=str, nargs='+', required=True,
                        help='One or more input path-enhanced dataset files (e.g., data.json, data.jsonl) or Hugging Face dataset directories.')
    parser.add_argument('--input_dataset_names', type=str, nargs='*',
                        help='Corresponding names for each input dataset (for logging & metadata). If not provided or fewer names than paths, names will be derived from paths.')
    parser.add_argument('--output_path', type=str, default='./data/processed_dpo_combined', # 调整了默认路径名
                        help='Base output directory for the new COMBINED DPO datasets. Strategy-specific subdirectories will be created here.')
    parser.add_argument('--candidate_strategy', type=str, required=True,
                        choices=[cs.value for cs in CandidateStrategy],
                        help='Strategy for constructing candidate relations for THIS RUN.')
    parser.add_argument('--positive_source_field', type=str, required=True,
                        choices=[ps.value for ps in PositiveSource],
                        help='Field in input data to use as source for positive relations for THIS RUN.')
    parser.add_argument('--max_selection_count', type=int, default=5,
                        help='Maximum number of relations the prompt asks the model to select.')
    parser.add_argument('--enable_relation_sampling', action='store_true',
                        help='Enable sampling of relations if the candidate pool exceeds a threshold.')
    parser.add_argument('--relation_sampling_threshold', type=int, default=25,
                        help='Threshold for candidate relations pool size to trigger sampling.')
    parser.add_argument('--num_distractors_to_sample', type=int, default=10,
                        help='Number of distractor relations to sample if sampling is triggered.')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Maximum number of items to process from EACH input path_data (-1 for all, 0 for none). Applied per source.')
    parser.add_argument('--neo4j_uri', type=str, default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'), help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default=os.getenv('NEO4J_USER', 'neo4j'), help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default=os.getenv('NEO4J_PASSWORD', 'password'), help='Neo4j password')

    args = parser.parse_args()

    if args.input_dataset_names and len(args.input_dataset_names) != len(args.input_paths):
        logger.error("CRITICAL: Number of --input_dataset_names MUST match number of --input_paths if provided. "
                     f"Got {len(args.input_dataset_names)} names for {len(args.input_paths)} paths.")
        if args.input_dataset_names:
            exit(1)

    template_builder_instance = None
    try:
        if not (hasattr(KnowledgeGraphTemplates, '__init__') and callable(KnowledgeGraphTemplates.__init__)):
            raise TypeError("KnowledgeGraphTemplates is not a callable class.")
        template_builder_instance = KnowledgeGraphTemplates()
    except Exception as e:
        logger.error(f"Failed to initialize KnowledgeGraphTemplates: {e}. Ensure 'src.template.KnowledgeGraphTemplates' is correctly defined and callable.", exc_info=True)
        exit(1)

    main(args, template_builder_instance)
