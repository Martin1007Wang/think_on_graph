import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset # type: ignore
from tqdm import tqdm # type: ignore
import logging
from collections import defaultdict
import random # Added for sampling
from enum import Enum

from src.knowledge_graph import KnowledgeGraph
from src.template import KnowledgeGraphTemplates


# 基本日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CandidateStrategy(Enum):
    PN_ONLY = "pn_only"  # Positive and Negative relations only
    KG_ALLHOP = "kg_allhop"  # All 1-hop relations from KG
    PN_KG_SUPPLEMENT = "pn_kg_supplement"  # Positive, Negative, supplemented by KG

class PositiveSource(Enum):
    POSITIVE_PATHS = "positive_paths"
    SHORTEST_PATHS = "shortest_paths"

@dataclass
class ProcessingConfig:
    max_selection_count: int = 5
    enable_relation_sampling: bool = False
    relation_sampling_threshold: int = 25
    num_distractors_to_sample: int = 10 # Used when sampling or supplementing
    candidate_strategy: CandidateStrategy = CandidateStrategy.PN_KG_SUPPLEMENT
    positive_source_field: PositiveSource = PositiveSource.POSITIVE_PATHS



def format_template(template_name: str, **kwargs: Any) -> str:    
    question = kwargs.get('question', '')
    entity = kwargs.get('entity', '')
    history = kwargs.get('history')  # Expects a pre-formatted string or None
    max_selection_count = kwargs.get('max_selection_count', 5)
    prompt_content = "Based on the following information:\n"
    prompt_content += f"- Question: {question}\n"
    prompt_content += f"- Current Entity: {entity}\n"
    
    if history:
        prompt_content += f"- Exploration History:\n{history.strip()}\n" # 使用 strip() 移除可能的前后空白

    instruction = f"\nPlease generate up to {max_selection_count} relations from the Current Entity "
    instruction += "that are most relevant to answering the Question"
    
    if history:
        instruction += ", considering the Exploration History provided."
    else:
        instruction += "."
    prompt_content += instruction + "\n"
        
    # 输出格式说明
    prompt_content += "\nYour Generated Relations (one relation name per line):"
    
    return prompt_content

def parse_path_to_segments(path_str: str) -> List[Tuple[str, str, str]]:
    """
    Parses a full path string into a list of (source, relation, target) segments.
    """
    if not path_str:
        return []
    segments: List[Tuple[str, str, str]] = []
    path_steps = path_str.split(" ; ")
    
    regex_pattern_to_use = r"(.+?)\s*-\[\s*(.+?)\s*\]->\s*(.+)" # DPO脚本中更健壮的正则
    
    for i, step in enumerate(path_steps):
        match = re.match(regex_pattern_to_use, step)
        if match:
            src = match.group(1).strip()
            rel = match.group(2).strip()
            tgt = match.group(3).strip()
            segments.append((src, rel, tgt))
        else:
            logger.warning(
                f"Malformed path step during SFT data generation. "
                f"Attempted Regex: '{regex_pattern_to_use}'. "
                f"Full Path String: '{path_str}'. "
                f"Problematic Step (repr): {repr(step)}. "
                f"Problematic Step (str): '{step}'."
            )
    return segments

def build_history_key(history_segments: List[Tuple[str, str, str]]) -> str:
    """Builds a canonical string representation of the history path segments."""
    if not history_segments:
        return ""
    return " ; ".join([f"{s}-[{r}]->{t}" for s, r, t in history_segments])


def create_sft_example_with_history_and_options(
    question: str,
    history_tuples: List[Tuple[str, str, str]],
    current_entity: str,
    prompt_available_relations: List[str], # 最终出现在提示中的关系选项
    gold_chosen_relations: List[str], # 这个状态下的“黄金”正面关系
    max_selection_count: int,
) -> Optional[Dict[str, Any]]:
    """
    Create an SFT instruction example with relation options in the prompt
    and the "correct" relations (subset of gold_chosen_relations) as the completion.
    Returns None if a valid SFT example cannot be formed.
    """
    unique_prompt_available_relations = sorted(list(set(prompt_available_relations)))
    
    if not unique_prompt_available_relations:
        logger.debug(f"SFT: No available relations for prompt for entity '{current_entity}' with history '{build_history_key(history_tuples)}'. Cannot form SFT example.")
        return None

    relation_options_lines = []
    rel_to_display_line: Dict[str, str] = {} # Map relation name to its display line like "[REL_0] relation_name"

    for rel_name in unique_prompt_available_relations:
        display_line = f"{rel_name}"
        relation_options_lines.append(f"      {display_line}") # Indentation for prompt
        rel_to_display_line[rel_name] = display_line
    relation_options_str = "\n".join(relation_options_lines)

    history_section_str = ""
    template_args: Dict[str, Any]
    if history_tuples:
        history_lines = ["* Exploration History:"]
        for i_hist, (src_hist, rel_hist, tgt_hist) in enumerate(history_tuples):
            history_lines.append(f"  Step {i_hist+1}: From '{src_hist}' explored relation '{rel_hist}', leading to '{tgt_hist}'.")
        history_section_str = "\n".join(history_lines) + "\n"
        template_name = "relation_selection_with_history" # Reuse DPO template logic
        template_args = {
            "question": question, "entity": current_entity, "history": history_section_str,
            "relations": relation_options_str, "max_selection_count": max_selection_count
        }
    else:
        template_name = "relation_selection" # No history section
        template_args = {
            "question": question, "entity": current_entity, "history": "",
            "relations": relation_options_str, "max_selection_count": max_selection_count
        }
    
    # The prompt will now end with a call to action appropriate for SFT
    # We can modify the template in KnowledgeGraphTemplates or append here.
    # For now, let's assume template_builder.format_template creates the main body.
    # We will append the SFT specific instruction.
    prompt = format_template(template_name, **template_args)


    # Completion: Gold chosen relations that are ACTUALLY AVAILABLE in the prompt
    completion_lines = []
    for rel_name in gold_chosen_relations:
        if rel_name in rel_to_display_line: # Check if this gold relation is in the prompt's list
            completion_lines.append(rel_to_display_line[rel_name])
    
    completion_lines.sort() # Sort for consistency
    completion_lines = completion_lines[:max_selection_count] # Respect max_selection_count

    if not completion_lines:
        logger.debug(f"SFT: Skipping example for entity '{current_entity}' with history '{build_history_key(history_tuples)}': None of the gold_chosen_relations were present in the final prompt_available_relations, or gold_chosen_relations list was empty.")
        return None
    
    completion_str = "\n".join(completion_lines)

    return {
        "prompt": prompt,
        "completion": completion_str 
        # SFT datasets often just have "text" field or "prompt"/"response" or "instruction"/"output"
        # We'll stick to "prompt" and "completion" for now.
    }

def process_path_item_for_sft(
    kg: Optional[KnowledgeGraph], 
    item: Dict[str, Any], 
    config: ProcessingConfig, 
) -> List[Dict[str, Any]]:
    question = item.get("question", "")
    sample_id = item.get("id", "unknown_sft_id")
    sft_examples: List[Dict[str, Any]] = []

    positive_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    negative_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set)) # Still useful for PN_ONLY/PN_KG_SUPPLEMENT

    positive_source_key = config.positive_source_field.value
    logger.debug(f"SFT Item {sample_id}: Using '{positive_source_key}' field as source for positive relations.")
    for p_path_str in item.get(positive_source_key, []):
        segments = parse_path_to_segments(p_path_str)
        if not segments and p_path_str: 
            logger.warning(f"SFT Item {sample_id}: Path from '{positive_source_key}' ('{p_path_str}') could not be parsed. Skipping for positive_next_relations.")
            continue
        current_history_segments: List[Tuple[str,str,str]] = []
        for i, (src, rel, tgt) in enumerate(segments):
            history_key = build_history_key(current_history_segments)
            positive_next_relations[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))

    for n_path_str in item.get('negative_paths', []):
        segments = parse_path_to_segments(n_path_str)
        if not segments and n_path_str:
            logger.warning(f"SFT Item {sample_id}: Negative path '{n_path_str}' could not be parsed. Skipping for negative_next_relations.")
            continue
        current_history_segments: List[Tuple[str,str,str]] = []
        for i, (src, rel, tgt) in enumerate(segments):
            history_key = build_history_key(current_history_segments)
            negative_next_relations[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))
    
    processed_examples_tracker = set()

    for p_path_str in item.get('positive_paths', []): # These define the sequence of decisions
        path_segments = parse_path_to_segments(p_path_str)
        if not path_segments and p_path_str: 
            continue
        
        current_history_for_prompt: List[Tuple[str, str, str]] = []

        for i_seg, (src_step, rel_step_this_path, tgt_step) in enumerate(path_segments):
            history_key_for_lookup = build_history_key(current_history_for_prompt)
            
            example_signature = (history_key_for_lookup, src_step)
            if example_signature in processed_examples_tracker:
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            processed_examples_tracker.add(example_signature)

            gold_chosen_rels_for_step_set = positive_next_relations[history_key_for_lookup].get(src_step, set())
            
            if not gold_chosen_rels_for_step_set:
                logger.debug(f"SFT Item {sample_id}: No globally positive relations for entity '{src_step}' with history '{history_key_for_lookup}'. Skipping SFT gen.")
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            current_gold_chosen_relations = sorted(list(gold_chosen_rels_for_step_set))
            current_gold_negative_relations = sorted(list(negative_next_relations[history_key_for_lookup].get(src_step, set())))
            
            candidate_pool_for_prompt_set = set()
            relations_from_kg_set = set()

            if kg and config.candidate_strategy in [CandidateStrategy.KG_ALLHOP, CandidateStrategy.PN_KG_SUPPLEMENT]:
                try:
                    relations_from_kg_list = kg.get_related_relations(src_step, "out")
                    if relations_from_kg_list is not None: 
                        relations_from_kg_set.update(relations_from_kg_list)
                except AttributeError:
                    logger.warning(f"KnowledgeGraph class may not be correctly initialized or lacks 'get_related_relations'. Item {sample_id}, Entity {src_step}. KG relations not used.")
                except Exception as e_kg:
                    logger.error(f"Error fetching relations from KG for {src_step}: {e_kg}. KG relations not used.")

            if config.candidate_strategy == CandidateStrategy.PN_ONLY:
                candidate_pool_for_prompt_set.update(current_gold_chosen_relations)
                candidate_pool_for_prompt_set.update(current_gold_negative_relations)
            elif config.candidate_strategy == CandidateStrategy.KG_ALLHOP:
                candidate_pool_for_prompt_set.update(relations_from_kg_set)
            elif config.candidate_strategy == CandidateStrategy.PN_KG_SUPPLEMENT:
                candidate_pool_for_prompt_set.update(current_gold_chosen_relations)
                candidate_pool_for_prompt_set.update(current_gold_negative_relations)
                supplementary_kg_candidates = list(relations_from_kg_set - candidate_pool_for_prompt_set)
                random.shuffle(supplementary_kg_candidates) # Shuffle before taking a slice
                candidate_pool_for_prompt_set.update(supplementary_kg_candidates[:config.num_distractors_to_sample])
            
            final_relations_for_prompt_list: List[str]
            if config.enable_relation_sampling and len(candidate_pool_for_prompt_set) > config.relation_sampling_threshold:
                sampled_set = set(current_gold_chosen_relations) # Prioritize gold chosen
                # Also add gold negative to ensure they might appear as options if strategy includes them
                sampled_set.update(current_gold_negative_relations)
                sampled_set = {r for r in sampled_set if r in candidate_pool_for_prompt_set} # Filter to what's in pool

                potential_distractors = list(candidate_pool_for_prompt_set - sampled_set)
                random.shuffle(potential_distractors)
                
                num_distractors_to_add = config.num_distractors_to_sample
                # Adjust num_distractors_to_add if the pool for sampling is already large due to gold relations
                # We want the total number of options to be roughly around threshold or num_distractors + gold.
                # This logic might need refinement based on desired final prompt size.
                # For simplicity, let's stick to DPO script's way of adding N distractors.
                distractors_to_add = potential_distractors[:num_distractors_to_add]
                sampled_set.update(distractors_to_add)
                final_relations_for_prompt_list = sorted(list(sampled_set))
            else:
                final_relations_for_prompt_list = sorted(list(candidate_pool_for_prompt_set))

            if not final_relations_for_prompt_list:
                logger.debug(f"SFT Item {sample_id}: No relations for prompt after strategy/sampling for '{src_step}', history '{history_key_for_lookup}'.")
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            example = create_sft_example_with_history_and_options(
                question=question,
                history_tuples=current_history_for_prompt,
                current_entity=src_step,
                prompt_available_relations=final_relations_for_prompt_list,
                gold_chosen_relations=current_gold_chosen_relations, 
                max_selection_count=config.max_selection_count,
            )
            
            if example:
                example["metadata"] = {
                    "original_id": sample_id,
                    "sft_id": f"{sample_id}_sft_step{i_seg}_entity_{src_step.replace('.', '_').replace(' ', '_')}_hist{len(history_key_for_lookup)}",
                    "hop_in_path": i_seg,
                    "current_source_entity": src_step,
                    "config_candidate_strategy": config.candidate_strategy.value,
                    "config_positive_source_field": config.positive_source_field.value,
                }
                sft_examples.append(example)
            
            current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
    return sft_examples

def create_sft_dataset(args: argparse.Namespace):
    logger.info(f"Loading path dataset from: {args.input_path}")
    
    path_data: List[Dict[str, Any]]
    if args.input_path.endswith('.json') or args.input_path.endswith('.jsonl'):
        try:
            with open(args.input_path, 'r', encoding='utf-8') as f:
                if args.input_path.endswith('.jsonl'):
                    path_data = [json.loads(line) for line in f]
                else:
                    path_data = json.load(f)
            if not isinstance(path_data, list):
                logger.error(f"File {args.input_path} does not contain a list of items.")
                return
        except Exception as e:
            logger.error(f"Error loading JSON/JSONL from {args.input_path}: {e}", exc_info=True)
            return
    else:
        try:
            hf_dataset = Dataset.load_from_disk(args.input_path)
            path_data = list(hf_dataset)
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset from {args.input_path}: {e}", exc_info=True)
            return
    
    total_items = len(path_data)
    logger.info(f"Loaded {total_items} path items from input.")
    
    if args.num_samples > 0 and args.num_samples < total_items:
        path_data = path_data[:args.num_samples] # Process a subset
        logger.info(f"Processing a subset of {len(path_data)} samples based on --num_samples.")
    
    config = ProcessingConfig(
        max_selection_count=args.max_selection_count,
        enable_relation_sampling=args.enable_relation_sampling,
        relation_sampling_threshold=args.relation_sampling_threshold,
        num_distractors_to_sample=args.num_distractors_to_sample,
        candidate_strategy=CandidateStrategy(args.candidate_strategy),
        positive_source_field=PositiveSource(args.positive_source_field)
    )
    
    logger.info(f"Starting SFT example generation with config: {config}")
    all_sft_examples: List[Dict[str, Any]] = []
    
    kg_instance = None
    if KnowledgeGraph: # Check if class was imported
        try:
            kg_instance = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraph: {e}. Proceeding without KG features.", exc_info=True)
            kg_instance = None # Ensure it's None if init fails
    else:
        logger.warning("KnowledgeGraph module not available. KG-dependent candidate strategies (kg_allhop, pn_kg_supplement) will not use KG data.")

    try:
        for item in tqdm(path_data, desc="Processing path items to SFT examples"):
            try:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping item as it's not a dictionary: {type(item)}")
                    continue
                sft_examples_for_item = process_path_item_for_sft(kg_instance, item, config)
                all_sft_examples.extend(sft_examples_for_item)
            except Exception as e_inner:
                item_id_info = item.get('id', 'unknown_id') if isinstance(item, dict) else 'unknown_item_structure'
                logger.error(f"Error processing item for SFT (ID: {item_id_info}): {e_inner}", exc_info=True)
    except Exception as e_outer:
        logger.error(f"Outer error during SFT example generation loop: {e_outer}", exc_info=True)
    finally:
        if kg_instance and hasattr(kg_instance, 'close'):
            kg_instance.close()
        if not all_sft_examples:
             logger.info("No SFT examples were generated after processing loop (possibly all filtered out or error).")


    if all_sft_examples:
        logger.info(f"SFT example generation complete. Total SFT examples: {len(all_sft_examples)}")
        
        output_name_parts = [
            args.base_output_name, # Renamed from output_name
            f"cand_{config.candidate_strategy.value}",
            f"pos_{config.positive_source_field.value}"
        ]
        dynamic_output_name = "_".join(output_name_parts)
        output_dir = os.path.join(args.output_path, dynamic_output_name)

        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
            return

        jsonl_output_path = os.path.join(output_dir, 'sft_data.jsonl') # Changed from sft_data.json
        try:
            with open(jsonl_output_path, 'w', encoding='utf-8') as f:
                for example in all_sft_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"SFT instruction dataset saved to JSON Lines: {jsonl_output_path}")
        except Exception as e:
            logger.error(f"Failed to save SFT dataset to JSON Lines: {e}", exc_info=True)
        
        try:
            if all(isinstance(ex, dict) for ex in all_sft_examples):
                hf_sft_dataset = Dataset.from_list(all_sft_examples)
                hf_sft_dataset.save_to_disk(output_dir)
                logger.info(f"SFT instruction dataset saved to Hugging Face disk format: {output_dir}")
            else:
                logger.error("Not all generated SFT examples are dictionaries, cannot save to Hugging Face disk format.")
        except Exception as e:
            logger.error(f"Failed to save SFT dataset to Hugging Face disk format in {output_dir}: {e}", exc_info=True)
    else:
        logger.warning("No SFT instruction examples were collected. Nothing to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create SFT instruction dataset from path data, using DPO-like candidate selection.")
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input path-enhanced dataset file (e.g., data.json, data.jsonl) or Hugging Face dataset directory.')
    parser.add_argument('--output_path', type=str, default='./data/processed_sft', # Changed default
                        help='Base output directory for the new SFT dataset.')
    parser.add_argument('--base_output_name', type=str, default='sft_instruct', # Renamed from output_name
                        help='Base name for the output SFT dataset directory (strategy/source info will be appended).')
    
    # Parameters from DPO script's ProcessingConfig
    parser.add_argument('--candidate_strategy', type=str, default=CandidateStrategy.PN_KG_SUPPLEMENT.value,
                        choices=[cs.value for cs in CandidateStrategy],
                        help='Strategy for constructing candidate relations for the prompt.')
    parser.add_argument('--positive_source_field', type=str, default=PositiveSource.POSITIVE_PATHS.value,
                        choices=[ps.value for ps in PositiveSource],
                        help='Field in input data to use as source for positive relations.')
    parser.add_argument('--max_selection_count', type=int, default=3, # Adjusted default for SFT
                        help='Maximum number of relations the model is asked to select / will be in completion.')
    parser.add_argument('--enable_relation_sampling', action='store_true',
                        help='Enable sampling of relations if the candidate pool exceeds a threshold.')
    parser.add_argument('--relation_sampling_threshold', type=int, default=20, # Adjusted default
                        help='Threshold for candidate relations pool size to trigger sampling.')
    parser.add_argument('--num_distractors_to_sample', type=int, default=7, # Adjusted default
                        help='Number of distractor relations to sample if sampling is triggered.')

    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Maximum number of items to process from the input path_data (-1 for all).')
    parser.add_argument('--neo4j_uri', type=str, default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'), help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default=os.getenv('NEO4J_USER', 'neo4j'), help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default=os.getenv('NEO4J_PASSWORD', 'password'), help='Neo4j password')
    
    args = parser.parse_args()

    # Ensure src modules can be imported or provide robust fallbacks
    if KnowledgeGraphTemplates is None or KnowledgeGraph is None:
        logger.critical("Essential 'src' modules (KnowledgeGraph, KnowledgeGraphTemplates) are missing. Please check your setup. Exiting.")
        exit(1)
        
    create_sft_dataset(args)