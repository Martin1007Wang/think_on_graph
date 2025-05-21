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
import random
from enum import Enum

# Assuming these are correctly implemented in your src directory
try:
    from src.knowledge_graph import KnowledgeGraph
    from src.template import KnowledgeGraphTemplates
except ImportError:
    logging.warning("Could not import KnowledgeGraph or KnowledgeGraphTemplates from src. KG-dependent features might not work.")
    class KnowledgeGraph: # type: ignore
        def __init__(self, uri, user, password): pass
        def get_related_relations(self, entity, direction): return []
        def close(self): pass
    class KnowledgeGraphTemplates: # type: ignore
        def format_template(self, template_name, **kwargs):
            prompt_content = f"Question: {kwargs.get('question', '')}\n"
            prompt_content += f"Current Entity: {kwargs.get('entity', '')}\n"
            if kwargs.get('history'):
                prompt_content += f"History:\n{kwargs.get('history')}\n"
            prompt_content += f"Available Relations:\n{kwargs.get('relations', '')}\n"
            prompt_content += f"Select up to {kwargs.get('max_selection_count', 5)} relations.\nYour Selection:"
            return prompt_content

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def parse_path_to_segments(path_str: str) -> List[Tuple[str, str, str]]:
    if not path_str:
        return []
    segments: List[Tuple[str, str, str]] = []
    path_steps = path_str.split(" ; ")
    regex_pattern_to_use = r"(.+?)\s*-\[\s*(.+?)\s*\]->\s*(.+)"
    for i, step in enumerate(path_steps):
        match = re.match(regex_pattern_to_use, step)
        if match:
            src = match.group(1).strip()
            rel = match.group(2).strip()
            tgt = match.group(3).strip()
            segments.append((src, rel, tgt))
        else:
            logger.warning(
                f"Malformed path step. "
                f"Attempted Regex: '{regex_pattern_to_use}'. "
                f"Full Path String: '{path_str}'. "
                f"Problematic Step (repr): {repr(step)}. "
                f"Problematic Step (str): '{step}'."
            )
    return segments

def build_history_key(history_segments: List[Tuple[str, str, str]]) -> str:
    if not history_segments:
        return ""
    return " ; ".join([f"{s}-[{r}]->{t}" for s, r, t in history_segments])

def create_relation_selection_example_with_history(
    question: str,
    history_tuples: List[Tuple[str, str, str]], # These contribute to the user_prompt_content
    current_entity: str,
    prompt_available_relations: List[str],
    gold_chosen_relations: List[str],
    gold_negative_relations: List[str],
    max_selection_count: int,
    template_builder: KnowledgeGraphTemplates
) -> Optional[Dict[str, Any]]:
    unique_prompt_available_relations = sorted(list(set(prompt_available_relations)))
    
    if not unique_prompt_available_relations:
        logger.debug(f"No available relations for prompt for entity '{current_entity}' with history '{build_history_key(history_tuples)}'. Cannot form DPO example.")
        return None

    relation_options_lines = []
    rel_to_display_line: Dict[str, str] = {}
    for i, rel_name in enumerate(unique_prompt_available_relations):
        rel_id_str = f"REL_{i}"
        display_line = f"[{rel_id_str}] {rel_name}"
        relation_options_lines.append(f"      {display_line}")
        rel_to_display_line[rel_name] = display_line
    relation_options_str = "\n".join(relation_options_lines)

    history_section_str = ""
    if history_tuples: # This history is part of the user's current turn content
        history_lines = ["* Exploration History:"]
        for i_hist, (src_hist, rel_hist, tgt_hist) in enumerate(history_tuples):
            history_lines.append(f"  Step {i_hist+1}: From '{src_hist}' explored relation '{rel_hist}', leading to '{tgt_hist}'.")
        history_section_str = "\n".join(history_lines) + "\n"
        template_name = "relation_selection_with_history"
        template_args = {
            "question": question, "entity": current_entity, "history": history_section_str,
            "relations": relation_options_str, "max_selection_count": max_selection_count
        }
    else:
        template_name = "relation_selection"
        template_args = {
            "question": question, "entity": current_entity, "history": "",
            "relations": relation_options_str, "max_selection_count": max_selection_count
        }
    
    user_prompt_content = template_builder.format_template(template_name, **template_args)

    chosen_response_content_lines = []
    for rel_name in gold_chosen_relations:
        if rel_name in rel_to_display_line:
            chosen_response_content_lines.append(rel_to_display_line[rel_name])
    chosen_response_content_lines.sort()
    chosen_response_content_lines = chosen_response_content_lines[:max_selection_count]

    if not chosen_response_content_lines:
        return None
    chosen_response_content_str = "\n".join(chosen_response_content_lines)

    candidate_rejected_display_lines = []
    for rel_name in gold_negative_relations:
        if rel_name in rel_to_display_line and rel_to_display_line[rel_name] not in chosen_response_content_lines:
            candidate_rejected_display_lines.append(rel_to_display_line[rel_name])
    
    other_available_not_chosen_or_gold_negative = []
    for rel_name in unique_prompt_available_relations:
        display_line = rel_to_display_line[rel_name]
        if display_line not in chosen_response_content_lines and display_line not in candidate_rejected_display_lines:
            other_available_not_chosen_or_gold_negative.append(display_line)
    other_available_not_chosen_or_gold_negative.sort()
    
    final_rejected_lines_set = set(candidate_rejected_display_lines)
    for line in other_available_not_chosen_or_gold_negative:
        if len(final_rejected_lines_set) < max_selection_count:
            final_rejected_lines_set.add(line)
        else:
            break
    rejected_response_content_lines = sorted(list(final_rejected_lines_set))[:max_selection_count]

    if not rejected_response_content_lines:
        return None
    rejected_response_content_str = "\n".join(rejected_response_content_lines)

    if chosen_response_content_str == rejected_response_content_str:
        return None

    # Construct the DPO example in the list-of-dictionaries format
    # The "prompt" for DPO is the sequence of turns leading up to the assistant's response.
    # In this specific task, the entire context (question, entity, history, available relations, task description)
    # is presented as a single user turn.
    # dpo_prompt_list = [{"role": "user", "content": user_prompt_content.strip()}]
    
    # # The "chosen" and "rejected" fields are lists containing a single assistant turn.
    # dpo_chosen_list = [{"role": "assistant", "content": chosen_response_content_str.strip()}]
    # dpo_rejected_list = [{"role": "assistant", "content": rejected_response_content_str.strip()}]

    return {
        "prompt": user_prompt_content.strip(),
        "chosen": chosen_response_content_str.strip(),
        "rejected": rejected_response_content_str.strip()
    }

def process_path_item(kg: Optional[KnowledgeGraph], item: Dict[str, Any], config: ProcessingConfig, template_builder: KnowledgeGraphTemplates) -> List[Dict[str, Any]]:
    question = item.get("question", "")
    sample_id = item.get("id", "unknown")
    preference_examples: List[Dict[str, Any]] = []

    positive_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    negative_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    positive_source_key = config.positive_source_field.value
    for p_path_str in item.get(positive_source_key, []):
        segments = parse_path_to_segments(p_path_str)
        if not segments and p_path_str: 
            logger.warning(f"Item {sample_id}: Path from '{positive_source_key}' ('{p_path_str}') could not be parsed. Skipping for positive_next_relations.")
            continue
        current_history_segments: List[Tuple[str,str,str]] = []
        for i, (src, rel, tgt) in enumerate(segments):
            history_key = build_history_key(current_history_segments)
            positive_next_relations[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))

    for n_path_str in item.get('negative_paths', []):
        segments = parse_path_to_segments(n_path_str)
        if not segments and n_path_str:
            logger.warning(f"Item {sample_id}: Negative path '{n_path_str}' could not be parsed. Skipping for negative_next_relations.")
            continue
        current_history_segments: List[Tuple[str,str,str]] = []
        for i, (src, rel, tgt) in enumerate(segments):
            history_key = build_history_key(current_history_segments)
            negative_next_relations[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))
    
    processed_examples_tracker = set()

    for p_path_str in item.get('positive_paths', []):
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
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            current_gold_chosen_relations = sorted(list(gold_chosen_rels_for_step_set))
            current_gold_negative_relations = sorted(list(negative_next_relations[history_key_for_lookup].get(src_step, set())))
            
            candidate_pool_for_prompt_set = set()
            relations_from_kg_set = set()

            if config.candidate_strategy in [CandidateStrategy.KG_ALLHOP, CandidateStrategy.PN_KG_SUPPLEMENT]:
                try:
                    if kg: 
                        relations_from_kg_list = kg.get_related_relations(src_step, "out")
                        if relations_from_kg_list is not None: 
                            relations_from_kg_set.update(relations_from_kg_list)
                except AttributeError:
                    logger.warning(f"KnowledgeGraph class does not have 'get_related_relations' or kg instance is None. Item {sample_id}, Entity {src_step}. KG relations not used.")
                except Exception as e_kg:
                    logger.error(f"Error fetching relations from KG for entity {src_step}: {e_kg}. KG relations not used.")

            if config.candidate_strategy == CandidateStrategy.PN_ONLY:
                candidate_pool_for_prompt_set.update(current_gold_chosen_relations)
                candidate_pool_for_prompt_set.update(current_gold_negative_relations)
            elif config.candidate_strategy == CandidateStrategy.KG_ALLHOP:
                candidate_pool_for_prompt_set.update(relations_from_kg_set)
            elif config.candidate_strategy == CandidateStrategy.PN_KG_SUPPLEMENT:
                candidate_pool_for_prompt_set.update(current_gold_chosen_relations)
                candidate_pool_for_prompt_set.update(current_gold_negative_relations)
                supplementary_kg_candidates = list(relations_from_kg_set - candidate_pool_for_prompt_set)
                random.shuffle(supplementary_kg_candidates)
                candidate_pool_for_prompt_set.update(supplementary_kg_candidates[:config.num_distractors_to_sample])
            
            final_relations_for_prompt_list: List[str]
            if config.enable_relation_sampling and len(candidate_pool_for_prompt_set) > config.relation_sampling_threshold:
                sampled_set = set(current_gold_chosen_relations)
                sampled_set.update(current_gold_negative_relations)
                sampled_set = {r for r in sampled_set if r in candidate_pool_for_prompt_set}
                potential_distractors = list(candidate_pool_for_prompt_set - sampled_set)
                random.shuffle(potential_distractors)
                num_distractors_to_add = config.num_distractors_to_sample
                distractors_to_add = potential_distractors[:num_distractors_to_add]
                sampled_set.update(distractors_to_add)
                final_relations_for_prompt_list = sorted(list(sampled_set))
            else:
                final_relations_for_prompt_list = sorted(list(candidate_pool_for_prompt_set))

            if not final_relations_for_prompt_list:
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            example = create_relation_selection_example_with_history(
                question=question,
                history_tuples=current_history_for_prompt, # This history is used to build the user_prompt_content
                current_entity=src_step,
                prompt_available_relations=final_relations_for_prompt_list,
                gold_chosen_relations=current_gold_chosen_relations, 
                gold_negative_relations=current_gold_negative_relations,
                max_selection_count=config.max_selection_count,
                template_builder=template_builder
            )
            
            if example is None:
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            example["metadata"] = {
                "original_id": sample_id,
                "dpo_id": f"{sample_id}_pathstep{i_seg}_entity_{src_step.replace('.', '_').replace(' ', '_')}_hist{len(history_key_for_lookup)}",
                "hop_in_path": i_seg,
                "current_source_entity": src_step,
                "current_history_key_for_lookup": history_key_for_lookup,
                "current_history_for_prompt_summary": [f"{s}-[{r}]->{t}" for s,r,t in current_history_for_prompt], # Renamed for clarity
                "prompt_available_relations": final_relations_for_prompt_list,
                "step_gold_chosen_relations": current_gold_chosen_relations,
                "step_gold_negative_relations": current_gold_negative_relations,
                "config_candidate_strategy": config.candidate_strategy.value,
                "config_positive_source_field": config.positive_source_field.value,
                "original_q_entity": item.get("q_entity", ""),
                "original_a_entity": item.get("a_entity", ""),
            }
            preference_examples.append(example)
            current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
            
    return preference_examples

def create_preference_dataset(args: argparse.Namespace, template_builder: KnowledgeGraphTemplates) -> None:
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
                logger.error(f"File {args.input_path} does not contain a list of items at the top level.")
                return
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {args.input_path}.", exc_info=True)
            return
        except FileNotFoundError:
            logger.error(f"Input file not found: {args.input_path}")
            return
    else: 
        try:
            hf_dataset = Dataset.load_from_disk(args.input_path)
            path_data = list(hf_dataset) 
        except FileNotFoundError:
            logger.error(f"Hugging Face dataset directory not found: {args.input_path}")
            return
        except Exception as e: 
            logger.error(f"Error loading Hugging Face dataset from {args.input_path}: {e}", exc_info=True)
            return
    
    total_items = len(path_data)
    logger.info(f"Loaded {total_items} path items from input.")
    
    if args.num_samples > 0 and args.num_samples < total_items:
        path_data = path_data[:args.num_samples]
        logger.info(f"Processing a subset of {len(path_data)} samples based on --num_samples.")
    
    config = ProcessingConfig(
        max_selection_count=args.max_selection_count,
        enable_relation_sampling=args.enable_relation_sampling,
        relation_sampling_threshold=args.relation_sampling_threshold,
        num_distractors_to_sample=args.num_distractors_to_sample,
        candidate_strategy=CandidateStrategy(args.candidate_strategy),
        positive_source_field=PositiveSource(args.positive_source_field)
    )
    
    logger.info(f"Starting DPO example generation with config: {config}")
    all_preference_examples: List[Dict[str, Any]] = []
    
    kg_instance: Optional[KnowledgeGraph] = None # Type hint
    if args.candidate_strategy != CandidateStrategy.PN_ONLY.value:
        try:
            if KnowledgeGraph is not None and hasattr(KnowledgeGraph, '__init__'): 
                 kg_instance = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
            else:
                logger.warning("KnowledgeGraph class not available or not a class, KG-dependent strategies will be limited.")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraph: {e}. KG-dependent strategies will be limited.", exc_info=True)
            
    for item in tqdm(path_data, desc="Processing path items to DPO examples"):
        try:
            if not isinstance(item, dict):
                logger.warning(f"Skipping item as it is not a dictionary: {type(item)}")
                continue
            preference_examples_for_item = process_path_item(kg_instance, item, config, template_builder)
            all_preference_examples.extend(preference_examples_for_item)
        except Exception as e_inner:
            item_id_info = item.get('id', 'unknown_id') if isinstance(item, dict) else 'unknown_item_structure'
            logger.error(f"Error processing item (ID: {item_id_info}): {e_inner}", exc_info=True)

    if kg_instance and hasattr(kg_instance, 'close'):
        kg_instance.close()

    if all_preference_examples:
        logger.info(f"DPO example generation complete. Total DPO examples: {len(all_preference_examples)}")
        
        output_name_parts = [
            args.base_output_name,
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

        json_output_path = os.path.join(output_dir, 'preference_data.jsonl')
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                for example in all_preference_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"DPO preference dataset saved to JSON Lines: {json_output_path}")
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save DPO preference dataset to JSON Lines: {e}", exc_info=True)
        
        try:
            if all(isinstance(ex, dict) for ex in all_preference_examples):
                hf_preference_dataset = Dataset.from_list(all_preference_examples)
                hf_preference_dataset.save_to_disk(output_dir)
                logger.info(f"DPO preference dataset saved to Hugging Face disk format: {output_dir}")
            else:
                logger.error("Not all generated examples are dictionaries, cannot save to Hugging Face disk format.")
        except Exception as e:
            logger.error(f"Failed to save DPO preference dataset to Hugging Face disk format in {output_dir}: {e}", exc_info=True)
            if os.path.exists(json_output_path):
                logger.info(f"JSON Lines save to {json_output_path} might still be available.")
    else:
        logger.warning("No DPO preference examples were collected. Nothing to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DPO preference dataset from path-enhanced data, with history and relation sampling.")
    parser.add_argument('--input_path', type=str, required=True,
                        help='Input path-enhanced dataset file (e.g., data.json, data.jsonl) or Hugging Face dataset directory.')
    parser.add_argument('--output_path', type=str, default='./data/processed_dpo',
                        help='Base output directory for the new DPO dataset.')
    parser.add_argument('--base_output_name', type=str, default='dpo_prefs',
                        help='Base name for the output DPO preference dataset directory (strategy/source info will be appended).')
    
    parser.add_argument('--candidate_strategy', type=str, default=CandidateStrategy.PN_KG_SUPPLEMENT.value,
                        choices=[cs.value for cs in CandidateStrategy],
                        help='Strategy for constructing candidate relations for the prompt.')
    parser.add_argument('--positive_source_field', type=str, default=PositiveSource.POSITIVE_PATHS.value,
                        choices=[ps.value for ps in PositiveSource],
                        help='Field in input data to use as source for positive relations (e.g., "positive_paths" or "shortest_paths").')

    parser.add_argument('--max_selection_count', type=int, default=5,
                        help='Maximum number of relations the prompt asks the model to select.')
    parser.add_argument('--enable_relation_sampling', action='store_true',
                        help='Enable sampling of relations if the candidate pool exceeds a threshold.')
    parser.add_argument('--relation_sampling_threshold', type=int, default=25,
                        help='Threshold for candidate relations pool size to trigger sampling.')
    parser.add_argument('--num_distractors_to_sample', type=int, default=10,
                        help='Number of distractor relations to sample if sampling is triggered.')

    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Maximum number of items to process from the input path_data (-1 for all).')
    parser.add_argument('--neo4j_uri', type=str, default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'), help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default=os.getenv('NEO4J_USER', 'neo4j'), help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default=os.getenv('NEO4J_PASSWORD', 'password'), help='Neo4j password')
    
    args = parser.parse_args()

    template_builder_instance = None
    # Ensure KnowledgeGraphTemplates is a class and can be instantiated
    if KnowledgeGraphTemplates is not None and callable(KnowledgeGraphTemplates): 
        try:
            template_builder_instance = KnowledgeGraphTemplates()
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphTemplates: {e}. Ensure 'src.template.KnowledgeGraphTemplates' is correctly defined and callable.")
            exit(1) 
    else:
        logger.error("KnowledgeGraphTemplates class definition not found or not callable. Cannot proceed.")
        exit(1)

    create_preference_dataset(args, template_builder_instance)
