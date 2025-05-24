import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Set, Optional
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from tqdm import tqdm
import logging
from collections import defaultdict
import random
from enum import Enum

from src.knowledge_graph import KnowledgeGraph
from src.template import KnowledgeGraphTemplates

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
    history_tuples: List[Tuple[str, str, str]],
    current_entity: str,
    all_possible_relations_for_step: List[str], # Renamed from prompt_available_relations
    gold_chosen_relations: List[str],
    gold_negative_relations: List[str],
    max_selection_count: int,
    template_builder: KnowledgeGraphTemplates
) -> Optional[Dict[str, Any]]:
    # unique_all_possible_relations is the set of all relations considered
    # candidates for this step, from which 'chosen' and 'rejected' will be derived.
    # They are not explicitly shown in the prompt.
    unique_all_possible_relations = sorted(list(set(all_possible_relations_for_step)))

    if not unique_all_possible_relations:
        logger.debug(f"No possible relations identified for entity '{current_entity}' with history '{build_history_key(history_tuples)}'. Cannot form DPO example.")
        return None

    # Format history section for the prompt
    history_section_str = ""
    if history_tuples:
        history_lines = []
        for i_hist, (src_hist, rel_hist, tgt_hist) in enumerate(history_tuples):
            history_lines.append(f"  Step {i_hist+1}: From '{src_hist}' explored relation '{rel_hist}', leading to '{tgt_hist}'.")
        history_section_str = "\n".join(history_lines)

    # Arguments for the new generation template format
    template_args = {
        "question": question,
        "entity": current_entity,
        "history": history_section_str,
        "max_selection_count": max_selection_count
        # "relations" (the formatted list with REL_X) is no longer passed
    }
    template_name = "relation_generation_with_history" if history_tuples else "relation_generation"
    user_prompt_content = format_template(template_name, **template_args)

    # --- Prepare CHOSEN and REJECTED responses without [REL_X] prefixes ---

    # CHOSEN: Directly from gold_chosen_relations, ensuring they are among the possible ones for this step.
    valid_gold_chosen_relations = [rel for rel in gold_chosen_relations if rel in unique_all_possible_relations]
    chosen_response_content_lines = sorted(list(set(valid_gold_chosen_relations)))[:max_selection_count]

    if not chosen_response_content_lines:
        logger.debug(f"No valid gold_chosen_relations found among all_possible_relations for entity '{current_entity}'. Chosen: {gold_chosen_relations}, Possible: {unique_all_possible_relations}")
        return None
    chosen_response_content_str = "\n".join(chosen_response_content_lines)

    # REJECTED:
    # 1. Start with gold_negative_relations (that are valid and not in chosen).
    # 2. Fill up with other relations from unique_all_possible_relations that are not chosen, up to max_selection_count.
    candidate_rejected_rels = {
        rel for rel in gold_negative_relations
        if rel in unique_all_possible_relations and rel not in chosen_response_content_lines
    }

    other_available_not_chosen = {
        rel for rel in unique_all_possible_relations
        if rel not in chosen_response_content_lines and rel not in candidate_rejected_rels
    }

    final_rejected_lines_set = set(candidate_rejected_rels)
    other_available_list_shuffled = list(other_available_not_chosen)
    random.shuffle(other_available_list_shuffled) # Shuffle to avoid bias

    for rel_name in other_available_list_shuffled:
        if len(final_rejected_lines_set) < max_selection_count:
            final_rejected_lines_set.add(rel_name)
        else:
            break
    
    # Ensure there's at least one rejected line if possible,
    # if we don't have enough from just gold_negative and shuffled others.
    if not final_rejected_lines_set and unique_all_possible_relations:
        potential_rejected = [r for r in unique_all_possible_relations if r not in chosen_response_content_lines]
        random.shuffle(potential_rejected)
        final_rejected_lines_set.update(potential_rejected[:max_selection_count])

    rejected_response_content_lines = sorted(list(final_rejected_lines_set))[:max_selection_count]

    if not rejected_response_content_lines:
        logger.debug(f"No rejected relations could be formed for entity '{current_entity}'. Chosen: {chosen_response_content_lines}, Possible: {unique_all_possible_relations}")
        return None
    rejected_response_content_str = "\n".join(rejected_response_content_lines)

    if chosen_response_content_str == rejected_response_content_str:
        logger.debug(f"Chosen and rejected responses are identical for entity '{current_entity}'. Chosen: {chosen_response_content_str}")
        return None

    # dpo_prompt_list = [{"role": "user", "content": user_prompt_content.strip()}]
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

    for p_path_str in item.get('positive_paths', []): # Iterate over positive paths to define "steps"
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
            
            candidate_pool_for_step_set = set() # This is the universe of possible relations for this step
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
                candidate_pool_for_step_set.update(current_gold_chosen_relations)
                candidate_pool_for_step_set.update(current_gold_negative_relations)
            elif config.candidate_strategy == CandidateStrategy.KG_ALLHOP:
                candidate_pool_for_step_set.update(relations_from_kg_set)
                # Ensure gold relations are in the pool if using KG_ALLHOP,
                # as they form the basis for chosen/rejected.
                candidate_pool_for_step_set.update(current_gold_chosen_relations) 
                candidate_pool_for_step_set.update(current_gold_negative_relations)
            elif config.candidate_strategy == CandidateStrategy.PN_KG_SUPPLEMENT:
                candidate_pool_for_step_set.update(current_gold_chosen_relations)
                candidate_pool_for_step_set.update(current_gold_negative_relations)
                supplementary_kg_candidates = list(relations_from_kg_set - candidate_pool_for_step_set)
                random.shuffle(supplementary_kg_candidates)
                candidate_pool_for_step_set.update(supplementary_kg_candidates[:config.num_distractors_to_sample])
            
            # all_relations_for_step_list is the set of all relations that *could* be generated by the model
            # for this specific step. 'chosen' and 'rejected' will be derived from this.
            all_relations_for_step_list: List[str]
            if config.enable_relation_sampling and len(candidate_pool_for_step_set) > config.relation_sampling_threshold:
                sampled_set = set(current_gold_chosen_relations) # Always include gold chosen
                sampled_set.update(current_gold_negative_relations) # and gold negative
                
                # Ensure sampled relations are within the candidate pool
                sampled_set = {r for r in sampled_set if r in candidate_pool_for_step_set}

                potential_distractors = list(candidate_pool_for_step_set - sampled_set)
                random.shuffle(potential_distractors)
                
                # Calculate how many distractors to add
                num_distractors_to_add = config.num_distractors_to_sample
                
                # Adjust if the total would exceed relation_sampling_threshold
                # The goal is for all_relations_for_step_list to not be excessively large,
                # but to contain gold relations and sufficient distractors.
                
                distractors_to_add = potential_distractors[:num_distractors_to_add]
                sampled_set.update(distractors_to_add)
                
                # If sampled_set is still larger than relation_sampling_threshold, truncate, prioritizing gold.
                if len(sampled_set) > config.relation_sampling_threshold:
                    temp_list = list(sampled_set - set(current_gold_chosen_relations) - set(current_gold_negative_relations))
                    random.shuffle(temp_list)
                    
                    final_sampled_set = set(current_gold_chosen_relations)
                    final_sampled_set.update(current_gold_negative_relations)
                    
                    remaining_slots = config.relation_sampling_threshold - len(final_sampled_set)
                    if remaining_slots > 0:
                        final_sampled_set.update(temp_list[:remaining_slots])
                    all_relations_for_step_list = sorted(list(final_sampled_set))
                else:
                     all_relations_for_step_list = sorted(list(sampled_set))
            else:
                all_relations_for_step_list = sorted(list(candidate_pool_for_step_set))

            if not all_relations_for_step_list:
                # This can happen if candidate_pool_for_step_set was empty.
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            example = create_relation_selection_example_with_history(
                question=question,
                history_tuples=current_history_for_prompt, 
                current_entity=src_step,
                all_possible_relations_for_step=all_relations_for_step_list, # This is the universe from which chosen/rejected are picked
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
                "current_history_for_prompt_summary": [f"{s}-[{r}]->{t}" for s,r,t in current_history_for_prompt],
                "all_possible_relations_for_step": all_relations_for_step_list, # field name updated for clarity
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
    # --- START MODIFICATION FOR MULTIPLE INPUT FILES ---
    logger.info(f"Attempting to load path data from input files: {args.input_files}")
    all_path_data: List[Dict[str, Any]] = []

    for input_file_path in args.input_files:
        logger.info(f"Processing input source: {input_file_path}")
        current_file_data: List[Dict[str, Any]] = []
        if os.path.isdir(input_file_path): # Assume it's a Hugging Face dataset directory
            try:
                hf_dataset = Dataset.load_from_disk(input_file_path)
                current_file_data = list(hf_dataset)
            except FileNotFoundError:
                logger.error(f"Hugging Face dataset directory not found: {input_file_path}. Skipping this source.")
                continue
            except Exception as e:
                logger.error(f"Error loading Hugging Face dataset from {input_file_path}: {e}. Skipping this source.", exc_info=True)
                continue
        elif input_file_path.endswith('.json') or input_file_path.endswith('.jsonl'):
            try:
                with open(input_file_path, 'r', encoding='utf-8') as f:
                    if input_file_path.endswith('.jsonl'):
                        current_file_data = [json.loads(line) for line in f if line.strip()]
                    else: # .json
                        loaded_json = json.load(f)
                        if not isinstance(loaded_json, list):
                            logger.error(f"File {input_file_path} (JSON) does not contain a list of items at the top level. Skipping this file.")
                            continue
                        current_file_data = loaded_json
            except json.JSONDecodeError:
                logger.error(f"Could not decode JSON from {input_file_path}. Skipping this file.", exc_info=True)
                continue
            except FileNotFoundError:
                logger.error(f"Input file not found: {input_file_path}. Skipping this file.")
                continue
        else:
            logger.warning(f"Unsupported file type or path structure for input: {input_file_path}. Skipping. Please provide .json, .jsonl, or a Hugging Face dataset directory.")
            continue

        # Basic validation for items in current_file_data
        if not isinstance(current_file_data, list) or not all(isinstance(item, dict) for item in current_file_data):
            logger.warning(f"Data loaded from {input_file_path} is not a list of dictionaries as expected. Skipping this source.")
            continue
        
        all_path_data.extend(current_file_data)
        logger.info(f"Loaded {len(current_file_data)} items from {input_file_path}. Total items so far: {len(all_path_data)}.")

    if not all_path_data:
        logger.error("No data loaded from any input files. Exiting.")
        return

    total_items = len(all_path_data)
    logger.info(f"Successfully loaded a total of {total_items} path items from all specified input sources.")
    # --- END MODIFICATION FOR MULTIPLE INPUT FILES ---

    if args.num_samples > 0 and args.num_samples < total_items:
        # path_data = path_data[:args.num_samples] # Original line
        all_path_data = all_path_data[:args.num_samples] # Use the combined list
        logger.info(f"Processing a subset of {len(all_path_data)} samples based on --num_samples.")

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

    kg_instance: Optional[KnowledgeGraph] = None
    if args.candidate_strategy != CandidateStrategy.PN_ONLY.value:
        try:
            if isinstance(KnowledgeGraph, type) and hasattr(KnowledgeGraph, '__init__'):
                kg_instance = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
            else: # Should not happen if mock or real class is defined
                logger.warning("KnowledgeGraph class not available or not a proper class, KG-dependent strategies will be limited.")
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraph: {e}. KG-dependent strategies will be limited.", exc_info=True)

    # Process the combined data
    for item in tqdm(all_path_data, desc="Processing path items to DPO examples"):
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
        logger.error("No preference examples generated. Exiting.")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DPO preference dataset from path-enhanced data, with history and relation sampling, using generation prompts.")
    # --- MODIFIED ARGUMENT ---
    parser.add_argument('--input_files', type=str, required=True, nargs='+',
                        help='One or more input path-enhanced dataset files (e.g., data1.json, data2.jsonl) or Hugging Face dataset directories.')
    # --- END MODIFIED ARGUMENT ---
    parser.add_argument('--output_path', type=str, default='./data/processed_dpo',
                        help='Base output directory for the new DPO dataset.')
    parser.add_argument('--base_output_name', type=str, default='dpo_prefs',
                        help='Base name for the output DPO preference dataset directory (strategy/source/prompt_style info will be appended).')

    parser.add_argument('--candidate_strategy', type=str, default=CandidateStrategy.PN_KG_SUPPLEMENT.value,
                        choices=[cs.value for cs in CandidateStrategy],
                        help='Strategy for constructing candidate relations for deriving chosen/rejected responses.')
    parser.add_argument('--positive_source_field', type=str, default=PositiveSource.POSITIVE_PATHS.value,
                        choices=[ps.value for ps in PositiveSource],
                        help='Field in input data to use as source for positive relations (e.g., "positive_paths" or "shortest_paths").')

    parser.add_argument('--max_selection_count', type=int, default=5,
                        help='Maximum number of relations the prompt asks the model to generate (also caps chosen/rejected lists).')
    parser.add_argument('--enable_relation_sampling', action='store_true',
                        help='Enable sampling of relations if the candidate pool (for deriving chosen/rejected) exceeds a threshold.')
    parser.add_argument('--relation_sampling_threshold', type=int, default=25,
                        help='Threshold for candidate relations pool size to trigger sampling.')
    parser.add_argument('--num_distractors_to_sample', type=int, default=10,
                        help='Number of distractor relations to sample if sampling is triggered or PN_KG_SUPPLEMENT is active.')

    parser.add_argument('--num_samples', type=int, default=-1,
                        help='Maximum number of items to process from the combined input path_data (-1 for all).')
    parser.add_argument('--neo4j_uri', type=str, default=os.getenv('NEO4J_URI', 'bolt://localhost:7687'), help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default=os.getenv('NEO4J_USER', 'neo4j'), help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default=os.getenv('NEO4J_PASSWORD', 'password'), help='Neo4j password')

    args = parser.parse_args()

    template_builder_instance = None
    if isinstance(KnowledgeGraphTemplates, type) and callable(KnowledgeGraphTemplates):
        try:
            template_builder_instance = KnowledgeGraphTemplates()
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraphTemplates: {e}. Ensure mock or 'src.template.KnowledgeGraphTemplates' is correctly defined and callable.")
            exit(1)
    else:
        logger.error("KnowledgeGraphTemplates class definition not found or not callable. Cannot proceed.")
        exit(1)

    create_preference_dataset(args, template_builder_instance)