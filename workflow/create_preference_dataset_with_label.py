import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Set, Optional # Added Optional
from dataclasses import dataclass
from datasets import load_dataset, Dataset # type: ignore
from tqdm import tqdm # type: ignore
import logging
from collections import defaultdict

# Assuming these are correctly implemented in your src directory
from src.knowledge_graph import KnowledgeGraph
from src.template import KnowledgeGraphTemplates

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    max_selection_count: int = 5

# This function is kept for potential other uses but not directly by the new process_path_item
def extract_relations_from_path_original(path_str: str) -> List[Tuple[str, str, int]]:
    """Extract relations from a path string with their source entity and hop level."""
    if not path_str:
        return []
    
    relations = []
    path_steps = path_str.split(" ; ")
    
    for i, step in enumerate(path_steps):
        match = re.match(r"([^-]+)-\[([^\]]+)\]->([^-]+)", step)
        if match:
            src, rel, tgt = match.groups()
            relations.append((src.strip(), rel.strip(), i)) # src, rel, hop
    
    return relations

# This function is kept for potential other uses but not directly by the new process_path_item
def group_relations_by_hop_original(paths: List[str]) -> Dict[int, Dict[str, Set[str]]]:
    """Group relations by hop level and source entity."""
    hop_relations: Dict[int, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    
    for path in paths:
        relations_in_path = extract_relations_from_path_original(path)
        for src, rel, hop in relations_in_path:
            hop_relations[hop][src].add(rel)
    
    return hop_relations

def parse_path_to_segments(path_str: str) -> List[Tuple[str, str, str]]:
    """
    Parses a full path string into a list of (source, relation, target) segments.
    Example: "EntityA-[rel1]->EntityB ; EntityB-[rel2]->EntityC"
    Returns: [("EntityA", "rel1", "EntityB"), ("EntityB", "rel2", "EntityC")]
    """
    if not path_str:
        return []
    segments: List[Tuple[str, str, str]] = []
    path_steps = path_str.split(" ; ")
    for step in path_steps:
        match = re.match(r"(.+?)-\[(.+?)\]->(.+)", step)
        if match:
            src, rel, tgt = match.groups()
            segments.append((src.strip(), rel.strip(), tgt.strip()))
        else:
            logger.warning(f"Malformed path step: '{step}' in path '{path_str}'")
    return segments

def build_history_key(history_segments: List[Tuple[str, str, str]]) -> str:
    """Builds a canonical string representation of the history path segments."""
    if not history_segments:
        return ""
    return " ; ".join([f"{s}-[{r}]->{t}" for s, r, t in history_segments])


def create_relation_selection_example_with_history(
    question: str,
    history_tuples: List[Tuple[str, str, str]], 
    current_entity: str,
    available_relations: List[str], 
    chosen_relations_on_path: List[str], 
    negative_relations_on_path: List[str], 
    max_selection_count: int,
    template_builder: KnowledgeGraphTemplates
) -> Optional[Dict[str, Any]]: # Return type changed to Optional
    """
    Create a preference example for relation selection with history.
    Returns None if a valid DPO pair (distinct, non-empty chosen/rejected) cannot be formed.
    """
    unique_available_relations = sorted(list(set(available_relations))) 
    
    # If there are no relations available at all for this step, no example can be formed.
    if not unique_available_relations:
        logger.debug(f"No available relations for entity '{current_entity}' with history. Cannot form DPO example.")
        return None

    relation_options_lines = []
    rel_to_display_line: Dict[str, str] = {}

    for i, rel_name in enumerate(unique_available_relations):
        rel_id_str = f"REL_{i}"
        display_line = f"[{rel_id_str}] {rel_name}"
        relation_options_lines.append(f"    {display_line}")
        rel_to_display_line[rel_name] = display_line
    relation_options_str = "\n".join(relation_options_lines)

    history_section_str = ""
    template_args: Dict[str, Any] 
    if history_tuples:
        history_lines = ["* Exploration History:"]
        for i, (src, rel, tgt) in enumerate(history_tuples):
            history_lines.append(f"  Step {i+1}: From '{src}' explored relation '{rel}', leading to '{tgt}'.")
        history_section_str = "\n".join(history_lines) + "\n"
        template_name = "relation_selection_with_history"
        template_args = {
            "question": question, "entity": current_entity, "history": history_section_str,
            "relations": relation_options_str, "max_selection_count": max_selection_count
        }
    else:
        template_name = "relation_selection"
        template_args = {
            "question": question, "entity": current_entity,
            "relations": relation_options_str, "max_selection_count": max_selection_count
        }
    prompt = template_builder.format_template(template_name, **template_args)

    # --- Create chosen response ---
    chosen_response_lines = []
    for rel_name in chosen_relations_on_path:
        if rel_name in rel_to_display_line: # Ensure the chosen relation is in the presented options
            chosen_response_lines.append(rel_to_display_line[rel_name])
    
    chosen_response_lines.sort() 
    chosen_response_lines = chosen_response_lines[:max_selection_count] 

    # Filter 1: Chosen must have actual relations
    if not chosen_response_lines:
        logger.debug(f"Skipping DPO example for entity '{current_entity}': No valid chosen relations based on positive_paths among available_relations.")
        return None
    chosen_response_str = "\n".join(chosen_response_lines)

    # --- Create rejected response ---
    candidate_rejected_display_lines = []
    # Prioritize explicit negative relations if they are different from chosen
    for rel_name in negative_relations_on_path:
        if rel_name in rel_to_display_line and rel_to_display_line[rel_name] not in chosen_response_lines:
            candidate_rejected_display_lines.append(rel_to_display_line[rel_name])

    num_needed_for_rejection = max_selection_count - len(set(candidate_rejected_display_lines))
    if num_needed_for_rejection > 0:
        other_available_not_chosen = []
        for rel_name in unique_available_relations: # Iterate over all unique relations presented in prompt
            display_line = rel_to_display_line[rel_name]
            if display_line not in chosen_response_lines and display_line not in candidate_rejected_display_lines:
                other_available_not_chosen.append(display_line)
        other_available_not_chosen.sort()
        candidate_rejected_display_lines.extend(other_available_not_chosen[:num_needed_for_rejection])

    rejected_response_lines_set = set()
    final_rejected_lines = []
    for line in candidate_rejected_display_lines:
        if line not in rejected_response_lines_set:
            final_rejected_lines.append(line)
            rejected_response_lines_set.add(line)
    final_rejected_lines.sort()
    rejected_response_lines = final_rejected_lines[:max_selection_count]

    # Filter 2: Rejected must have actual relations
    if not rejected_response_lines:
        logger.debug(f"Skipping DPO example for entity '{current_entity}': No valid rejected relations could be formed from available options.")
        return None
    rejected_response_str = "\n".join(rejected_response_lines)

    # Filter 3: Chosen and Rejected must be different
    if chosen_response_str == rejected_response_str:
        logger.debug(f"Skipping DPO example for entity '{current_entity}': Chosen and Rejected responses are identical. Chosen: '{chosen_response_str}'")
        return None

    return {
        "prompt": prompt,
        "chosen": chosen_response_str,
        "rejected": rejected_response_str
    }

def process_path_item(kg: KnowledgeGraph, item: Dict[str, Any], config: ProcessingConfig, template_builder: KnowledgeGraphTemplates) -> List[Dict[str, Any]]:
    question = item.get("question", "")
    sample_id = item.get("id", "unknown")
    preference_examples: List[Dict[str, Any]] = []

    positive_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    negative_next_relations: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    # all_relations_at_step is used if not fetching from KG, or to ensure path relations are included
    all_relations_at_step_from_paths: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))


    for p_path_str in item.get('positive_paths', []):
        segments = parse_path_to_segments(p_path_str)
        current_history_segments: List[Tuple[str,str,str]] = []
        for i, (src, rel, tgt) in enumerate(segments):
            history_key = build_history_key(current_history_segments)
            positive_next_relations[history_key][src].add(rel)
            all_relations_at_step_from_paths[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))

    for n_path_str in item.get('negative_paths', []):
        segments = parse_path_to_segments(n_path_str)
        current_history_segments: List[Tuple[str,str,str]] = []
        for i, (src, rel, tgt) in enumerate(segments):
            history_key = build_history_key(current_history_segments)
            negative_next_relations[history_key][src].add(rel)
            all_relations_at_step_from_paths[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))
    
    processed_examples_tracker = set() 

    for p_path_str in item.get('positive_paths', []):
        path_segments = parse_path_to_segments(p_path_str)
        current_history_for_prompt: List[Tuple[str, str, str]] = []

        for i, (src_step, rel_step_this_path, tgt_step) in enumerate(path_segments):
            history_key_for_lookup = build_history_key(current_history_for_prompt)
            
            example_signature = (history_key_for_lookup, src_step)
            if example_signature in processed_examples_tracker:
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            processed_examples_tracker.add(example_signature)

            chosen_rels_for_step_set = positive_next_relations[history_key_for_lookup].get(src_step, set())
            # We must have positive relations for this step to be a "chosen" step.
            if not chosen_rels_for_step_set:
                logger.debug(f"Item {sample_id}: No positive relations defined for entity '{src_step}' with history_key '{history_key_for_lookup}' in positive_next_relations. Skipping step.")
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            current_chosen_relations = sorted(list(chosen_rels_for_step_set))
            current_negative_relations = sorted(list(negative_next_relations[history_key_for_lookup].get(src_step, set())))
            
            # Fetching all outgoing relations for the current entity from the KG
            # This is the line from your latest version.
            # Ensure your KnowledgeGraph class has `get_related_relations(entity_name: str, direction: str)`
            # and it returns List[str] (relation names).
            try:
                # Assuming kg.get_related_relations returns a list of relation strings
                relations_from_kg = kg.get_related_relations(src_step, "out") 
                current_available_relations_for_prompt_set = set(relations_from_kg)
                # Ensure relations from this step's positive/negative paths are included
                current_available_relations_for_prompt_set.update(all_relations_at_step_from_paths[history_key_for_lookup].get(src_step, set()))
                current_available_relations_for_prompt = sorted(list(current_available_relations_for_prompt_set))

            except AttributeError:
                logger.error(f"KnowledgeGraph class does not have 'get_related_relations' method. Falling back to path-defined relations for Item {sample_id}, Entity {src_step}.")
                current_available_relations_for_prompt = sorted(list(all_relations_at_step_from_paths[history_key_for_lookup].get(src_step, set())))
            except Exception as e_kg:
                logger.error(f"Error fetching relations from KG for entity {src_step}: {e_kg}. Falling back to path-defined relations.")
                current_available_relations_for_prompt = sorted(list(all_relations_at_step_from_paths[history_key_for_lookup].get(src_step, set())))


            # If, after all efforts, no relations are available to be shown in the prompt, skip.
            if not current_available_relations_for_prompt:
                logger.debug(f"Item {sample_id}: No available relations (from paths or KG) for prompt for entity '{src_step}' with history_key '{history_key_for_lookup}'. Skipping DPO example for this step.")
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue
            
            example = create_relation_selection_example_with_history(
                question=question,
                history_tuples=current_history_for_prompt,
                current_entity=src_step,
                available_relations=current_available_relations_for_prompt,
                chosen_relations_on_path=current_chosen_relations,
                negative_relations_on_path=current_negative_relations,
                max_selection_count=config.max_selection_count,
                template_builder=template_builder
            )
            
            if example is None: # If create_... function returned None due to failed validation
                logger.debug(f"Item {sample_id}, Entity {src_step}, History '{history_key_for_lookup}': Did not generate a valid DPO pair (e.g. empty chosen/rejected or identical).")
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue # Move to the next segment in the current positive path
            
            example["metadata"] = {
                "original_id": sample_id,
                "dpo_id": f"{sample_id}_pathstep{i}_entity_{src_step.replace('.', '_').replace(' ', '_')}_hist{len(history_key_for_lookup)}",
                "hop_in_path": i, 
                "current_source_entity": src_step,
                "current_history_key_for_lookup": history_key_for_lookup,
                "current_history_for_prompt": [f"{s}-[{r}]->{t}" for s,r,t in current_history_for_prompt],
                "prompt_available_relations": current_available_relations_for_prompt,
                "step_chosen_relations": current_chosen_relations,
                "step_negative_relations": current_negative_relations,
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
    
    config = ProcessingConfig(max_selection_count=args.max_selection_count) 
    
    logger.info(f"Starting DPO example generation with max_selection_count={config.max_selection_count}...")
    all_preference_examples: List[Dict[str, Any]] = []
    
    try:
        # Ensure your KnowledgeGraph class is correctly implemented and imported
        # and that it has a method get_related_relations(entity_name: str, direction: str) -> List[str]
        kg = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
    except Exception as e:
        logger.error(f"Failed to initialize KnowledgeGraph: {e}. Ensure 'src.knowledge_graph.KnowledgeGraph' is correctly defined and Neo4j is accessible.")
        return

    
    try:
        for item in tqdm(path_data, desc="Processing path items to DPO examples"):
            try:
                if not isinstance(item, dict):
                    logger.warning(f"Skipping item as it is not a dictionary: {type(item)}")
                    continue
                preference_examples_for_item = process_path_item(kg, item, config, template_builder)
                all_preference_examples.extend(preference_examples_for_item)
            except Exception as e_inner:
                item_id_info = item.get('id', 'unknown_id') if isinstance(item, dict) else 'unknown_item_structure'
                logger.error(f"Error processing item (ID: {item_id_info}): {e_inner}", exc_info=True)
    except Exception as e_outer: 
        logger.error(f"Outer error during DPO example generation loop: {e_outer}", exc_info=True)
    finally: 
        if all_preference_examples:
            logger.info(f"Attempting to save {len(all_preference_examples)} generated DPO examples...")
        else:
            logger.info("No DPO examples were generated to save after processing loop (possibly all filtered out).")

    if all_preference_examples:
        logger.info(f"DPO example generation complete. Total DPO examples: {len(all_preference_examples)}")
        
        output_dir = os.path.join(args.output_path, args.output_name)
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
            hf_preference_dataset = Dataset.from_list(all_preference_examples)
            hf_preference_dataset.save_to_disk(output_dir)
            logger.info(f"DPO preference dataset saved to Hugging Face disk format: {output_dir}")
        except Exception as e: 
            logger.error(f"Failed to save DPO preference dataset to Hugging Face disk format in {output_dir}: {e}", exc_info=True)
            if os.path.exists(json_output_path):
                logger.info(f"JSON Lines save to {json_output_path} might still be available.")
    else:
        logger.warning("No DPO preference examples were collected. Nothing to save.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create DPO preference dataset from path-enhanced data, with history.")
    parser.add_argument('--input_path', type=str, required=True, 
                        help='Input path-enhanced dataset file (e.g., data.json, data.jsonl) or Hugging Face dataset directory.')
    parser.add_argument('--output_path', type=str, default='./data/processed_dpo', 
                        help='Base output directory for the new DPO dataset.')
    parser.add_argument('--output_name', type=str, default='my_dpo_preference_dataset', 
                        help='Directory name for the output DPO preference dataset (under output_path).')
    parser.add_argument('--max_selection_count', type=int, default=5, 
                        help='Maximum number of relations the prompt asks the model to select.')
    parser.add_argument('--num_samples', type=int, default=-1, 
                        help='Maximum number of items to process from the input path_data (-1 for all).')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default='password', help='Neo4j password')
    
    args = parser.parse_args()
    
    # Ensure your KnowledgeGraphTemplates class is correctly implemented and imported
    try:
        template_builder = KnowledgeGraphTemplates() # Assuming it loads templates on init or has a load method
    except Exception as e:
        logger.error(f"Failed to initialize KnowledgeGraphTemplates: {e}. Ensure 'src.template.KnowledgeGraphTemplates' is correctly defined.")
        exit(1) # Exit if templates can't be loaded

    create_preference_dataset(args, template_builder)