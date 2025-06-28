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
from src.knowledge_graph import KnowledgeGraph
from src.template import KnowledgeGraphTemplates

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 预编译正则表达式以提高性能
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

def parse_path_to_segments(path_str: str) -> List[Tuple[str, str, str]]:
    if not path_str:
        return []
    segments: List[Tuple[str, str, str]] = []
    path_steps = path_str.split(" ; ")
    for i, step in enumerate(path_steps):
        match = PATH_STEP_REGEX.match(step) # 使用预编译的正则表达式
        if match:
            src = match.group(1).strip()
            rel = match.group(2).strip()
            tgt = match.group(3).strip()
            segments.append((src, rel, tgt))
        else:
            logger.warning(
                f"Malformed path step. "
                f"Attempted Regex: '{PATH_STEP_REGEX.pattern}'. "
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
    for rel_name in unique_prompt_available_relations:
        display_line = f"{rel_name}"
        relation_options_lines.append(f"      {display_line}") # Indentation for prompt
        rel_to_display_line[rel_name] = display_line
    relation_options_str = "\n".join(relation_options_lines)

    history_section_str = ""
    template_name: str
    if history_tuples:
        history_lines = ["* Exploration History:"]
        for i_hist, (src_hist, rel_hist, tgt_hist) in enumerate(history_tuples):
            history_lines.append(f"  Step {i_hist+1}: From '{src_hist}' explored relation '{rel_hist}', leading to '{tgt_hist}'.")
        history_section_str = "\n".join(history_lines) + "\n"
        template_name = "relation_selection_with_history"
    else:
        template_name = "relation_selection"

    template_args = {
        "question": question, "entity": current_entity, "history": history_section_str,
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
        logger.debug(f"No gold chosen relations found in available relations for entity '{current_entity}'. Skipping DPO example.")
        return None
    chosen_response_content_str = "\n".join(chosen_response_content_lines)

    candidate_rejected_display_lines = []
    for rel_name in gold_negative_relations:
        if rel_name in rel_to_display_line and rel_to_display_line[rel_name] not in chosen_response_content_lines:
            candidate_rejected_display_lines.append(rel_to_display_line[rel_name])

    other_available_not_chosen_or_gold_negative = []
    for rel_name in unique_prompt_available_relations:
        display_line_for_rel = rel_to_display_line[rel_name]
        if display_line_for_rel not in chosen_response_content_lines and \
           display_line_for_rel not in candidate_rejected_display_lines:
            other_available_not_chosen_or_gold_negative.append(display_line_for_rel)
    other_available_not_chosen_or_gold_negative.sort()

    final_rejected_lines_set = set(candidate_rejected_display_lines)
    for line in other_available_not_chosen_or_gold_negative:
        if len(final_rejected_lines_set) < max_selection_count:
            final_rejected_lines_set.add(line)
        else:
            break

    rejected_response_content_lines = sorted(list(final_rejected_lines_set))[:max_selection_count]

    if not rejected_response_content_lines:
        logger.debug(f"Could not form a non-empty rejected response for entity '{current_entity}'. Skipping DPO example.")
        return None
    rejected_response_content_str = "\n".join(rejected_response_content_lines)

    if chosen_response_content_str == rejected_response_content_str:
        logger.debug(f"Chosen and rejected responses are identical for entity '{current_entity}'. Skipping DPO example.")
        return None

    return {
        "prompt": user_prompt_content.strip(),
        "chosen": chosen_response_content_str.strip(),
        "rejected": rejected_response_content_str.strip()
    }

def process_path_item(
    kg: Optional[KnowledgeGraph],
    item: Dict[str, Any],
    config: ProcessingConfig,
    template_builder: KnowledgeGraphTemplates,
    kg_relations_cache: Dict[str, Set[str]] # 使用共享的KG缓存
) -> List[Dict[str, Any]]:
    question = item.get("question", "")
    sample_id = str(item.get("id", f"unknown_sample_{random.getrandbits(32)}")) # 确保sample_id是字符串且有默认值
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
        for src, rel, tgt in segments:
            history_key = build_history_key(current_history_segments)
            positive_next_relations[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))

    for n_path_str in item.get('negative_paths', []):
        segments = parse_path_to_segments(n_path_str)
        if not segments and n_path_str:
            logger.warning(f"Item {sample_id}: Negative path '{n_path_str}' could not be parsed. Skipping for negative_next_relations.")
            continue
        current_history_segments: List[Tuple[str,str,str]] = []
        for src, rel, tgt in segments:
            history_key = build_history_key(current_history_segments)
            negative_next_relations[history_key][src].add(rel)
            current_history_segments.append((src, rel, tgt))

    processed_examples_tracker: Set[Tuple[str, str]] = set()

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

            candidate_pool_for_prompt_set: Set[str] = set()
            relations_from_kg_this_step_set: Set[str] = set()

            if config.candidate_strategy in [CandidateStrategy.KG_ALLHOP, CandidateStrategy.PN_KG_SUPPLEMENT]:
                if src_step in kg_relations_cache:
                    relations_from_kg_this_step_set.update(kg_relations_cache[src_step])
                elif kg:
                    try:
                        relations_from_kg_list = kg.get_related_relations(src_step, "out")
                        if relations_from_kg_list is not None:
                            relations_from_kg_this_step_set.update(r.strip() for r in relations_from_kg_list if r and r.strip())
                        kg_relations_cache[src_step] = relations_from_kg_this_step_set # 更新共享缓存
                    except Exception as e_kg:
                        logger.error(f"Error fetching relations from KG for entity {src_step} (Item {sample_id}): {e_kg}. KG relations not used for this step.")

            if config.candidate_strategy == CandidateStrategy.PN_ONLY:
                candidate_pool_for_prompt_set.update(current_gold_chosen_relations)
                candidate_pool_for_prompt_set.update(current_gold_negative_relations)
            elif config.candidate_strategy == CandidateStrategy.KG_ALLHOP:
                candidate_pool_for_prompt_set.update(relations_from_kg_this_step_set)
            elif config.candidate_strategy == CandidateStrategy.PN_KG_SUPPLEMENT:
                candidate_pool_for_prompt_set.update(current_gold_chosen_relations)
                candidate_pool_for_prompt_set.update(current_gold_negative_relations)
                supplementary_kg_candidates = list(relations_from_kg_this_step_set - candidate_pool_for_prompt_set)
                random.shuffle(supplementary_kg_candidates)
                candidate_pool_for_prompt_set.update(supplementary_kg_candidates[:config.num_distractors_to_sample])

            final_relations_for_prompt_list: List[str]
            if config.enable_relation_sampling and len(candidate_pool_for_prompt_set) > config.relation_sampling_threshold:
                sampled_set = set(r for r in current_gold_chosen_relations if r in candidate_pool_for_prompt_set)
                sampled_set.update(r for r in current_gold_negative_relations if r in candidate_pool_for_prompt_set)
                
                potential_distractors = list(candidate_pool_for_prompt_set - sampled_set)
                random.shuffle(potential_distractors)
                
                num_needed_distractors = config.num_distractors_to_sample
                distractors_to_add = potential_distractors[:num_needed_distractors]
                sampled_set.update(distractors_to_add)
                final_relations_for_prompt_list = sorted(list(sampled_set))
            else:
                final_relations_for_prompt_list = sorted(list(candidate_pool_for_prompt_set))

            if not final_relations_for_prompt_list:
                current_history_for_prompt.append((src_step, rel_step_this_path, tgt_step))
                continue

            example = create_relation_selection_example_with_history(
                question=question,
                history_tuples=current_history_for_prompt,
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
                "original_dataset_source": item.get("dataset_source_name", "unknown_source"), # 添加原始数据集来源信息
                "dpo_id": f"{sample_id}_pathstep{i_seg}_entity_{src_step.replace('.', '_').replace(' ', '_')}_histlen{len(current_history_for_prompt)}",
                "hop_in_path": i_seg,
                "current_source_entity": src_step,
                "current_history_key_for_lookup": history_key_for_lookup,
                "current_history_for_prompt_summary": [f"{s}-[{r}]->{t}" for s,r,t in current_history_for_prompt],
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

def load_path_data_from_single_source(input_path_str: str, num_samples_to_load: int, original_dataset_name_for_metadata: str) -> List[Dict[str, Any]]:
    path_data: List[Dict[str, Any]] = []
    logger.info(f"Attempting to load path data from: {input_path_str} (Source Name: {original_dataset_name_for_metadata})")
    if input_path_str.endswith(('.json', '.jsonl')):
        try:
            with open(input_path_str, 'r', encoding='utf-8') as f:
                if input_path_str.endswith('.jsonl'):
                    path_data = [json.loads(line) for line in f if line.strip()]
                else: # .json
                    loaded_json = json.load(f)
                    if isinstance(loaded_json, list):
                        path_data = loaded_json
                    else:
                        logger.warning(f"File {input_path_str} is a single JSON object, not a list. Attempting to treat as a dataset dictionary.")
                        if isinstance(loaded_json, dict):
                            for key in ["data", "train", "test", "validation"]:
                                if key in loaded_json and isinstance(loaded_json[key], list):
                                    path_data = loaded_json[key]
                                    logger.info(f"Using list from key '{key}' in {input_path_str}.")
                                    break
                            if not path_data:
                                logger.error(f"File {input_path_str} is a single JSON object and no suitable list of records found.")
                                return []
                        else:
                            logger.error(f"File {input_path_str} does not contain a list of items at the top level.")
                            return []
            if not path_data or not isinstance(path_data, list) or not all(isinstance(item, dict) for item in path_data):
                logger.error(f"Data loaded from {input_path_str} is not a list of dictionaries.")
                return []
        except json.JSONDecodeError:
            logger.error(f"Could not decode JSON from {input_path_str}.", exc_info=True)
            return []
        except FileNotFoundError:
            logger.error(f"Input file not found: {input_path_str}")
            return []
    else: # Assume it's a Hugging Face dataset directory
        try:
            hf_dataset = Dataset.load_from_disk(input_path_str)
            path_data = list(hf_dataset)
        except FileNotFoundError:
            logger.error(f"Hugging Face dataset directory not found: {input_path_str}")
            return []
        except Exception as e:
            logger.error(f"Error loading Hugging Face dataset from {input_path_str}: {e}", exc_info=True)
            return []

    total_items_loaded = len(path_data)
    logger.info(f"Loaded {total_items_loaded} path items from {input_path_str} (Source: {original_dataset_name_for_metadata}).")

    # 为每个item添加原始数据集名称元数据，以便后续追踪
    for item in path_data:
        if isinstance(item, dict):
            item["dataset_source_name"] = original_dataset_name_for_metadata


    if num_samples_to_load > 0 and num_samples_to_load < total_items_loaded:
        path_data = path_data[:num_samples_to_load] # 保持顺序，所以用切片
        logger.info(f"Processing a subset of {len(path_data)} samples from {input_path_str} (Source: {original_dataset_name_for_metadata}) based on --num_samples.")
    elif num_samples_to_load == 0 :
        logger.info(f"Processing 0 samples from {input_path_str} (Source: {original_dataset_name_for_metadata}) as per --num_samples=0.")
        return []
        
    return path_data

def create_preference_dataset_for_source(
    input_path_str: str,
    original_dataset_name: str, # 用于日志和tqdm
    num_samples_to_load: int,
    config: ProcessingConfig,
    template_builder: KnowledgeGraphTemplates,
    kg_instance: Optional[KnowledgeGraph],
    kg_relations_cache: Dict[str, Set[str]] # 传入共享的KG缓存
) -> List[Dict[str, Any]]:
    """为单个输入数据源处理并返回偏好数据列表。"""
    logger.info(f"===== Processing dataset source: {input_path_str} (Original Name: {original_dataset_name}) =====")
    # 在加载时传入 original_dataset_name 以便item可以携带此信息
    path_data = load_path_data_from_single_source(input_path_str, num_samples_to_load, original_dataset_name)


    if not path_data:
        logger.warning(f"No data loaded or 0 samples requested from {input_path_str} (Source: {original_dataset_name}). Skipping preference dataset generation for this source.")
        return []

    source_preference_examples: List[Dict[str, Any]] = []

    for item in tqdm(path_data, desc=f"Processing items from [{original_dataset_name}] for DPO config [{config.candidate_strategy.value}/{config.positive_source_field.value}]"):
        try:
            if not isinstance(item, dict):
                logger.warning(f"Skipping item as it is not a dictionary: {type(item)} from {original_dataset_name}")
                continue
            
            # 'id' 和 'dataset_source_name' 应该已在 load_path_data_from_single_source 中处理或添加
            item_to_process = item # item 已经包含了 dataset_source_name

            preference_examples_for_item = process_path_item(
                kg_instance,
                item_to_process,
                config,
                template_builder,
                kg_relations_cache
            )
            source_preference_examples.extend(preference_examples_for_item)
        except Exception as e_inner:
            item_id_info = item.get('id', 'unknown_id_in_error') if isinstance(item, dict) else 'unknown_item_structure_in_error'
            logger.error(f"Error processing item (ID: {item_id_info} from {original_dataset_name}): {e_inner}", exc_info=True)
    
    logger.info(f"Finished processing dataset source: {original_dataset_name}. Generated {len(source_preference_examples)} DPO examples from this source.")
    return source_preference_examples


def main_create_preference_dataset(args: argparse.Namespace, template_builder: KnowledgeGraphTemplates) -> None:
    kg_instance: Optional[KnowledgeGraph] = None
    kg_relations_cache: Dict[str, Set[str]] = {} # 在main函数级别初始化KG缓存

    if CandidateStrategy(args.candidate_strategy) != CandidateStrategy.PN_ONLY:
        try:
            logger.info(f"Initializing KnowledgeGraph with URI: {args.neo4j_uri}")
            kg_instance = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        except Exception as e:
            logger.error(f"Failed to initialize KnowledgeGraph: {e}. KG-dependent strategies will be limited.", exc_info=True)

    current_run_config = ProcessingConfig(
        max_selection_count=args.max_selection_count,
        enable_relation_sampling=args.enable_relation_sampling,
        relation_sampling_threshold=args.relation_sampling_threshold,
        num_distractors_to_sample=args.num_distractors_to_sample,
        candidate_strategy=CandidateStrategy(args.candidate_strategy),
        positive_source_field=PositiveSource(args.positive_source_field)
    )
    logger.info(f"Processing with configuration: {current_run_config}")

    all_combined_dpo_examples: List[Dict[str, Any]] = []

    for i, input_path_str in enumerate(args.input_paths):
        original_dataset_name = args.input_dataset_names[i] if args.input_dataset_names and i < len(args.input_dataset_names) else os.path.splitext(os.path.basename(input_path_str))[0]

        examples_from_source = create_preference_dataset_for_source(
            input_path_str=input_path_str,
            original_dataset_name=original_dataset_name,
            num_samples_to_load=args.num_samples,
            config=current_run_config,
            template_builder=template_builder,
            kg_instance=kg_instance,
            kg_relations_cache=kg_relations_cache # 传递共享的KG缓存
        )
        all_combined_dpo_examples.extend(examples_from_source) # 按顺序合并

    if all_combined_dpo_examples:
        logger.info(f"Total DPO examples combined from all sources for config [{current_run_config.candidate_strategy.value} | {current_run_config.positive_source_field.value}]: {len(all_combined_dpo_examples)}")

        # 构建合并后的输出目录名，不再包含单个数据集名
        combined_output_folder_name_parts = [
            f"cand_{current_run_config.candidate_strategy.value}",
            f"pos_{current_run_config.positive_source_field.value}",
            "combined" # 添加 "combined" 后缀以明确表示
        ]
        combined_output_folder_name = "_".join(combined_output_folder_name_parts)
        final_output_dir = os.path.join(args.output_path, combined_output_folder_name)

        try:
            os.makedirs(final_output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create combined output directory {final_output_dir}: {e}", exc_info=True)
            if kg_instance and hasattr(kg_instance, 'close'): kg_instance.close()
            return

        json_output_path = os.path.join(final_output_dir, 'preference_data.jsonl')
        try:
            with open(json_output_path, 'w', encoding='utf-8') as f:
                for example in all_combined_dpo_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            logger.info(f"Combined DPO preference dataset saved to JSON Lines: {json_output_path}")
        except (IOError, TypeError) as e:
            logger.error(f"Failed to save combined DPO preference dataset to JSON Lines: {e}", exc_info=True)

        try:
            if all(isinstance(ex, dict) for ex in all_combined_dpo_examples):
                hf_preference_dataset = Dataset.from_list(all_combined_dpo_examples)
                hf_dataset_path = os.path.join(final_output_dir, "hf_dataset")
                # os.makedirs(hf_dataset_path, exist_ok=True) # save_to_disk会自己创建
                hf_preference_dataset.save_to_disk(hf_dataset_path)
                logger.info(f"Combined DPO preference dataset saved to Hugging Face disk format: {hf_dataset_path}")
            else:
                logger.error("Not all generated combined examples are dictionaries, cannot save to Hugging Face disk format.")
        except Exception as e:
            logger.error(f"Failed to save combined DPO preference dataset to Hugging Face disk format in {final_output_dir}: {e}", exc_info=True)
    else:
        logger.warning(f"No DPO preference examples were collected from any source for config [{current_run_config.candidate_strategy.value} | {current_run_config.positive_source_field.value}]. Nothing to save.")

    if kg_instance and hasattr(kg_instance, 'close'):
        logger.info("Closing KnowledgeGraph connection.")
        kg_instance.close()

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

    main_create_preference_dataset(args, template_builder_instance)