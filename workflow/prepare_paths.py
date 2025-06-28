import torch
import json
import argparse
import os
import re
from typing import List, Tuple, Dict, Any, Set
from dataclasses import dataclass
from datasets import load_dataset, Dataset # type: ignore
from tqdm import tqdm # type: ignore
import logging
from collections import defaultdict
import concurrent.futures
from src.path_generator import PathGenerator
from src.knowledge_graph import KnowledgeGraph

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')
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

def path_to_tuple(path: List[Tuple[str, str, str]]) -> Tuple[Tuple[str, str, str], ...]:
    return tuple(path) if path else tuple()

def _parse_entities(entities_raw: Any) -> List[str]:
    if isinstance(entities_raw, str) and entities_raw.strip():
        return [entities_raw.strip()]
    if isinstance(entities_raw, list):
        return [str(item).strip() for item in entities_raw if item and str(item).strip()]
    return []

def process_entity_pair(q_entity: str, a_entity: str, question: str, path_generator: PathGenerator, max_negatives_per_pair: int) -> Dict[str, Any]:
    if not q_entity or not a_entity:
        return {"shortest_paths": [], "semantic_paths": [], "negative_paths": [], "positive_paths": [] }

    def format_unique_paths_from_list_of_paths(list_of_raw_paths: List[List[Tuple[str, str, str]]]) -> List[str]:
        if not list_of_raw_paths: return []
        unique_path_structures = {path_to_tuple(p) for p in list_of_raw_paths if p is not None and p}
        return [format_path_for_json(list(p_struct)) for p_struct in unique_path_structures]

    raw_shortest_paths = path_generator.get_shortest_paths(q_entity, a_entity)
    raw_shortest_paths = raw_shortest_paths if raw_shortest_paths is not None else []
    formatted_shortest_paths = format_unique_paths_from_list_of_paths(raw_shortest_paths)
    
    # 假设没有专门的 semantic_paths 查找，或者它已合并到 positive_paths 逻辑中
    formatted_semantic_paths: List[str] = [] # 保持字段存在，但可能为空

    all_positive_path_structures: Set[Tuple[Tuple[str, str, str], ...]] = set()
    if raw_shortest_paths: # 假设所有找到的最短路径都是正路径
        all_positive_path_structures.update({path_to_tuple(p) for p in raw_shortest_paths if p})
    
    # 如果有专门的语义路径查找，并希望将它们也视为正路径：
    # raw_semantic_paths = path_generator.get_semantic_paths(q_entity, a_entity, question) # 假设有此方法
    # if raw_semantic_paths:
    #     all_positive_path_structures.update({path_to_tuple(p) for p in raw_semantic_paths if p})

    formatted_positive_paths = [format_path_for_json(list(p_struct)) for p_struct in all_positive_path_structures if p_struct]
    
    list_of_raw_negative_paths: List[List[Tuple[str, str, str]]] = []
    if question and all_positive_path_structures:
        for p_struct_tuple in all_positive_path_structures:
            if not p_struct_tuple: continue
            negs_for_p = path_generator.get_negative_paths(list(p_struct_tuple), question, a_entity, max_negatives_per_pair)
            if negs_for_p: list_of_raw_negative_paths.extend(negs_for_p)
            
    formatted_negative_paths = format_unique_paths_from_list_of_paths(list_of_raw_negative_paths)
    
    return {
        "shortest_paths": formatted_shortest_paths, # 可能与 positive_paths 重叠或为其子集
        "semantic_paths": formatted_semantic_paths, # 独立字段，按需填充
        "negative_paths": formatted_negative_paths,
        "positive_paths": formatted_positive_paths # 包含所有认为是"好"的路径
    }

def process_sample_task(sample_tuple: Tuple[Dict[str, Any], ProcessingConfig, PathGenerator]) -> List[Dict[str, Any]]:
    sample, config, path_generator = sample_tuple
    dataset_tag = sample.get('_dataset_tag', '') # 获取传递过来的 dataset_tag
    try:
        q_entities = _parse_entities(sample.get('q_entity'))
        a_entities = _parse_entities(sample.get('a_entity', []))
        question = sample.get('question', "").strip()
        sample_id = sample.get('id', 'unknown_sample_id')
        
        all_path_results_for_sample: List[Dict[str, Any]] = []

        if not q_entities:
            logger.debug(f"{dataset_tag} Sample {sample_id} has no valid query entities. Skipping.")
            return all_path_results_for_sample
        if not a_entities:
            logger.debug(f"{dataset_tag} Sample {sample_id} has no valid answer entities. Skipping generation for this sample.")
            # 仍可返回一个空的，或带有id信息的条目，表明它被“看到”了但没有处理
            # return [{"id": sample_id, "question": question, "q_entity": q_entities[0] if q_entities else "", "a_entity": "", "positive_paths": [], "negative_paths": [], "error": "No answer entities"}]
            return all_path_results_for_sample


        for q_entity_item in q_entities:
            pairs_processed_for_q_entity = 0
            for a_entity_item in a_entities:
                if pairs_processed_for_q_entity >= config.max_pairs:
                    logger.debug(f"{dataset_tag} Reached max_pairs ({config.max_pairs}) for q_entity '{q_entity_item}' in sample {sample_id}.")
                    break
                
                # logger.debug(f"{dataset_tag} Processing pair: ('{q_entity_item}', '{a_entity_item}') for sample {sample_id}")
                pair_path_data = process_entity_pair(q_entity_item, a_entity_item, question, path_generator, config.max_negatives_per_pair)
                
                if pair_path_data["positive_paths"] or pair_path_data["negative_paths"]:
                    result_item = {
                        "id": sample_id,
                        "question": question,
                        "q_entity": q_entity_item,
                        "a_entity": a_entity_item,
                        **pair_path_data
                    }
                    all_path_results_for_sample.append(result_item)
                # else:
                    # logger.debug(f"{dataset_tag} No positive or negative paths found for pair ('{q_entity_item}', '{a_entity_item}') in sample {sample_id}.")

                pairs_processed_for_q_entity += 1
        
        # if not all_path_results_for_sample:
            # logger.debug(f"{dataset_tag} No path data generated for sample {sample_id} after processing all pairs.")

        return all_path_results_for_sample
    except Exception as e:
        sample_id_info = sample.get('id', 'unknown_id') if isinstance(sample, dict) else 'unknown_sample_structure'
        logger.error(f"{dataset_tag} Error processing sample task (ID: {sample_id_info}): {e}", exc_info=True)
        return []


def sanitize_for_path(name: str) -> str:
    name = name.replace('/', '_').replace('\\', '_')
    name = re.sub(r'[^\w\s_.-]', '', name)
    name = re.sub(r'\s+', '_', name).strip('_')
    return name if name else "unnamed_dataset"


def prepare_paths(args: argparse.Namespace):
    kg = None
    try:
        logger.info("Initializing knowledge graph (once for all datasets)...")
        kg = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        
        logger.info("Initializing KG embeddings (once for all datasets)...")
        kg.initialize_embeddings(
            model_name=args.model_name,
            embedding_encode_batch_size=args.embedding_encode_batch_size,
            force_recompute=True
        )

        logger.info("Initializing path generator (once for all datasets)...")
        processing_config = ProcessingConfig(
            max_pairs=args.max_pairs,
            max_negatives_per_pair=args.max_negatives_per_pair,
            max_path_length=args.max_path_length,
            top_k_relations=args.top_k_relations
        )
        path_generator = PathGenerator(
            kg=kg,
            max_path_length=processing_config.max_path_length,
            top_k_relations=processing_config.top_k_relations
        )

        if len(args.dataset_inputs) != len(args.splits):
            logger.error("Error: The number of --dataset_inputs must match the number of --splits.")
            # ... (error logging) ...
            return

        for i in range(len(args.dataset_inputs)):
            current_dataset_input = args.dataset_inputs[i]
            current_split = args.splits[i]
            dataset_tag = f"[{sanitize_for_path(current_dataset_input)}_{current_split}]"
            logger.info(f"====== {dataset_tag} Starting processing for dataset: '{current_dataset_input}', split: '{current_split}' ({i+1}/{len(args.dataset_inputs)}) ======")

            base_name = os.path.basename(current_dataset_input) if os.path.exists(current_dataset_input) else current_dataset_input
            cleaned_dataset_name_for_path = sanitize_for_path(base_name)
            current_output_sub_dir_name = f"{cleaned_dataset_name_for_path}_{current_split}"
            current_output_dir = os.path.join(args.output_path, current_output_sub_dir_name)
            
            try:
                os.makedirs(current_output_dir, exist_ok=True)
                logger.info(f"{dataset_tag} Output directory: {current_output_dir}")
            except OSError as e:
                logger.error(f"{dataset_tag} Failed to create output directory {current_output_dir}: {e}", exc_info=True)
                continue

            temp_jsonl_output_path = os.path.join(current_output_dir, 'path_data_temp.jsonl')
            processed_sample_ids: Set[str] = set()
            resuming_from_temp = False
            file_open_mode = 'w' # Default to write

            if args.resume_processing and os.path.exists(temp_jsonl_output_path):
                logger.info(f"{dataset_tag} Found existing temp file: {temp_jsonl_output_path}. Attempting to load processed IDs for resumption.")
                try:
                    with open(temp_jsonl_output_path, 'r', encoding='utf-8') as f_temp_read:
                        for line_num, line in enumerate(f_temp_read):
                            try:
                                item = json.loads(line)
                                # 每个写入的item都应该有'id'，这个'id'对应原始样本的ID
                                if 'id' in item:
                                    processed_sample_ids.add(item['id'])
                                else:
                                    logger.warning(f"{dataset_tag} Item at line {line_num+1} in temp file lacks 'id' key. Temp file: {temp_jsonl_output_path}")
                            except json.JSONDecodeError:
                                logger.warning(f"{dataset_tag} Skipping malformed JSON line {line_num+1} in temp file: {temp_jsonl_output_path} Content: {line.strip()}")
                    if processed_sample_ids:
                        logger.info(f"{dataset_tag} Loaded {len(processed_sample_ids)} unique sample IDs from temp file. Will skip these and append new results.")
                        resuming_from_temp = True
                        file_open_mode = 'a' # Switch to append mode
                    else:
                        logger.info(f"{dataset_tag} Temp file was empty or no valid IDs found. Starting fresh (overwrite).")
                        # file_open_mode remains 'w'
                except Exception as e_read_temp:
                    logger.error(f"{dataset_tag} Error reading existing temp file {temp_jsonl_output_path}: {e_read_temp}. Will process from scratch (overwrite).", exc_info=True)
                    # file_open_mode remains 'w', potentially overwriting a corrupted file

            dataset_iterable: Any
            logger.info(f"{dataset_tag} Loading dataset from: {current_dataset_input}")
            try:
                # ... (dataset loading logic as before) ...
                if os.path.exists(current_dataset_input):
                    if current_dataset_input.endswith(".jsonl"):
                        dataset_list_concrete = []
                        with open(current_dataset_input, 'r', encoding='utf-8') as f:
                            for line in f: dataset_list_concrete.append(json.loads(line))
                        dataset_iterable = dataset_list_concrete
                    elif current_dataset_input.endswith(".json"):
                        with open(current_dataset_input, 'r', encoding='utf-8') as f:
                            dataset_list_concrete = json.load(f)
                        if not isinstance(dataset_list_concrete, list):
                            logger.error(f"{dataset_tag} JSON file {current_dataset_input} does not contain a list.")
                            continue
                        dataset_iterable = dataset_list_concrete
                    else:
                        from datasets import load_from_disk
                        dataset_iterable = load_from_disk(current_dataset_input)
                else:
                    dataset_iterable = load_dataset(current_dataset_input, name=args.dataset_config_name, split=current_split) # Added dataset_config_name
                logger.info(f"{dataset_tag} Dataset loaded successfully.")

                # Ensure dataset_iterable is a list before filtering or slicing
                # This materialization is important for consistent handling
                if not isinstance(dataset_iterable, list):
                    logger.info(f"{dataset_tag} Converting dataset_iterable to list...")
                    dataset_iterable = list(tqdm(dataset_iterable, desc=f"{dataset_tag} Materializing dataset"))

                # Filter out already processed samples if resuming
                if resuming_from_temp and processed_sample_ids:
                    original_len = len(dataset_iterable)
                    dataset_iterable = [sample for sample in dataset_iterable if sample.get('id') not in processed_sample_ids]
                    logger.info(f"{dataset_tag} Resuming: Filtered out {original_len - len(dataset_iterable)} already processed samples based on IDs in temp file.")
                
                # Apply num_samples to the (potentially filtered) dataset_iterable
                if args.num_samples > 0:
                    if len(dataset_iterable) > args.num_samples:
                         logger.info(f"{dataset_tag} Applying num_samples limit: {args.num_samples} (on remaining/total samples)")
                         dataset_iterable = dataset_iterable[:args.num_samples]
                
                logger.info(f"{dataset_tag} Effective number of samples to process in this run: {len(dataset_iterable)}")
                if not dataset_iterable and resuming_from_temp:
                    logger.info(f"{dataset_tag} All samples were already processed according to the temp file. Nothing new to process for this dataset.")
                    # Continue to finalization, which will use the existing temp file
                elif not dataset_iterable:
                     logger.info(f"{dataset_tag} No samples to process after loading/filtering (and not resuming or temp file was empty).")
                     # Skip processing, proceed to finalization (which will likely find no data)


            except Exception as e:
                logger.error(f"{dataset_tag} Failed to load or prepare dataset '{current_dataset_input}' (split: {current_split}): {e}", exc_info=True)
                continue

            # Per-dataset processing logic
            processed_items_count_current_run = 0
            if dataset_iterable: # Only proceed if there are samples to process in this run
                try:
                    logger.info(f"{dataset_tag} Preparing tasks for path generation...")
                    # Add dataset_tag to sample for better logging within process_sample_task
                    tasks_args = [( {**sample, '_dataset_tag': dataset_tag}, processing_config, path_generator) 
                                  for sample in tqdm(dataset_iterable, desc=f"{dataset_tag} Preparing tasks")]
                    total_tasks = len(tasks_args)
                    logger.info(f"{dataset_tag} Prepared to process {total_tasks} new samples.")
                    
                    num_threads = args.num_threads if args.num_threads > 0 else min(32, (os.cpu_count() or 1) + 4)
                    
                    # Open temp file in determined mode (append or write)
                    with open(temp_jsonl_output_path, file_open_mode, encoding='utf-8') as f_jsonl:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                            results_iterator = executor.map(process_sample_task, tasks_args)
                            
                            for sample_path_results_list in tqdm(results_iterator, total=total_tasks, desc=f"{dataset_tag} Processing samples"):
                                if sample_path_results_list:
                                    for result_item in sample_path_results_list:
                                        f_jsonl.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                                        processed_items_count_current_run += 1
                    logger.info(f"{dataset_tag} This run added {processed_items_count_current_run} data items to temporary JSONL: {temp_jsonl_output_path}")

                except Exception as e_proc:
                    logger.error(f"{dataset_tag} Error during path generation for dataset {current_dataset_input}: {e_proc}", exc_info=True)
            
            # Finalization for the current dataset
            # This step now always relies on the content of temp_jsonl_output_path,
            # which might contain only old data (if resuming and no new samples),
            # or a mix of old and new, or only new data.
            if os.path.exists(temp_jsonl_output_path) and os.path.getsize(temp_jsonl_output_path) > 0:
                logger.info(f"{dataset_tag} Finalizing outputs from {temp_jsonl_output_path}...")
                all_generated_path_data_from_jsonl: List[Dict[str, Any]] = []
                try:
                    with open(temp_jsonl_output_path, 'r', encoding='utf-8') as f_jsonl_read:
                        for line in f_jsonl_read:
                            try:
                                all_generated_path_data_from_jsonl.append(json.loads(line))
                            except json.JSONDecodeError as jde:
                                logger.warning(f"{dataset_tag} Malformed JSON line in {temp_jsonl_output_path} during final read: {jde}. Line: '{line.strip()}'")
                except Exception as e:
                    logger.error(f"{dataset_tag} Failed to read temporary JSONL file {temp_jsonl_output_path} for final processing: {e}", exc_info=True)

                if all_generated_path_data_from_jsonl:
                    logger.info(f"{dataset_tag} Loaded {len(all_generated_path_data_from_jsonl)} total items from temp file for final saving.")
                    json_output_path = os.path.join(current_output_dir, 'path_data.json')
                    try:
                        with open(json_output_path, 'w', encoding='utf-8') as f_json:
                            json.dump(all_generated_path_data_from_jsonl, f_json, ensure_ascii=False, indent=2)
                        logger.info(f"{dataset_tag} Final path dataset saved to JSON: {json_output_path}")
                    except (IOError, TypeError) as e:
                        logger.error(f"{dataset_tag} Failed to save final path dataset to JSON {json_output_path}: {e}", exc_info=True)

                    try:
                        processed_hf_dataset = Dataset.from_list(all_generated_path_data_from_jsonl)
                        hf_output_dir = os.path.join(current_output_dir, "hf_dataset")
                        os.makedirs(hf_output_dir, exist_ok=True)
                        processed_hf_dataset.save_to_disk(hf_output_dir)
                        logger.info(f"{dataset_tag} Final path dataset saved to Hugging Face disk format: {hf_output_dir}")
                    except Exception as e:
                        logger.error(f"{dataset_tag} Failed to save dataset to Hugging Face disk format in {hf_output_dir}: {e}", exc_info=True)
                    
                    # Optionally, remove the temp file only if NOT resuming or if resume is "complete and finalize"
                    # For a robust resume, it's often better to keep the temp file until the very end of the entire script,
                    # or manage its lifecycle more explicitly (e.g. rename to .done or delete only on full successful completion of all datasets).
                    # For now, let's keep the original behavior of removing it per dataset.
                    if not args.keep_temp_files: # Add a new argument
                        try:
                            os.remove(temp_jsonl_output_path)
                            logger.info(f"{dataset_tag} Removed temporary file: {temp_jsonl_output_path}")
                        except OSError as e:
                            logger.error(f"{dataset_tag} Error removing temporary file {temp_jsonl_output_path}: {e}", exc_info=True)
                    else:
                        logger.info(f"{dataset_tag} Keeping temporary file as per --keep_temp_files: {temp_jsonl_output_path}")

                else: # all_generated_path_data_from_jsonl is empty
                    logger.warning(f"{dataset_tag} No data loaded from temporary JSONL {temp_jsonl_output_path} after processing. Final files not created.")
            else:
                logger.warning(f"{dataset_tag} Temporary file {temp_jsonl_output_path} not found or was empty. Nothing to save in final step for this dataset.")
            
            logger.info(f"====== {dataset_tag} Finished all processing for dataset: '{current_dataset_input}', split: '{current_split}' ======\n")

    except Exception as e_global:
        logger.error(f"Global error occurred: {e_global}", exc_info=True)
    finally:
        if kg and hasattr(kg, 'close'):
            logger.info("Closing KnowledgeGraph connection (after processing all datasets).")
            kg.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset with paths from KG for multiple datasets, with resume capability.")
    
    parser.add_argument('--dataset_inputs', type=str, nargs='+', required=True,
                        help='List of input dataset paths (local JSON/JSONL/HF_disk_dir, or HF dataset identifier)')
    parser.add_argument('--dataset_config_name', type=str, default=None,
                        help='Name of the dataset configuration for Hugging Face datasets (e.g., "wikitext-2-raw-v1" for wikitext). Only one can be specified globally for now.')
    parser.add_argument('--splits', type=str, nargs='+', required=True,
                        help='List of corresponding dataset splits. Must match --dataset_inputs.')
    
    parser.add_argument('--output_path', type=str, default='data/processed', help='Base output directory')
    
    parser.add_argument('--neo4j_uri', type=str, default=os.environ.get("NEO4J_URI", 'bolt://localhost:7687'))
    parser.add_argument('--neo4j_user', type=str, default=os.environ.get("NEO4J_USER",'neo4j'))
    parser.add_argument('--neo4j_password', type=str, default=os.environ.get("NEO4J_PASSWORD",'password'))
    
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2')
    # force_recompute_embeddings might not be relevant if KG embeddings are handled by KG class internally
    # parser.add_argument('--force_recompute_embeddings', action='store_true') 
    parser.add_argument('--embedding_encode_batch_size', type=int, default=1024)

    parser.add_argument('--max_path_length', type=int, default=3)
    parser.add_argument('--top_k_relations', type=int, default=5)
    
    parser.add_argument('--max_pairs', type=int, default=3)
    parser.add_argument('--max_negatives_per_pair', type=int, default=2)
    
    parser.add_argument('--num_samples', type=int, default=-1, help='Max samples per dataset (-1 for all). Applied *after* filtering for resume.')
    parser.add_argument('--num_threads', type=int, default=0, help='Threads. 0 for auto: min(32, cpus + 4).') # Default 0 for auto

    # New arguments for resume functionality
    parser.add_argument('--resume_processing', action='store_true',
                        help='Enable resuming from existing temp files. Will append to temp files if found.')
    parser.add_argument('--keep_temp_files', action='store_true',
                        help='Do not delete the temporary .jsonl file after processing each dataset.')

    args = parser.parse_args()
    
    # Small validation for num_threads based on new default
    if args.num_threads == 0:
        args.num_threads = min(32, (os.cpu_count() or 1) + 4)
        logger.info(f"Using automatically determined number of threads: {args.num_threads}")
    else:
        logger.info(f"Using specified number of threads: {args.num_threads}")

    prepare_paths(args)