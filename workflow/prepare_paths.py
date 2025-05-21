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

try:
    from src.path_generator import PathGenerator
    from src.knowledge_graph import KnowledgeGraph
except ImportError:
    logger_ph = logging.getLogger(__name__ + "_placeholder") 
    logger_ph.warning("Could not import PathGenerator or KnowledgeGraph from src. Using placeholders.")

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

def process_entity_pair(q_entity: str, a_entity: str, question: str, path_generator: PathGenerator, max_negatives_per_pair: int) -> Dict[str, Any]:
    if not q_entity or not a_entity: # Question check is now primarily in process_sample_task
        return {"shortest_paths": [], "semantic_paths": [], "negative_paths": [], "positive_paths": [] }

    def format_unique_paths_from_list_of_paths(list_of_raw_paths: List[List[Tuple[str, str, str]]]) -> List[str]:
        if not list_of_raw_paths: return []
        # Filter out None paths before converting to tuple for hashing, if any raw_path could be None
        unique_path_structures = {path_to_tuple(p) for p in list_of_raw_paths if p is not None and p}
        return [format_path_for_json(list(p_struct)) for p_struct in unique_path_structures]

    # Get shortest paths
    raw_shortest_paths = path_generator.get_shortest_paths(q_entity, a_entity)
    # Ensure it's a list even if None is returned by placeholder, though `or []` was also fine
    raw_shortest_paths = raw_shortest_paths if raw_shortest_paths is not None else [] 
    formatted_shortest_paths = format_unique_paths_from_list_of_paths(raw_shortest_paths)

    # Get semantic paths
    # Ensure question is non-empty for semantic paths
    raw_semantic_paths = []
    if question: # Only call if question is available
        raw_semantic_path_single, _ = path_generator.get_semantic_path(q_entity, a_entity, question)
        if raw_semantic_path_single: # Check if a path was found
            raw_semantic_paths = [raw_semantic_path_single]
    formatted_semantic_paths = format_unique_paths_from_list_of_paths(raw_semantic_paths)
    
    # Combine for positive paths
    # Ensure all elements for set union are actual path lists, not None
    all_positive_path_structures = set()
    if raw_shortest_paths:
         all_positive_path_structures.update({path_to_tuple(p) for p in raw_shortest_paths if p})
    if raw_semantic_paths: # raw_semantic_paths is already [path] or []
         all_positive_path_structures.update({path_to_tuple(p) for p in raw_semantic_paths if p})
    formatted_positive_paths = [format_path_for_json(list(p_struct)) for p_struct in all_positive_path_structures if p_struct]

    # Get negative paths
    list_of_raw_negative_paths = []
    if question and all_positive_path_structures: # Only generate if question and positive paths exist
        for p_struct_tuple in all_positive_path_structures:
            if not p_struct_tuple: continue 
            # Pass the original list of tuples for the path structure
            negs_for_p = path_generator.get_negative_paths(list(p_struct_tuple), question, a_entity, max_negatives_per_pair)
            if negs_for_p: list_of_raw_negative_paths.extend(negs_for_p)
    formatted_negative_paths = format_unique_paths_from_list_of_paths(list_of_raw_negative_paths)

    return {
        "shortest_paths": formatted_shortest_paths, 
        "semantic_paths": formatted_semantic_paths,
        "negative_paths": formatted_negative_paths, 
        "positive_paths": formatted_positive_paths
    }

def _parse_entities(entities_raw: Any) -> List[str]:
    if isinstance(entities_raw, str) and entities_raw.strip():
        return [entities_raw.strip()]
    if isinstance(entities_raw, list):
        return [str(item).strip() for item in entities_raw if item and str(item).strip()]
    return []

def process_sample_task(sample_tuple: Tuple[Dict[str, Any], ProcessingConfig, PathGenerator]) -> List[Dict[str, Any]]:
    sample, config, path_generator = sample_tuple
    try:
        q_entities = _parse_entities(sample.get('q_entity'))
        a_entities = _parse_entities(sample.get('a_entity', []))
        question = sample.get('question', "").strip()
        sample_id = sample.get('id', 'unknown_sample_id')

        all_path_results_for_sample: List[Dict[str, Any]] = []

        if not q_entities: # Question check is now less strict here, as some paths might not need it.
            logger.debug(f"Sample {sample_id} has no valid query entities. Skipping path generation for this sample.")
            return all_path_results_for_sample 
        
        if not question:
            logger.debug(f"Sample {sample_id} has no question. Semantic and negative paths may be skipped for pairs in this sample.")
            # Path generation will proceed, but process_entity_pair will handle empty question for specific path types.

        for q_entity_item in q_entities:
            pairs_processed_for_q_entity = 0
            if not a_entities:
                logger.debug(f"Sample {sample_id}, q_entity '{q_entity_item}' has no valid answer entities. Skipping pairs for this q_entity.")
                continue

            for a_entity_item in a_entities:
                if pairs_processed_for_q_entity >= config.max_pairs:
                    logger.debug(f"Reached max_pairs ({config.max_pairs}) for q_entity '{q_entity_item}'.")
                    break

                pair_path_data = process_entity_pair(
                    q_entity_item, a_entity_item, question, # Pass potentially empty question
                    path_generator, config.max_negatives_per_pair
                )
                
                # MODIFIED Conditional saving: 
                # Save IF (positive_paths is NOT empty) OR (negative_paths is NOT empty)
                if pair_path_data["positive_paths"] or pair_path_data["negative_paths"]:
                    result_item = {
                        "id": sample_id, "question": question,
                        "q_entity": q_entity_item, "a_entity": a_entity_item,
                        **pair_path_data 
                    }
                    all_path_results_for_sample.append(result_item)
                else:
                    logger.debug(f"Skipping save for pair ({q_entity_item}, {a_entity_item}) in sample {sample_id} as both positive_paths and negative_paths are empty.")

                pairs_processed_for_q_entity += 1
        return all_path_results_for_sample
    except Exception as e:
        sample_id_info = sample.get('id', 'unknown_id') if isinstance(sample, dict) else 'unknown_sample_structure'
        logger.error(f"Error processing sample (ID: {sample_id_info}) in thread {os.getpid()}: {e}", exc_info=True)
        return []

def prepare_paths(args: argparse.Namespace):
    logger.info(f"Loading dataset from: {args.data_path}")
    dataset_iterable: Any 
    try:
        if os.path.exists(args.data_path):
            if args.data_path.endswith(".jsonl"):
                dataset_list_concrete = []
                with open(args.data_path, 'r', encoding='utf-8') as f:
                    for line in f: dataset_list_concrete.append(json.loads(line))
                dataset_iterable = dataset_list_concrete
            elif args.data_path.endswith(".json"):
                with open(args.data_path, 'r', encoding='utf-8') as f:
                    dataset_list_concrete = json.load(f)
                if not isinstance(dataset_list_concrete, list):
                    logger.error(f"JSON file {args.data_path} does not contain a list of items.")
                    return
                dataset_iterable = dataset_list_concrete
            else: 
                from datasets import load_from_disk # type: ignore
                dataset_iterable = load_from_disk(args.data_path)
        else: 
            dataset_iterable = load_dataset(args.data_path, split=args.split)

        if args.num_samples > 0 :
            if isinstance(dataset_iterable, list):
                 dataset_iterable = dataset_iterable[:args.num_samples]
            else: 
                original_len = -1
                try: original_len = len(dataset_iterable)
                except TypeError: logger.info("Dataset iterator has no len().")

                try:
                    if original_len != -1:
                        dataset_iterable = dataset_iterable.select(range(min(args.num_samples, original_len)))
                    else: # Cannot use select if len is unknown, iterate manually
                        raise AttributeError("Cannot use select without known length or if not HF dataset.")
                except (AttributeError, TypeError): 
                     logger.warning("Could not use .select() on dataset_iterable or len() failed, attempting manual slicing for num_samples.")
                     temp_list = []
                     for i, item_sample in enumerate(dataset_iterable):
                         if i >= args.num_samples:
                             break
                         temp_list.append(item_sample)
                     dataset_iterable = temp_list
    except Exception as e:
        logger.error(f"Failed to load dataset from '{args.data_path}' (split: {args.split}): {e}", exc_info=True)
        return

    kg = None
    output_dir = os.path.join(args.output_path, args.output_name)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}", exc_info=True)
        return
    
    temp_jsonl_output_path = os.path.join(output_dir, 'path_data_temp.jsonl')

    try:
        logger.info("Initializing knowledge graph...")
        kg = KnowledgeGraph(args.neo4j_uri, args.neo4j_user, args.neo4j_password)
        
        kg.initialize_embeddings(
            model_name=args.model_name,
            embedding_encode_batch_size=args.embedding_encode_batch_size
        )

        logger.info("Initializing path generator...")
        config = ProcessingConfig(
            max_pairs=args.max_pairs,
            max_negatives_per_pair=args.max_negatives_per_pair,
            max_path_length=args.max_path_length,
            top_k_relations=args.top_k_relations
        )
        path_generator = PathGenerator(
            kg=kg,
            max_path_length=config.max_path_length,
            top_k_relations=config.top_k_relations
        )

        logger.info("Preparing tasks for path generation...")
        tasks_args = [(sample, config, path_generator) for sample in tqdm(dataset_iterable, desc="Preparing tasks")]
        total_tasks = len(tasks_args)
        logger.info(f"Prepared to process {total_tasks} samples using ThreadPoolExecutor.")
        
        num_threads = args.num_threads if args.num_threads > 0 else min(32, (os.cpu_count() or 1) + 4)
        logger.info(f"Using {num_threads} worker threads.")

        processed_items_count = 0 
        with open(temp_jsonl_output_path, 'w', encoding='utf-8') as f_jsonl:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results_iterator = executor.map(process_sample_task, tasks_args)
                
                for sample_path_results_list in tqdm(results_iterator, total=total_tasks, desc="Processing samples"):
                    if sample_path_results_list: 
                        for result_item in sample_path_results_list: 
                            f_jsonl.write(json.dumps(result_item, ensure_ascii=False) + "\n")
                            processed_items_count +=1
        
        logger.info(f"Finished processing. {processed_items_count} filtered data items written to temporary JSONL: {temp_jsonl_output_path}")

    except Exception as e_outer:
        logger.error(f"Outer error during processing: {e_outer}", exc_info=True)
    finally:
        if kg and hasattr(kg, 'close'):
            logger.info("Closing KnowledgeGraph connection.")
            kg.close()

    if os.path.exists(temp_jsonl_output_path):
        logger.info(f"Finalizing outputs from {temp_jsonl_output_path}...")
        all_generated_path_data_from_jsonl: List[Dict[str, Any]] = []
        try:
            with open(temp_jsonl_output_path, 'r', encoding='utf-8') as f_jsonl_read:
                for line in f_jsonl_read:
                    all_generated_path_data_from_jsonl.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read temporary JSONL file {temp_jsonl_output_path} for final processing: {e}", exc_info=True)

        if all_generated_path_data_from_jsonl:
            json_output_path = os.path.join(output_dir, 'path_data.json')
            try:
                with open(json_output_path, 'w', encoding='utf-8') as f_json:
                    json.dump(all_generated_path_data_from_jsonl, f_json, ensure_ascii=False, indent=2)
                logger.info(f"Final path dataset saved to JSON: {json_output_path}")
            except (IOError, TypeError) as e:
                logger.error(f"Failed to save final path dataset to JSON {json_output_path}: {e}", exc_info=True)

            try:
                processed_hf_dataset = Dataset.from_list(all_generated_path_data_from_jsonl)
                hf_output_dir = os.path.join(output_dir, "hf_dataset") 
                os.makedirs(hf_output_dir, exist_ok=True)
                processed_hf_dataset.save_to_disk(hf_output_dir)
                logger.info(f"Final path dataset saved to Hugging Face disk format: {hf_output_dir}")
            except Exception as e:
                logger.error(f"Failed to save dataset to Hugging Face disk format in {hf_output_dir}: {e}", exc_info=True)
        else:
            logger.warning("No data loaded from temporary JSONL. Final files not created.")
        
        try:
            os.remove(temp_jsonl_output_path)
            logger.info(f"Removed temporary file: {temp_jsonl_output_path}")
        except OSError as e:
            logger.error(f"Error removing temporary file {temp_jsonl_output_path}: {e}", exc_info=True)
    else:
        if processed_items_count == 0:
             logger.warning("No data items met the saving criteria. No output files generated.")
        else: # processed_items_count > 0 but file doesn't exist (should not happen if no error before)
             logger.warning(f"Temporary file {temp_jsonl_output_path} not found despite items processed, or was empty. Nothing to save in final step.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset with various path types from a Knowledge Graph")
    parser.add_argument('--data_path', type=str, required=True, help='Input dataset path (local JSON/JSONL file, HF dataset dir, or HF dataset identifier)')
    parser.add_argument('--dataset_name', type=str, default='my_dataset_cache', help='Identifier for this dataset, used in caching paths for embeddings (e.g., webqsp, cwq)')
    parser.add_argument('--split', type=str, default='train', help='Dataset split (e.g., train, validation, test), used for caching and if loading from Hugging Face Hub')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Base output directory for all processed data')
    parser.add_argument('--output_name', type=str, default='path_enhanced_data', help='Specific directory name for this run_s output (under output_path)')
    
    parser.add_argument('--neo4j_uri', type=str, default=os.environ.get("NEO4J_URI", 'bolt://localhost:7687'), help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default=os.environ.get("NEO4J_USER",'neo4j'), help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default=os.environ.get("NEO4J_PASSWORD",'password'), help='Neo4j password')
    
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Pretrained SentenceTransformer model name for embeddings used in semantic pathfinding')
    parser.add_argument('--force_recompute_embeddings', action='store_true', help='Force recomputation of KG embeddings even if cache exists.')
    parser.add_argument('--embedding_encode_batch_size', type=int, default=1024, help='Batch size for SentenceTransformer encoding during KG embedding initialization.')

    parser.add_argument('--max_path_length', type=int, default=3, help='Maximum path length for PathGenerator')
    parser.add_argument('--top_k_relations', type=int, default=5, help='Default Top K relations for PathGenerator strategies')
    
    parser.add_argument('--max_pairs', type=int, default=3, help='Maximum (q_entity, a_entity) pairs to process per input sample/question')
    parser.add_argument('--max_negatives_per_pair', type=int, default=2, help='Maximum negative paths PathGenerator tries to generate per positive path found for a (q_entity, a_entity) pair')
    
    parser.add_argument('--num_samples', type=int, default=-1, help='Maximum samples to process from input dataset (-1 for all)')
    parser.add_argument('--num_threads', type=int, default=0, help='Number of threads for ThreadPoolExecutor. Default (0) uses min(32, os.cpu_count() + 4).')

    args = parser.parse_args()
    prepare_paths(args)
