import argparse
import datetime
import json
import logging
import os
import time
from typing import Set, List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

'''from src.llms import get_registed_model
from src.llms.base_language_model import BaseLanguageModel
from src.evaluator import run_evaluation
from src.knowledge_graph import KnowledgeGraph
from src.knowledge_explorer import KnowledgeExplorer
from src.utils.utils import FileHandler, load_target_ids, filter_dataset, setup_logging
from src.llms.base_hf_causal_model import HfCausalModel # Try to import your actual class'''

def initialize_models(args: argparse.Namespace, explore_model_class: type, predict_model_class: type) -> Tuple[BaseLanguageModel, BaseLanguageModel]:
    logging.info("Initializing model objects...")
    try:
        logging.info(f"Creating exploration model object for: {args.explore_model_name}")
        explore_model = explore_model_class.from_args(args)
        if (getattr(args, 'explore_model_path', None) == getattr(args, 'predict_model_path', None)):
            logging.info("Models point to the same path. A single instance will be used.")
            predict_model = explore_model
        else:
            logging.info(f"Creating separate prediction model object for: {args.predict_model_name}")
            predict_model = predict_model_class.from_args(args)
        logging.info("Model object initialization successful.")
        return explore_model, predict_model
    except Exception as e:
        logging.error(f"Fatal error during model object instantiation: {e}", exc_info=True)
        raise

def run_evaluation_and_stats(predictions_path: str) -> None:
    if not os.path.exists(predictions_path) or os.path.getsize(predictions_path) == 0:
        return
    runtimes, llm_calls, llm_tokens = [], [], []
    with open(predictions_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                runtimes.append(data.get("runtime_s", 0))
                llm_calls.append(data.get("llm_calls", 0))
                llm_tokens.append(data.get("llm_tokens", 0))
            except json.JSONDecodeError:
                logging.warning(f"无法解析行: {line.strip()}")
    logging.info("----- Overall Statistics -----")
    stats = {
        "Runtime (s)": runtimes,
        "LLM Calls": llm_calls,
        "LLM Tokens": llm_tokens,
    }
    header = f"{'Metric':<15} | {'Avg.':>12} | {'Min.':>12} | {'Max.':>12}"
    logging.info(header)
    logging.info("-" * len(header))
    for name, data in stats.items():
        avg = np.mean(data)
        min_val = np.min(data)
        max_val = np.max(data)
        logging.info(f"{name:<15} | {avg:>12.2f} | {min_val:>12.0f} | {max_val:>12.0f}")
    logging.info("------------------------------")
    logging.info("Running standard evaluation...")
    run_evaluation(predictions_path)


def main(args: argparse.Namespace) -> None:
    setup_logging(args.debug)
    start_time_main = datetime.datetime.now()
    logging.info(f"Starting KGQA process at {start_time_main.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Run arguments: {vars(args)}")

    explore_model, predict_model, kg = None, None, None
    try:
        # --- 1. Lightweight Initialization ---
        explore_model_class = get_registed_model(args.explore_model_name)
        predict_model_class = get_registed_model(args.predict_model_name)
        explore_model, predict_model = initialize_models(args, explore_model_class, predict_model_class)
        kg = KnowledgeGraph(uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password)
        explorer = KnowledgeExplorer(
            kg=kg,
            explore_model=explore_model,
            predict_model=predict_model,
            max_rounds=args.max_rounds,
            max_selection_count=args.max_selection_count
        )
        
        output_dir = os.path.join(args.predict_path, args.data_name, args.explore_model_name,
                                  args.predict_model_name,
                                  f"{args.prefix}iterative-rounds{args.max_rounds}-topk{args.max_selection_count}")
        os.makedirs(output_dir, exist_ok=True)
        predictions_path = os.path.join(output_dir, 'predictions.jsonl')
        FileHandler.save_args(output_dir, args)
        
        dataset = load_dataset(os.path.join(args.data_path, args.data_name), split=args.split)
        processed_set = FileHandler.get_processed_ids(predictions_path)
        target_id_set = load_target_ids(getattr(args, 'target_ids', None))
        to_process = filter_dataset(dataset, target_id_set, processed_set)

        if not to_process:
            logging.info("No questions left to process.")
            run_evaluation(predictions_path)
            return
            
        # --- 3. ONE-TIME MODEL LOADING ---
        logging.info("Preparing ALL models for inference (Load once, reside throughout)...")
        explore_model.prepare_for_inference()
        
        if id(explore_model) != id(predict_model):
            predict_model.prepare_for_inference()
        
        logging.info("All required models are loaded and ready.")
        
        # --- 4. Core Processing Loop ---
        logging.info(f"Starting processing of {len(to_process)} questions...")
        
        file_mode = "w" if args.force and not processed_set else "a"
        with open(predictions_path, file_mode, encoding='utf-8') as fout:
            for data in tqdm(to_process, desc="Processing questions"):
                result = explorer.process_question(data)
                if result:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    except Exception as e:
        logging.error(f"A fatal error occurred in the main process: {e}", exc_info=True)
    
    finally:
        # --- 5. Final Resource Cleanup ---
        logging.info("Final resource cleanup...")
        if explore_model and explore_model.is_ready:
            explore_model.unload_resources()
        if predict_model and predict_model is not explore_model and predict_model.is_ready:
             predict_model.unload_resources()
        if kg and hasattr(kg, 'close'):
            kg.close()

        # --- 6. Final Summary and Evaluation ---
        logging.info("----- Processing Complete -----")
        run_evaluation(predictions_path)
        logging.info("KGQA process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering via Iterative Reasoning")

    # Data & Paths
    parser.add_argument('--data_path', type=str, default='rmanluo', help="Path to the dataset directory")
    parser.add_argument('--data_name', type=str, default='RoG-webqsp', help="Name of the dataset")
    parser.add_argument('--split', type=str, default='test', help="Dataset split")
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning_Optimized', help="Base directory for saving results")
    parser.add_argument('--target_ids', type=str, default="", help='Comma-separated IDs or path to a file with IDs')

    # Execution Control
    parser.add_argument('--force', action='store_true', help='Force overwrite existing predictions file.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--prefix', type=str, default="", help="Prefix for the output subdirectory name")
    
    # Model Configuration
    parser.add_argument('--explore_model_name', type=str, default='hf-causal-model', help="Name of the registered exploration model")
    parser.add_argument('--explore_model_path', type=str, default=None, help="Path to the exploration model")
    parser.add_argument('--predict_model_name', type=str, default='hf-causal-model', help="Name of the registered prediction model")
    parser.add_argument('--predict_model_path', type=str, default=None, help="Path to the prediction model")

    # KG & Explorer Configuration
    parser.add_argument('--max_rounds', type=int, default=2, help="Maximum rounds of iterative reasoning")
    parser.add_argument('--max_selection_count', type=int, default=5, help="Maximum number of relations to explore per step")
    parser.add_argument('--neo4j_uri', type=str, default=os.getenv("NEO4J_URI", "bolt://localhost:7687"), help="Neo4j URI")
    parser.add_argument('--neo4j_user', type=str, default=os.getenv("NEO4J_USER", "neo4j"), help="Neo4j username")
    parser.add_argument('--neo4j_password', type=str, default=os.getenv("NEO4J_PASSWORD", "password"), help="Neo4j password")
    
    try:
        temp_args, _ = parser.parse_known_args()
        explore_model_class = get_registed_model(temp_args.explore_model_name)
        if hasattr(explore_model_class, 'add_args'):
            explore_model_class.add_args(parser)
        if temp_args.explore_model_name != temp_args.predict_model_name:
            predict_model_class = get_registed_model(temp_args.predict_model_name)
            if hasattr(predict_model_class, 'add_args'):
                predict_model_class.add_args(parser)
    except Exception as e:
        logging.error(f"Failed to add model-specific arguments: {e}", exc_info=True)
        exit(1)

    final_args = parser.parse_args()
    main(final_args)
