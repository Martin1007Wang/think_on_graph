import os
import argparse
import json
import logging
import datetime
import contextlib
import time
import threading # For thread-safe file writing
from typing import Set, List, Dict, Any, Tuple, Optional, Iterator, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datasets import load_dataset
from src.llms import get_registed_model
from src.utils.qa_utils import eval_path_result_w_ans
from src.knowledge_graph import KnowledgeGraph
from src.knowledge_explorer import KnowledgeExplorer
from src.llms.base_language_model import BaseLanguageModel
import torch
import sys

# --- Configuration ---
DEFAULT_NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
DEFAULT_NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
DEFAULT_NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "YOUR_PASSWORD_HERE") # *** Avoid hardcoding passwords ***

# --- File Handling ---
class FileHandler:
    @staticmethod
    def get_processed_ids(path: str) -> Set[str]:
        """Reads processed IDs from a JSON Lines file."""
        processed_ids = set()
        if not os.path.exists(path):
            return processed_ids
        try:
            with open(path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "id" in item:
                            processed_ids.add(item["id"])
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON line in {path}: {line.strip()}")
        except IOError as e:
            logging.error(f"Error reading processed file {path}: {e}")
        return processed_ids

    @staticmethod
    def save_args(output_dir: str, args: argparse.Namespace) -> None:
        """Saves arguments to a file."""
        os.makedirs(output_dir, exist_ok=True)
        args_path = os.path.join(output_dir, 'args.txt')
        try:
            with open(args_path, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        except IOError as e:
            logging.error(f"Error saving arguments to {args_path}: {e}")

# --- Data Loading & Filtering ---
def load_target_ids(target_ids_arg: str) -> Set[str]:
    """Loads target IDs from an argument string or a file."""
    target_id_set = set()
    if not target_ids_arg:
        return target_id_set
    if os.path.isfile(target_ids_arg):
        try:
            with open(target_ids_arg) as f:
                target_id_set = set(line.strip() for line in f if line.strip())
        except IOError as e:
            logging.error(f"Error reading target IDs file {target_ids_arg}: {e}")
    else:
        target_id_set = set(i.strip() for i in target_ids_arg.split(',') if i.strip())
    logging.info(f"Loaded {len(target_id_set)} target IDs.")
    return target_id_set

def filter_dataset(dataset: Any, target_id_set: Set[str], processed_set: Set[str]) -> List[Dict[str, Any]]:
    """Filters dataset based on target IDs and excludes processed IDs."""
    if target_id_set:
        dataset_iter = [d for d in dataset if d.get("id") in target_id_set]
        logging.info(f"Filtered dataset to {len(dataset_iter)} questions matching target IDs.")
    else:
        dataset_iter = list(dataset) # Convert to list if not already
        logging.info(f"Processing all {len(dataset_iter)} questions in dataset (no target IDs specified).")

    to_process = [d for d in dataset_iter if d.get("id") not in processed_set]

    if not dataset_iter:
        logging.warning("The dataset (after filtering by target_ids) is empty.")
    elif not to_process:
        logging.info("All target questions have already been processed. Nothing new to do.")
    else:
        logging.info(f"Found {len(to_process)} questions to process ({len(processed_set)} already done).")

    return to_process

# --- Core Processing ---
def process_question_wrapper(explorer: KnowledgeExplorer, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Wraps the explorer call with timing and error handling."""
    start_time = time.time()
    q_id = data.get("id", "UNKNOWN_ID") # Handle cases where 'id' might be missing
    try:
        res = explorer.process_question(data)
        if res is not None:
            processing_time = time.time() - start_time
            logging.debug(f"Processed question {q_id} in {processing_time:.2f}s")
            if 'id' not in res and 'id' in data:
                 res['id'] = data['id']
            return res
        else:
            logging.warning(f"Processing question {q_id} returned None.")
            return None
    except Exception as e:
        processing_time = time.time() - start_time
        logging.error(f"Error processing id {q_id} after {processing_time:.2f}s: {e}", exc_info=True)
        return {"id": q_id, "error": str(e)}


# --- Utilities ---
def setup_logging(debug: bool = False) -> None:
    """Configures logging."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s' # Added threadName
    logging.basicConfig(level=log_level, format=log_format)

def evaluate_results(predictions_path: str) -> None:
    """Evaluates results if the predictions file exists and is not empty."""
    if not os.path.exists(predictions_path):
        logging.warning(f"Predictions file not found: {predictions_path}. Cannot evaluate.")
        return
    if os.path.getsize(predictions_path) == 0:
        logging.warning(f"Predictions file is empty: {predictions_path}. Cannot evaluate.")
        return

    logging.info(f"Evaluating results from {predictions_path}...")
    try:
        eval_path_result_w_ans(predictions_path)
        logging.info("Evaluation completed.")
    except FileNotFoundError:
         logging.error(f"Evaluation function failed: Predictions file not found at {predictions_path}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}", exc_info=True)

def initialize_models(args: argparse.Namespace, explore_model_class: type, predict_model_class: type) -> Tuple[BaseLanguageModel, BaseLanguageModel]:
    """Initializes exploration and prediction models."""
    logging.info("Initializing models...")

    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        logging.info(f"Found {num_gpus} GPU(s). Using device_map='{args.device_map}'.")
    else:
        logging.warning("No GPU available. Using CPU.")
        if args.device_map not in ["cpu", "auto"]:
             logging.warning(f"Device map set to '{args.device_map}' but no GPU found. Forcing to 'cpu'.")
             args.device_map = "cpu"

    try:
        explore_model = explore_model_class(args)
        explore_model.prepare_for_inference(args.explore_model_path) # Pass path if needed by implementation

        if (args.explore_model_name == args.predict_model_name and
            args.explore_model_path == args.predict_model_path):
            logging.info("Using the same model instance for exploration and prediction.")
            predict_model = explore_model
        else:
            logging.info(f"Initializing separate prediction model: {args.predict_model_name}")
            predict_model = predict_model_class(args)
            predict_path = args.predict_model_path if args.predict_model_path else args.explore_model_path
            if not predict_path:
                 logging.warning(f"No specific path for predict_model '{args.predict_model_name}', and explore_model_path is also None. Model might use default weights.")
            predict_model.prepare_for_inference(predict_path)

        logging.info("Model initialization successful.")
        return explore_model, predict_model

    except Exception as e:
        logging.error(f"Fatal error during model initialization: {e}", exc_info=True)
        if "CUDA" in str(e):
            logging.error("A CUDA error occurred during model initialization. "
                          "Check GPU memory, driver compatibility, and model requirements.")
            logging.error("Consider trying device_map='auto' or specifying a single GPU like 'cuda:0'.")
        raise

# --- Main Orchestration ---
def main(args: argparse.Namespace) -> None:
    """Main function to run the KGQA process."""
    setup_logging(args.debug)
    start_time_main = datetime.datetime.now()
    logging.info(f"Starting KGQA process at {start_time_main.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Run arguments: {vars(args)}") # Log arguments

    # --- Initialize Models ---
    try:
        explore_model_class = get_registed_model(args.explore_model_name)
        predict_model_class = get_registed_model(args.predict_model_name)
        explore_model, predict_model = initialize_models(args, explore_model_class, predict_model_class)
    except Exception as e:
        logging.error(f"Fatal error during model initialization: {e}", exc_info=True)
        return

    # --- Initialize KG ---
    try:
        kg = KnowledgeGraph(uri=args.neo4j_uri, user=args.neo4j_user,
                            password=args.neo4j_password, model_name=args.embedding_model)
        logging.info("Knowledge Graph connection successful.")
    except Exception as e:
        logging.error(f"Fatal error connecting to Knowledge Graph at {args.neo4j_uri}: {e}", exc_info=True)
        return

    # --- Initialize Explorer ---
    explorer = KnowledgeExplorer(
        kg=kg,
        explore_model=explore_model,
        predict_model=predict_model,
        max_rounds=args.max_rounds,
        max_selection_count=args.max_selection_count
    )

    # --- Setup Output ---
    output_dir = os.path.join(args.predict_path, args.data_name, args.explore_model_name,
                              args.predict_model_name,
                              f"{args.prefix}iterative-rounds{args.max_rounds}-topk{args.max_selection_count}")
    logging.info(f"Results will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    FileHandler.save_args(output_dir, args) # Save effective args
    predictions_path = os.path.join(output_dir, 'predictions.jsonl')

    # --- Load Data & Determine Work ---
    try:
        dataset = load_dataset(os.path.join(args.data_path, args.data_name), split=args.split)
        logging.info(f"Successfully loaded dataset '{args.data_name}' split '{args.split}' with {len(dataset)} items.")
    except Exception as e:
        logging.error(f"Fatal error loading dataset: {e}", exc_info=True)
        return

    processed_set = FileHandler.get_processed_ids(predictions_path)
    logging.info(f"Found {len(processed_set)} already processed question IDs.")
    if not args.force and len(processed_set) > 0:
         logging.info("Resuming previous run. Use --force to overwrite.")
    elif args.force:
         logging.warning("Argument --force specified. Overwriting existing predictions file.")
         processed_set.clear()

    target_id_set = load_target_ids(args.target_ids)
    to_process = filter_dataset(dataset, target_id_set, processed_set)

    if not to_process:
        logging.info("No questions left to process.")
        evaluate_results(predictions_path)
        return

    # --- Sequential Processing ---
    processed_count = 0
    failed_count = 0
    failed_ids = []

    logging.info(f"Starting processing of {len(to_process)} questions...")

    try:
        file_mode = "w" if args.force else "a"
        with open(predictions_path, file_mode) as fout:
            progress_bar = tqdm(total=len(to_process), desc="Processing questions sequentially")
            for data in to_process:
                q_id = data.get("id", "UNKNOWN_ID")
                result = process_question_wrapper(explorer, data)
                if result is not None:
                    if isinstance(result, dict) and "error" in result and "id" in result:
                        failed_count += 1
                        failed_ids.append(result["id"])
                        logging.warning(f"Main: Recorded failure for question ID {result['id']} (See wrapper log for details)")
                    elif isinstance(result, dict):
                        try:
                            fout.write(json.dumps(result) + "\n")
                            fout.flush()
                            processed_count += 1
                        except TypeError as json_err:
                            failed_count += 1
                            failed_ids.append(q_id)
                            logging.error(f"Main: Failed to serialize result for Q:{q_id} to JSON: {json_err}. Result was: {result}")
                        except Exception as write_err:
                            failed_count += 1
                            failed_ids.append(q_id)
                            logging.error(f"Main: Failed to write result for Q:{q_id}: {write_err}")
                    else:
                        failed_count += 1
                        failed_ids.append(q_id)
                        logging.error(f"Main: Wrapper for Q:{q_id} returned unexpected type: {type(result)}. Value: {result}")
                else:
                    failed_count += 1
                    failed_ids.append(q_id)
                    logging.warning(f"Main: Processing returned None object for question ID {q_id}")
                progress_bar.update(1)
            progress_bar.close()

    except IOError as e:
        logging.error(f"Fatal error writing to output file {predictions_path}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred during sequential processing: {e}", exc_info=True)

    # --- Final Reporting ---
    end_time_main = datetime.datetime.now()
    duration_minutes = (end_time_main - start_time_main).total_seconds() / 60.0

    logging.info("-" * 50)
    logging.info("Processing Summary:")
    logging.info(f"  Total time: {duration_minutes:.2f} minutes")
    if processed_count > 0:
        logging.info(f"  Successfully processed: {processed_count} questions")
        logging.info(f"  Average time per successful question: {(duration_minutes * 60) / processed_count:.2f} seconds")
    else:
        logging.info("  No questions were successfully processed in this run.")

    if failed_count > 0:
        logging.warning(f"  Failed to process: {failed_count} questions")
        logging.warning(f"  Failed IDs: {failed_ids if len(failed_ids) < 20 else str(failed_ids[:20]) + '...'}") # Log some failed IDs

    logging.info(f"Results saved to: {predictions_path}")
    logging.info("-" * 50)

    # --- Evaluate ---
    if processed_count > 0 or os.path.exists(predictions_path): # Evaluate if we processed something or if file exists from previous runs
         evaluate_results(predictions_path)
    else:
         logging.info("Skipping evaluation as no results were processed and no prior results file exists.")

    logging.info("KGQA process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering via Iterative Reasoning")

    # Data & Paths
    parser.add_argument('--data_path', type=str, default='rmanluo', help="Path to the dataset directory (Hugging Face datasets format)")
    parser.add_argument('--data_name', type=str, default='RoG-webqsp', help="Name of the dataset")
    parser.add_argument('--split', type=str, default='test', help="Dataset split (e.g., 'test', 'train', 'test[:100]')") # Changed default
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning_Optimized', help="Base directory for saving results")
    parser.add_argument('--target_ids', type=str, default="", help='Comma-separated IDs or path to a file with IDs (one per line) to process only specific questions.')

    # Execution Control
    parser.add_argument('--force', action='store_true', help='Force overwrite existing predictions file instead of appending.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--prefix', type=str, default="", help="Prefix for the output subdirectory name")
    parser.add_argument('--max_workers', type=int, default=None, help="Max worker threads for parallel processing (default: auto-adjust)")

    # Model Configuration
    parser.add_argument('--explore_model_name', type=str, default='gcr-Llama-2-7b-chat-hf', help="Name of the registered exploration model")
    parser.add_argument('--explore_model_path', type=str, default=None, help="Path to the exploration model weights/directory (optional)")
    parser.add_argument('--predict_model_name', type=str, default='gcr-Llama-2-7b-chat-hf', help="Name of the registered prediction model")
    parser.add_argument('--predict_model_path', type=str, default=None, help="Path to the prediction model weights/directory (optional, defaults to explore_model_path if same model)")
    parser.add_argument('--device_map', type=str, default="auto", help="Device mapping strategy for models (e.g., 'auto', 'cuda:0', 'cpu', 'balanced') passed to Hugging Face")
    # parser.add_argument('--use_single_gpu', type=lambda x: (str(x).lower() == 'true'), default=True, help='DEPRECATED: Use --device_map instead.') # Removed confusing arg

    # KG & Explorer Configuration
    parser.add_argument('--max_rounds', type=int, default=2, help="Maximum rounds of iterative reasoning in KnowledgeExplorer")
    parser.add_argument('--max_selection_count', type=int, default=5, help="Maximum number of relations to explore per step in KnowledgeExplorer")
    parser.add_argument('--neo4j_uri', type=str, default=DEFAULT_NEO4J_URI, help="Neo4j connection URI")
    parser.add_argument('--neo4j_user', type=str, default=DEFAULT_NEO4J_USER, help="Neo4j username")
    parser.add_argument('--neo4j_password', type=str, default=DEFAULT_NEO4J_PASSWORD, help="Neo4j password (better to set via NEO4J_PASSWORD env var)")
    parser.add_argument('--embedding_model', type=str, default='msmarco-distilbert-base-tas-b', help="Sentence Transformer model for embeddings in KG")
    # parser.add_argument('--undirected', type=lambda x: (str(x).lower() == 'true'), default=False) # This argument seems unused in the provided main logic

    # Parse base args first to get model names
    base_args, remaining_argv = parser.parse_known_args()

    # Dynamically add model-specific arguments
    try:
        explore_model_class = get_registed_model(base_args.explore_model_name)
        explore_model_class.add_args(parser)
        logging.debug(f"Added args for explore model: {base_args.explore_model_name}")

        if base_args.explore_model_name != base_args.predict_model_name:
            predict_model_class = get_registed_model(base_args.predict_model_name)
            predict_model_class.add_args(parser)
            logging.debug(f"Added args for distinct predict model: {base_args.predict_model_name}")
        else:
             logging.debug("Explore and predict models are the same type, not adding predict args separately.")

    except Exception as e:
         logging.error(f"Failed to get or add arguments for models: {e}", exc_info=True)
         # Exit if model classes cannot be loaded, as configuration is incomplete
         exit(1)

    # Parse all arguments including the dynamically added ones
    args = parser.parse_args(remaining_argv + sys.argv[1:]) # Combine remaining args with original sys.argv


    # --- Security Warning for Password ---
    if args.neo4j_password == "YOUR_PASSWORD_HERE" or args.neo4j_password == "Martin1007Wang": # Check against placeholder and old default
         logging.warning("Neo4j password is set to a default or placeholder value. "
                       "It is strongly recommended to set the password via the NEO4J_PASSWORD environment variable "
                       "or use a secure configuration method instead of command-line arguments.")


    # Call main function without passing models (they are initialized inside main now)
    main(args)