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
import pickle
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
    if not target_ids_arg: # Handles None or empty string
        return target_id_set
    if os.path.isfile(target_ids_arg):
        try:
            with open(target_ids_arg) as f:
                target_id_set = set(line.strip() for line in f if line.strip())
        except IOError as e:
            logging.error(f"Error reading target IDs file {target_ids_arg}: {e}")
    else:
        # Assuming target_ids_arg is a comma-separated string if not a file
        target_id_set = set(i.strip() for i in target_ids_arg.split(',') if i.strip())
    logging.info(f"Loaded {len(target_id_set)} target IDs from input: '{target_ids_arg}'.")
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

    if not dataset_iter and target_id_set: # Modified condition to be more specific
        logging.warning("The dataset (after filtering by target_ids) is empty, or no provided target IDs were found in the dataset.")
    elif not to_process:
        logging.info("All target questions have already been processed or no questions to process after filtering.")
    else:
        logging.info(f"Found {len(to_process)} questions to process ({len(processed_set)} already done from the filtered set).")

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
            if 'id' not in res and 'id' in data: # Ensure 'id' is in the result
                res['id'] = data['id']
            return res
        else:
            logging.warning(f"Processing question {q_id} returned None.")
            # Return a dict with id and error for consistent failure tracking
            return {"id": q_id, "error": "Processing returned None"}
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
        logging.info(f"Found {num_gpus} GPU(s). Using device_map='auto'.")
    else:
        logging.warning("No GPU available. Using CPU.")

    try:
        explore_model = explore_model_class(args)
        explore_model.prepare_for_inference(args.explore_model_path)

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
        if "CUDA" in str(e).upper(): # Make CUDA check case-insensitive
            logging.error("A CUDA error occurred during model initialization. "
                          "Check GPU memory, driver compatibility, and model requirements.")
            logging.error("Consider trying device_map='auto' or specifying a single GPU like 'cuda:0' or 'cpu'.")
        raise

# --- Main Orchestration ---
def main(args: argparse.Namespace) -> None:
    """Main function to run the KGQA process."""
    setup_logging(args.debug)
    start_time_main = datetime.datetime.now()
    logging.info(f"Starting KGQA process at {start_time_main.strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Run arguments: {vars(args)}")

    try:
        explore_model_class = get_registed_model(args.explore_model_name)
        predict_model_class = get_registed_model(args.predict_model_name)
        explore_model, predict_model = initialize_models(args, explore_model_class, predict_model_class)
    except Exception as e:
        # initialize_models already logs the error, so just return
        return

    try:
        kg = KnowledgeGraph(uri=args.neo4j_uri, user=args.neo4j_user,
                            password=args.neo4j_password)
        logging.info("Knowledge Graph connection successful.")
    except Exception as e:
        logging.error(f"Fatal error connecting to Knowledge Graph at {args.neo4j_uri}: {e}", exc_info=True)
        return

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
    logging.info(f"Results will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    FileHandler.save_args(output_dir, args)
    predictions_path = os.path.join(output_dir, 'predictions.jsonl')

    try:
        dataset = load_dataset(os.path.join(args.data_path, args.data_name), split=args.split)
        logging.info(f"Successfully loaded dataset '{args.data_name}' split '{args.split}' with {len(dataset)} items.")
    except Exception as e:
        logging.error(f"Fatal error loading dataset: {e}", exc_info=True)
        return

    processed_set = FileHandler.get_processed_ids(predictions_path)
    logging.info(f"Found {len(processed_set)} already processed question IDs from {predictions_path}.")
    if not args.force and len(processed_set) > 0:
        logging.info("Resuming previous run. Use --force to overwrite.")
    elif args.force and os.path.exists(predictions_path): # Only relevant if file exists
        logging.warning("Argument --force specified. Overwriting existing predictions file.")
        processed_set.clear() # Clear for filtering, file will be opened in 'w' mode

    # --- MODIFIED SECTION for target_id_set ---
    target_id_set: Set[str]
    if isinstance(args.target_ids, list): # Check if it was programmatically set to a list
        logging.info(f"Using programmatically provided list of target IDs: {args.target_ids}")
        target_id_set = set(args.target_ids)
    elif isinstance(args.target_ids, str): # Process as string (CLI arg or default)
        target_id_set = load_target_ids(args.target_ids)
    else: # Should not happen with argparse
        logging.warning(f"Unexpected type for args.target_ids: {type(args.target_ids)}. Treating as no target IDs.")
        target_id_set = set()
    # --- END MODIFIED SECTION ---

    to_process = filter_dataset(dataset, target_id_set, processed_set)

    if not to_process:
        logging.info("No questions left to process based on current filters and processed IDs.")
        evaluate_results(predictions_path) # Still try to evaluate if file exists
        return

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
                if result is not None and isinstance(result, dict): # Ensure result is a dict
                    # process_question_wrapper now ensures 'id' and 'error' for failures
                    if "error" in result:
                        failed_count += 1
                        failed_ids.append(result.get("id", q_id)) # Use ID from result if available
                        # Error already logged by wrapper, main log is for summary
                        logging.debug(f"Main: Recorded failure for question ID {result.get('id', q_id)}")
                    else:
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
                else: # Result is None or not a dict (should be handled by wrapper, but as a safeguard)
                    failed_count += 1
                    failed_ids.append(q_id)
                    logging.warning(f"Main: Processing returned None or unexpected type for Q_ID {q_id}. Result: {result}")
                progress_bar.update(1)
            progress_bar.close()

    except IOError as e:
        logging.error(f"Fatal error writing to output file {predictions_path}: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"An unexpected error occurred during sequential processing: {e}", exc_info=True)

    end_time_main = datetime.datetime.now()
    duration_main = end_time_main - start_time_main
    duration_minutes = duration_main.total_seconds() / 60.0

    logging.info("-" * 50)
    logging.info("Processing Summary:")
    logging.info(f"  Total processing duration: {str(duration_main).split('.')[0]} (HH:MM:SS)")
    logging.info(f"  Total time: {duration_minutes:.2f} minutes")
    if processed_count > 0:
        logging.info(f"  Successfully processed: {processed_count} questions")
        avg_time_sec = (duration_main.total_seconds() / processed_count) if processed_count > 0 else 0
        logging.info(f"  Average time per successful question: {avg_time_sec:.2f} seconds")
    else:
        logging.info("  No questions were successfully processed in this run.")

    if failed_count > 0:
        logging.warning(f"  Failed to process: {failed_count} questions")
        logging.warning(f"  Failed IDs: {failed_ids if len(failed_ids) < 20 else str(failed_ids[:20]) + '... (see logs for more)'}")

    logging.info(f"Results saved to: {predictions_path}")
    logging.info("-" * 50)

    if processed_count > 0 or (os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0):
        evaluate_results(predictions_path)
    else:
        logging.info("Skipping evaluation as no results were processed and no prior results file exists or is empty.")

    logging.info("KGQA process finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering via Iterative Reasoning")

    # Data & Paths
    parser.add_argument('--data_path', type=str, default='rmanluo', help="Path to the dataset directory (Hugging Face datasets format)")
    parser.add_argument('--data_name', type=str, default='RoG-webqsp', help="Name of the dataset")
    parser.add_argument('--split', type=str, default='test', help="Dataset split (e.g., 'test', 'train', 'test[:100]')")
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning_Optimized', help="Base directory for saving results")
    # Changed default to empty string, behavior now explicitly handled in main
    parser.add_argument('--target_ids', type=str, default="", help='Comma-separated IDs or path to a file with IDs (one per line) to process only specific questions. Can be overridden programmatically.')

    # Execution Control
    parser.add_argument('--force', action='store_true', help='Force overwrite existing predictions file instead of appending.')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--prefix', type=str, default="", help="Prefix for the output subdirectory name")
    # Max workers is not used in the provided sequential processing logic, but kept for potential future use
    parser.add_argument('--max_workers', type=int, default=None, help="Max worker threads for parallel processing (default: auto-adjust, CURRENTLY UNUSED)")

    # Model Configuration
    parser.add_argument('--explore_model_name', type=str, default='gcr-Llama-2-7b-chat-hf', help="Name of the registered exploration model")
    parser.add_argument('--explore_model_path', type=str, default=None, help="Path to the exploration model weights/directory (optional)")
    parser.add_argument('--predict_model_name', type=str, default='gcr-Llama-2-7b-chat-hf', help="Name of the registered prediction model")
    parser.add_argument('--predict_model_path', type=str, default=None, help="Path to the prediction model weights/directory (optional, defaults to explore_model_path if same model)")

    # KG & Explorer Configuration
    parser.add_argument('--max_rounds', type=int, default=2, help="Maximum rounds of iterative reasoning in KnowledgeExplorer")
    parser.add_argument('--max_selection_count', type=int, default=5, help="Maximum number of relations to explore per step in KnowledgeExplorer")
    parser.add_argument('--neo4j_uri', type=str, default=DEFAULT_NEO4J_URI, help="Neo4j connection URI")
    parser.add_argument('--neo4j_user', type=str, default=DEFAULT_NEO4J_USER, help="Neo4j username")
    parser.add_argument('--neo4j_password', type=str, default=DEFAULT_NEO4J_PASSWORD, help="Neo4j password (better to set via NEO4J_PASSWORD env var)")
    parser.add_argument('--embedding_model', type=str, default='msmarco-distilbert-base-tas-b', help="Sentence Transformer model for embeddings in KG")

    # Parse base args first to get model names
    # This ensures model-specific args are added before final parsing
    temp_args_for_model_loading, remaining_argv = parser.parse_known_args()

    try:
        explore_model_class = get_registed_model(temp_args_for_model_loading.explore_model_name)
        explore_model_class.add_args(parser) # Add explore model's specific arguments
        logging.debug(f"Added args for explore model: {temp_args_for_model_loading.explore_model_name}")

        if temp_args_for_model_loading.explore_model_name != temp_args_for_model_loading.predict_model_name:
            predict_model_class = get_registed_model(temp_args_for_model_loading.predict_model_name)
            predict_model_class.add_args(parser) # Add predict model's specific arguments
            logging.debug(f"Added args for distinct predict model: {temp_args_for_model_loading.predict_model_name}")
        else:
            logging.debug("Explore and predict models are the same type, not adding predict args separately.")
    except Exception as e:
        logging.error(f"Failed to get or add arguments for models: {e}", exc_info=True)
        exit(1) # Critical error, cannot proceed with misconfigured args

    # Parse all arguments including the dynamically added ones and original system args
    # Need to re-parse all args as add_args might have added new ones.
    args = parser.parse_args()


    # --- Security Warning for Password ---
    if args.neo4j_password == "YOUR_PASSWORD_HERE" or args.neo4j_password == "Martin1007Wang": # Check against placeholder and old default
        logging.warning("Neo4j password is set to a default or placeholder value. "
                        "It is strongly recommended to set the password via the NEO4J_PASSWORD environment variable "
                        "or use a secure configuration method instead of command-line arguments.")

    # --- Programmatic Override for target_ids ---
    # This line now sets args.target_ids to a list.
    # The main() function has been updated to handle this case.
    args.target_ids = ["WebQTest-26", "WebQTest-1259", "WebQTest-264", "WebQTest-688", "WebQTest-189","WebQTest-179","WebQTest-71","WebQTest-278","WebQTest-751","WebQTest-202"]
    logging.info(f"Programmatically overriding target_ids to: {args.target_ids}")


    main(args)