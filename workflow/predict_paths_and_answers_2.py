import os
import argparse
import json
import logging
import datetime
from typing import Set, List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from datasets import load_dataset
from src.llms import get_registed_model
from src.utils.qa_utils import eval_path_result_w_ans
from src.knowledge_graph import KnowledgeGraph
from src.knowledge_explorer import KnowledgeExplorer


class FileHandler:
    @staticmethod
    def get_output_file(path: str, force: bool = False) -> Tuple[Any, List[str]]:
        """Get file handle and list of already processed IDs."""
        processed = []
        if not os.path.exists(path) or force:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return open(path, "w"), processed
        
        try:
            with open(path, "r") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if "id" in item:
                            processed.append(item["id"])
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logging.warning(f"Error reading existing file {path}: {e}")
            return open(path, "w"), processed
            
        return open(path, "a"), processed

    @staticmethod
    def save_args(output_dir: str, args: argparse.Namespace) -> None:
        """Save command line arguments to a file."""
        os.makedirs(output_dir, exist_ok=True)
        args_path = os.path.join(output_dir, 'args.txt')
        try:
            with open(args_path, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            logging.info(f"Arguments saved to {args_path}")
        except Exception as e:
            logging.error(f"Failed to save arguments: {e}")


def load_target_ids(target_ids_arg: str) -> Set[str]:
    """Load target IDs from file or comma-separated string."""
    target_id_set = set()
    if not target_ids_arg:
        return target_id_set
        
    if os.path.isfile(target_ids_arg):
        try:
            with open(target_ids_arg) as f:
                target_id_set = set(line.strip() for line in f if line.strip())
            logging.info(f"Loaded {len(target_id_set)} target IDs from file")
        except Exception as e:
            logging.error(f"Error loading target IDs file: {e}")
    else:
        target_id_set = set(i.strip() for i in target_ids_arg.split(',') if i.strip())
        logging.info(f"Using {len(target_id_set)} target IDs from command line")
    
    return target_id_set


def filter_dataset(dataset: Any, target_id_set: Set[str]) -> List[Dict[str, Any]]:
    """Filter dataset based on target IDs."""
    if not target_id_set:
        logging.info(f"Processing all {len(dataset)} questions in dataset")
        return dataset
        
    filtered_dataset = [d for d in dataset if d.get("id") in target_id_set]
    logging.info(f"Found {len(filtered_dataset)} questions matching target IDs")
    return filtered_dataset


def setup_logging(debug: bool = False) -> None:
    """Configure logging settings."""
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)


def evaluate_results(predictions_path: str, step: Optional[int] = None) -> None:
    """Evaluate prediction results."""
    if not os.path.exists(predictions_path) or os.path.getsize(predictions_path) == 0:
        logging.warning("No predictions to evaluate")
        return
        
    prefix = f"[Step {step}]" if step is not None else "[Final]"
    logging.info(f"{prefix} Evaluating results...")
    
    try:
        eval_path_result_w_ans(predictions_path)
    except Exception as e:
        logging.error(f"Error during evaluation: {e}", exc_info=True)


def process_question(explorer: KnowledgeExplorer, data: Dict[str, Any], 
                    processed_set: Set[str], debug: bool) -> Optional[Dict[str, Any]]:
    """Process a single question with error handling."""
    question_id = data.get("id", "unknown_id")
    if question_id in processed_set:
        return None
        
    try:
        res = explorer.process_question(data, processed_set)
        if res is not None and debug:
            logging.debug(f"Processed question {question_id} successfully")
        return res
    except Exception as e:
        logging.error(f"Error processing id {question_id}: {e}", exc_info=True)
        return None


def main(args: argparse.Namespace) -> None:
    setup_logging(args.debug)
    kg = None
    fout = None
    start_time = datetime.datetime.now()
    logging.info(f"Starting knowledge graph QA process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Initialize knowledge graph
        try:
            kg = KnowledgeGraph(
                uri=args.neo4j_uri, 
                user=args.neo4j_user, 
                password=args.neo4j_password, 
                model_name=args.embedding_model
            )
        except Exception as e:
            logging.error(f"Failed to initialize knowledge graph: {e}", exc_info=True)
            return
            
        # Load dataset
        try:
            dataset = load_dataset(os.path.join(args.data_path, args.data_name), split=args.split)
            logging.info(f"Successfully loaded dataset with {len(dataset)} items")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}", exc_info=True)
            return

        # Setup output paths
        post_fix = f"{args.prefix}iterative-rounds{args.max_rounds}-topk{args.max_k_relations}"
        data_name = args.data_name + "_undirected" if args.undirected else args.data_name
        output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
        logging.info(f"Results will be saved to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize language model
        try:
            LLM = get_registed_model(args.model_name)
            model = LLM(args)
            model.prepare_for_inference()
            logging.info("Model initialization successful")
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}", exc_info=True)
            return
            
        # Setup output file and track processed questions
        FileHandler.save_args(output_dir, args)
        predictions_path = os.path.join(output_dir, 'predictions.jsonl')
        fout, processed_list = FileHandler.get_output_file(predictions_path, force=args.force)
        processed_set = set(processed_list)
        logging.info(f"Found {len(processed_set)} already processed questions")
        
        # Initialize knowledge explorer
        explorer = KnowledgeExplorer(
            kg=kg, 
            model=model,
            max_rounds=args.max_rounds,
            max_k_relations=args.max_k_relations
        )
        
        # Prepare dataset
        target_id_set = load_target_ids(args.target_ids)
        dataset_iter = filter_dataset(dataset, target_id_set)
        
        if not dataset_iter:
            logging.warning("No questions to process. Check your target_ids or dataset.")
            return
            
        # Process questions
        success_count = 0
        error_count = 0
        skip_count = 0
        eval_every = min(100, max(1, len(dataset_iter) // 10))  # Dynamic evaluation frequency
        processed = 0
        
        for idx, data in enumerate(tqdm(dataset_iter, desc="Processing questions", total=len(dataset_iter))):
            question_id = data.get("id", "unknown_id")
            if question_id in processed_set:
                skip_count += 1
                continue
                
            result = process_question(explorer, data, processed_set, args.debug)
            
            if result is not None:
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                success_count += 1
                processed += 1
            else:
                error_count += 1

            # Periodic evaluation
            if processed > 0 and processed % eval_every == 0:
                fout.flush()
                evaluate_results(predictions_path, processed)

        # Final statistics and evaluation
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        logging.info(f"Processing completed in {duration:.2f} minutes")
        logging.info(f"Statistics: {success_count} successful, {error_count} errors, {skip_count} skipped")

        if fout:
            fout.close()
            fout = None
            
        evaluate_results(predictions_path)
            
    except Exception as e:
        logging.error(f"Unexpected error in main process: {e}", exc_info=True)
    finally:
        if fout and not fout.closed:
            fout.close()
        if kg:
            try:
                kg.close()
                logging.info("Knowledge graph connection closed")
            except Exception as e:
                logging.error(f"Error closing knowledge graph: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering via Iterative Reasoning")
    parser.add_argument('--data_path', type=str, default='rmanluo')
    parser.add_argument('--data_name', type=str, default='RoG-webqsp')
    parser.add_argument('--split', type=str, default='test[:100]')
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning_v7')
    parser.add_argument('--force', action='store_true', help='Force overwrite existing results')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--prefix', type=str, default="")
    parser.add_argument('--model_name', type=str, default='gcr-Llama-2-7b-chat-hf')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--max_rounds', type=int, default=3)
    parser.add_argument('--max_k_relations', type=int, default=5)
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang')
    parser.add_argument('--embedding_model', type=str, default='msmarco-distilbert-base-tas-b')
    parser.add_argument('--undirected', type=lambda x: (str(x).lower() == 'true'), default=False)
    parser.add_argument('--target_ids', type=str, default="", help='Comma-separated IDs or path to file with IDs')
    
    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()
    
    main(args)
