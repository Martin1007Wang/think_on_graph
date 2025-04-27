import os
import argparse
from typing import Any
import json
import logging
from tqdm import tqdm
from datasets import load_dataset
from src.llms import get_registed_model
from src.utils.qa_utils import eval_path_result_w_ans
from src.knowledge_graph import KnowledgeGraph
from src.knowledge_explorer import KnowledgeExplorer
import datetime

class FileHandler:
    @staticmethod
    def get_output_file(path: str, force: bool = False):
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
    def save_args(output_dir: str, args: argparse.Namespace):
        os.makedirs(output_dir, exist_ok=True)
        args_path = os.path.join(output_dir, 'args.txt')
        try:
            with open(args_path, 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            logging.info(f"Arguments saved to {args_path}")
        except Exception as e:
            logging.error(f"Failed to save arguments: {e}")


def main(args: argparse.Namespace):
    kg = None
    fout = None
    start_time = datetime.datetime.now()
    logging.info(f"Starting knowledge graph QA process at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
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
            
        try:
            dataset = load_dataset(os.path.join(args.data_path, args.data_name), split=args.split)
            logging.info(f"Successfully loaded dataset with {len(dataset)} items")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}", exc_info=True)
            return

        post_fix = f"{args.prefix}iterative-rounds{args.max_rounds}-topk{args.max_k_relations}"
        data_name = args.data_name + "_undirected" if args.undirected else args.data_name
        output_dir = os.path.join(args.predict_path, data_name, args.model_name, args.split, post_fix)
        logging.info(f"Results will be saved to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            LLM = get_registed_model(args.model_name)
            model = LLM(args)
            model.prepare_for_inference()
            logging.info("Model initialization successful")
        except Exception as e:
            logging.error(f"Failed to initialize language model: {e}", exc_info=True)
            return
            
        FileHandler.save_args(output_dir, args)
        predictions_path = os.path.join(output_dir, 'predictions.jsonl')
        fout, processed_list = FileHandler.get_output_file(predictions_path, force=args.force)
        processed_set = set(processed_list)
        logging.info(f"Found {len(processed_set)} already processed questions")
        
        explorer = KnowledgeExplorer(
            kg=kg, 
            model=model,
            max_rounds=args.max_rounds,
            max_k_relations=args.max_k_relations
        )
        
        dataset_iter = _prepare_dataset_iterator(args, dataset)
        total = len(dataset_iter)
        
        if total == 0:
            logging.warning("No questions to process. Check your target_ids or dataset.")
            return
            
        success_count = 0
        error_count = 0
        skip_count = 0
        eval_every = 100
        processed = 0
        total = len(dataset_iter)
        for idx, data in enumerate(tqdm(dataset_iter, desc="Processing questions", total=total)):
            question_id = data.get("id", "unknown_id")
            if question_id in processed_set:
                skip_count += 1
                continue
            try:
                res = explorer.process_question(data, processed_set)
                if res is not None:
                    fout.write(json.dumps(res) + "\n")
                    fout.flush()
                    success_count += 1
                    processed += 1
                    if args.debug:
                        logging.debug(f"Processed question {question_id} successfully")
            except Exception as e:
                logging.error(f"Error processing id {question_id}: {e}", exc_info=True)
                error_count += 1

            if processed > 0 and processed % eval_every == 0:
                fout.flush()
                if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
                    logging.info(f"[Step] Evaluating results after {processed} processed...")
                    try:
                        eval_path_result_w_ans(predictions_path)
                    except Exception as e:
                        logging.error(f"Error during step evaluation: {e}", exc_info=True)

        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60.0
        logging.info(f"Processing completed in {duration:.2f} minutes")
        logging.info(f"Statistics: {success_count} successful, {error_count} errors, {skip_count} skipped")

        if fout:
            fout.close()
            fout = None
        if os.path.exists(predictions_path) and os.path.getsize(predictions_path) > 0:
            logging.info("[Final] Evaluating results...")
            try:
                eval_path_result_w_ans(predictions_path)
            except Exception as e:
                logging.error(f"Error during final evaluation: {e}", exc_info=True)
        else:
            logging.warning("No predictions to evaluate")
            
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


def _prepare_dataset_iterator(args, dataset):
    target_id_set = set()
    if getattr(args, "target_ids", None):
        if os.path.isfile(args.target_ids):
            try:
                with open(args.target_ids) as f:
                    target_id_set = set(line.strip() for line in f if line.strip())
                logging.info(f"Loaded {len(target_id_set)} target IDs from file")
            except Exception as e:
                logging.error(f"Error loading target IDs file: {e}")
        else:
            target_id_set = set(i.strip() for i in args.target_ids.split(',') if i.strip())
            logging.info(f"Using {len(target_id_set)} target IDs from command line")
    
    if target_id_set:
        dataset_iter = [d for d in dataset if d.get("id") in target_id_set]
        logging.info(f"Found {len(dataset_iter)} questions matching target IDs")
    else:
        dataset_iter = dataset
        logging.info(f"Processing all {len(dataset_iter)} questions in dataset")
        
    return dataset_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Graph Question Answering via Iterative Reasoning")
    parser.add_argument('--data_path', type=str, default='rmanluo')
    parser.add_argument('--data_name', type=str, default='RoG-webqsp')
    parser.add_argument('--split', type=str, default='test[:100]')
    parser.add_argument('--predict_path', type=str, default='results/IterativeReasoning_v6')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--debug', action='store_true')
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
    parser.add_argument('--target_ids', type=str, default="")
    args, _ = parser.parse_known_args()
    LLM = get_registed_model(args.model_name)
    LLM.add_args(parser)
    args = parser.parse_args()
    main(args)