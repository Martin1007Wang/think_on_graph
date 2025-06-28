
import json
import logging
import string
from functools import lru_cache, wraps
import argparse
import os
from typing import Set, List, Dict, Any
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def read_prompt(prompt_path):
    with open(prompt_path, 'r') as f:
        prompt_template = f"""{f.read()}"""
    return prompt_template

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_multiple_jsonl(file_path_list):
    data = []
    for path in file_path_list:
        data.extend(load_jsonl(path))
    return data

def list_to_string(l: list) -> str:
    prompt = '"{}"'
    return ', '.join([prompt.format(i) for i in l])

def rule_to_string(rule: list, sep_token = "<SEP>", bop = "<PATH>", eop = "</PATH>") -> str:
    if len(rule) == 1:
        rule_string = rule[0]
    else:
        rule_string = sep_token.join(rule)
    return bop + rule_string + eop

class InstructFormater(object):
    def __init__(self, prompt_path):
        '''
        _summary_

        Args:
            prompt_template (_type_): 
            instruct_template (_type_): _description_
        '''
        self.prompt_template = read_prompt(prompt_path)

    def format(self, instruction, message):
        return self.prompt_template.format(instruction=instruction, input=message)
    
def error_handler(default_return=None, error_message="Error in operation"):
    """集中式错误处理装饰器，减少重复的try-except块"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{error_message}: {e}", exc_info=True)
                return default_return
        return wrapper
    return decorator

def setup_logging(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    log_format = '%(asctime)s - %(levelname)s - [%(threadName)s] - %(name)s - %(message)s'
    logging.basicConfig(level=log_level, format=log_format)
    logging.info(f"Logging setup with level: {'DEBUG' if debug else 'INFO'}")

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

            
