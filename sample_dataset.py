#!/usr/bin/env python3
import json
import pickle
from typing import List, Dict, Any
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def extract_ids_from_jsonl(file_path: str) -> List[str]:
    """
    Extract IDs from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file.
        
    Returns:
        List of extracted IDs.
    """
    ids = []
    try:
        with open(file_path, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        ids.append(data["id"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line as JSON: {line[:50]}...")
                    continue
        
        logger.info(f"Extracted {len(ids)} IDs from {file_path}")
        return ids
    
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.error(f"Error extracting IDs from {file_path}: {e}")
        return []

def save_list_to_pickle(obj: List[str], file_path: str) -> bool:
    """
    Save a list object to a pickle file.
    
    Args:
        obj: The list to save.
        file_path: Path where to save the pickle file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved list with {len(obj)} items to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving list to {file_path}: {e}")
        return False

def save_list_to_txt(obj: List[str], file_path: str) -> bool:
    """
    Save a list object to a text file, one item per line.
    
    Args:
        obj: The list to save.
        file_path: Path where to save the text file.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            for item in obj:
                f.write(f"{item}\n")
        logger.info(f"Saved list with {len(obj)} items to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving list to {file_path}: {e}")
        return False

def main():
    # Path to the sampled predictions file
    sampled_predictions_path = "/mnt/wangjingxiong/think_on_graph/results/IterativeReasoning_v10/RoG-webqsp/GCR-lora-dpo_v3_with_label-Llama-3.1-8B-Instruct-all-available-relations/deepseek-chat/iterative-rounds2-topk5/sampled_predictions.jsonl"
    
    # Extract IDs
    ids = extract_ids_from_jsonl(sampled_predictions_path)
    
    if not ids:
        logger.error("No IDs were extracted.")
        return
    
    # Save to pickle file
    output_dir = "/mnt/wangjingxiong/think_on_graph/sampled_data"
    pickle_path = f"{output_dir}/sampled_ids.pkl"
    save_list_to_pickle(ids, pickle_path)
    
    # Also save as text file for easier viewing
    txt_path = f"{output_dir}/sampled_ids.txt"
    save_list_to_txt(ids, txt_path)
    
    # Print sample of IDs
    logger.info(f"Sample of extracted IDs (first 5): {ids[:5]}")

if __name__ == "__main__":
    main() 