import os
from typing import List, Tuple
import logging
import argparse
from src.knowledge_graph import KnowledgeGraph
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING) # 假设你使用了官方的neo4j驱动
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Load one or more datasets into Neo4j.")
    parser.add_argument('--data_path', type=str, default='rmanluo', 
                        help="Base path or prefix for dataset names (e.g., a Hugging Face username or local directory).")
    parser.add_argument('--dataset_names', '-d', type=str, nargs='+', default=['RoG-webqsp'],
                        help="List of dataset names (e.g., 'RoG-webqsp', 'MyDataset').")
    parser.add_argument('--splits', '-s', type=str, nargs='+', default=['train'],
                        help="List of corresponding dataset splits (e.g., 'train', 'test', 'validation').")
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help="Neo4j URI.")
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help="Neo4j username.")
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang', help="Neo4j password.")
    parser.add_argument('--clear', type=bool, default=True, 
                        help="Clear Neo4j database before loading any data. Default is True.")
    
    args = parser.parse_args()

    logger.info("Running with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    if len(args.dataset_names) != len(args.splits):
        logger.error("Error: The number of dataset names must match the number of splits.")
        logger.error(f"  Number of dataset names: {len(args.dataset_names)} ({args.dataset_names})")
        logger.error(f"  Number of splits: {len(args.splits)} ({args.splits})")
        return
    kg = KnowledgeGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password
    )
    try:
        if args.clear:
            logger.info("Clearing the database as per --clear=True flag.")
            kg.clear_database()
        for i in range(len(args.dataset_names)):
            current_dataset_name = args.dataset_names[i]
            current_split = args.splits[i]
            logger.info(f"Processing dataset '{current_dataset_name}' with split '{current_split}' ({i+1}/{len(args.dataset_names)}).")
            if args.data_path and args.data_path.strip(): # Ensure data_path is not empty or just whitespace
                dataset_identifier = os.path.join(args.data_path, current_dataset_name)
            else:
                dataset_identifier = current_dataset_name
            
            logger.info(f"Constructed dataset identifier: '{dataset_identifier}' for dataset name '{current_dataset_name}' and split '{current_split}'.")
            if not os.path.exists(dataset_identifier) and not ('/' in dataset_identifier and args.data_path): # Heuristic: if it looks like HF ID (contains '/') AND data_path was used, it might be an ID
                logger.warning(f"Warning: Local path '{dataset_identifier}' does not exist. "
                               f"If this is a Hugging Face dataset ID, this warning might be ignorable "
                               f"if the loading function handles it.")
            
            # If dataset_identifier is intended to be a directory, you might want os.path.isdir(dataset_identifier)
            # The original warning said "... does not exist as a file".

            kg.load_graph_from_dataset(dataset_identifier, hf_dataset_split=current_split)
            logger.info(f"Successfully initiated loading for dataset '{current_dataset_name}', split '{current_split}'.")

    except Exception as e:
        logger.error(f"An error occurred during the process: {e}", exc_info=True)
    finally:
        logger.info("Closing KnowledgeGraph connection.")
        kg.close()

if __name__ == "__main__":
    main()