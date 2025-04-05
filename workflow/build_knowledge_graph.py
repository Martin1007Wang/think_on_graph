import os
from typing import List, Tuple
import logging
from src.knowledge_graph import KnowledgeGraph

logging.basicConfig(level=logging.INFO)
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='rmanluo')
    parser.add_argument('--dataset', '-d', type=str, default='RoG-webqsp')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687')
    parser.add_argument('--neo4j_user', type=str, default='neo4j')
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='Pretrained model name')
    parser.add_argument('--clear', type=bool, default=True, help='Clear database before loading')
    
    args = parser.parse_args()

    logger.info("Running with arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    kg = KnowledgeGraph(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        model_name=args.model_name,
    )
    
    try:
        if args.clear:
            kg.clear_database()

        input_file = os.path.join(args.data_path, args.dataset)
        logger.info(f"Full dataset path: {input_file}")

        if not os.path.exists(input_file):
            logger.warning(f"Warning: {input_file} does not exist as a file")

        kg.load_graph_from_dataset(input_file, args.dataset, args.split)
    finally:
        kg.close()

if __name__ == "__main__":
    main()