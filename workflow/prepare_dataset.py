import json
import argparse
import os
import gc
import logging
from typing import List, Tuple, Dict, Any, Optional
from datasets import load_dataset, Dataset
from src.knowledge_graph import KnowledgeGraph
from src.path_generator import PathGenerator
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def format_path_for_json(path: List[Tuple[str, str, str]]) -> str:
    return " ; ".join(f"{src} {rel} {tgt}" for src, rel, tgt in path) if path else ""

def process_entity_pair(
    q_entity: str,
    a_entity: str,
    question: str,
    path_generator: PathGenerator,
    max_negatives_per_pair: int
) -> Optional[Dict[str, Any]]:
    if not q_entity or not a_entity:
        return None
    try:
        golden_path = path_generator.get_golden_path(q_entity, a_entity)
        positive_path, _ = path_generator.get_positive_path(q_entity, a_entity, question)
        if not positive_path:
            return None
        negative_paths = path_generator.get_negative_paths(positive_path, question, a_entity, max_negatives_per_pair)
        negative_paths = negative_paths[:max_negatives_per_pair]
        return {
            "q_entity": q_entity,
            "a_entity": a_entity,
            "golden_path": format_path_for_json(golden_path),
            "positive_path": format_path_for_json(positive_path),
            "negative_paths": [format_path_for_json(np) for np in negative_paths]
        }
    except Exception as e:
        logger.debug(f"Failed to process pair {q_entity} -> {a_entity}: {e}")
        return None

def process_single_sample(
    sample: Dict[str, Any],
    path_generator: PathGenerator,
    max_pairs: int = 5,
    max_negatives_per_pair: int = 5,
    add_semantic_entities: bool = True,
    semantic_entities_count: int = 3
) -> Dict[str, Any]:
    q_entities = sample['q_entity'] if isinstance(sample['q_entity'], list) else [sample['q_entity']]
    a_entities = sample.get('a_entity', []) if isinstance(sample.get('a_entity', []), list) else [sample.get('a_entity', '')]
    question = sample['question']

    real_q_entities = [q for q in q_entities if q]
    sample_result = {
        "id": sample.get('id', 'unknown'),
        "question": question,
        "q_entities": real_q_entities.copy(),
        "a_entities": a_entities,
        "paths": []
    }

    all_entities = real_q_entities[:]
    if add_semantic_entities and path_generator.kg and not sample_result["paths"]:
        related_entities = path_generator.kg.get_related_entities_by_question(question, n=semantic_entities_count)
        semantic_q_entities = [entity_id for entity_id, _ in related_entities if entity_id not in real_q_entities]
        all_entities.extend(semantic_q_entities)

    for q_entity in all_entities:
        pairs = [(q_entity, a) for a in a_entities if a][:max_pairs]
        for q_entity, a_entity in pairs:
            pair_result = process_entity_pair(q_entity, a_entity, question, path_generator, max_negatives_per_pair)
            if pair_result:
                sample_result["paths"].append(pair_result)
                if q_entity not in real_q_entities:
                    sample_result["q_entities"].append(q_entity)
                return sample_result
    if not sample_result["paths"]:
        logger.warning(f"No positive path found for question: {question}")
    return sample_result

def prepare_dataset(args: argparse.Namespace):
    logger.info(f"Loading dataset: {args.data_path}")
    try:
        dataset = load_dataset(args.data_path, split=args.split)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise

    logger.info("Initializing knowledge graph")
    kg = None
    try:
        kg = KnowledgeGraph(
            uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            model_name=args.model_name
        )
        kg.initialize_embeddings(dataset=args.dataset, split=args.split)
    except Exception as e:
        logger.error(f"Failed to initialize knowledge graph: {e}")
        raise

    path_generator = PathGenerator(
        kg=kg,
        max_path_length=args.max_path_length,
        top_k_relations=args.top_k_relations,
    )

    try:
        output_dir = os.path.join(args.output_path, args.output_name)
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for sample in tqdm(dataset, desc="Processing dataset samples"):
            if not isinstance(sample, dict) or 'question' not in sample:
                logger.warning(f"Skipping invalid sample: {sample.get('id', 'unknown')}")
                continue
            result = process_single_sample(
                sample, path_generator, args.max_pairs, args.max_negatives_per_pair,
                args.add_semantic_entities, args.semantic_entities_count
            )
            results.append(result)
            if len(results) % 1000 == 0:
                path_generator.relation_cache.clear()
                gc.collect()

        json_output_path = os.path.join(output_dir, 'data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved as JSON to {json_output_path}")

        processed_dataset = Dataset.from_list(results)
        processed_dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved in Dataset format to {output_dir}")

    finally:
        if kg:
            kg.close()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset with various path types")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--dataset', type=str, default='RoG-webqsp', help='Dataset name')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--output_name', type=str, default='path_enhanced_dataset', help='Name of the output dataset')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default='password', help='Neo4j password')
    parser.add_argument('--max_path_length', type=int, default=3, help='Maximum path length')
    parser.add_argument('--top_k_relations', type=int, default=5, help='Top-K relations to consider')
    parser.add_argument('--max_pairs', type=int, default=5, help='Maximum pairs per sample')
    parser.add_argument('--max_negatives_per_pair', type=int, default=5, help='Maximum negatives per pair')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='Pretrained model name')
    parser.add_argument('--add_semantic_entities', type=bool, default=True, help='Whether to add semantic entities')
    parser.add_argument('--semantic_entities_count', type=int, default=3, help='Number of semantic entities to add')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')

    args = parser.parse_args()
    logging.getLogger().setLevel(args.log_level)
    prepare_dataset(args)