import json
import argparse
from typing import List, Tuple, Dict, Any
import gc
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from src.knowledge_graph import KnowledgeGraph
from src.path_generator import PathGenerator
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_sample(
    sample: Dict[str, Any],
    path_generator: PathGenerator,
    max_pairs: int = 5,
    max_negatives_per_pair: int = 5,
    add_semantic_entities: bool = True,
    semantic_entities_count: int = 3
) -> Dict[str, Any]:
    """Process a single sample to generate paths, prioritizing real entities and incrementally adding semantic entities."""
    q_entities = sample['q_entity'] if isinstance(sample['q_entity'], list) else [sample['q_entity']]
    a_entities = sample.get('a_entity', []) if isinstance(sample.get('a_entity', []), list) else [sample.get('a_entity', '')]
    question = sample['question']

    real_q_entities = [q for q in q_entities if q]
    semantic_q_entities = []

    if add_semantic_entities and path_generator.kg:
        related_entities = path_generator.kg.get_related_entities_by_question(
            question=question, n=semantic_entities_count
        )
        semantic_q_entities = [entity_id for entity_id, score in related_entities if entity_id not in real_q_entities]

    sample_result = {
        "id": sample.get('id', 'unknown'),
        "question": question,
        "q_entities": real_q_entities.copy(),
        "a_entities": a_entities,
        "paths": []
    }

    pairs = [(q, a) for q in real_q_entities for a in a_entities if q and a]
    positive_found = False

    for i, (q_entity, a_entity) in enumerate(pairs):
        if i >= max_pairs:
            break
        golden_path = path_generator.get_golden_path(q_entity, a_entity)
        positive_path, visited = path_generator.get_positive_path(q_entity, a_entity, question)
        
        if positive_path:
            negative_paths = path_generator.get_negative_paths(
                positive_path, question, a_entity, max_negatives_per_pair
            )
            negative_paths = negative_paths[:max_negatives_per_pair]
            pair = {
                "q_entity": q_entity,
                "a_entity": a_entity,
                "golden_path": format_path_for_json(golden_path),
                "positive_path": format_path_for_json(positive_path),
                "negative_paths": [format_path_for_json(np) for np in negative_paths]
            }
            sample_result["paths"].append(pair)
            positive_found = True
            logger.info(f"Positive path found for real entity {q_entity} to {a_entity}")
            break

    if not positive_found and semantic_q_entities:
        logger.debug("No positive path found with real entities, incrementally trying semantic entities.")
        for q_entity in semantic_q_entities:
            pairs = [(q_entity, a) for a in a_entities if a]
            for i, (q_entity, a_entity) in enumerate(pairs):
                if i >= max_pairs:
                    break
                golden_path = path_generator.get_golden_path(q_entity, a_entity)
                positive_path, visited = path_generator.get_positive_path(q_entity, a_entity, question)
                
                if positive_path:
                    negative_paths = path_generator.get_negative_paths(
                        positive_path, question, a_entity, max_negatives_per_pair
                    )
                    negative_paths = negative_paths[:max_negatives_per_pair]
                    pair = {
                        "q_entity": q_entity,
                        "a_entity": a_entity,
                        "golden_path": format_path_for_json(golden_path),
                        "positive_path": format_path_for_json(positive_path),
                        "negative_paths": [format_path_for_json(np) for np in negative_paths]
                    }
                    sample_result["paths"].append(pair)
                    sample_result["q_entities"].append(q_entity)
                    positive_found = True
                    logger.info(f"Positive path found with semantic entity {q_entity} to {a_entity}")
                    break
            if positive_found:
                break

    if not positive_found:
        logger.warning(f"No positive path found for question: {question}")

    return sample_result

def process_samples(
    samples: Dict[str, List[Any]],
    path_generator: PathGenerator,
    max_pairs: int = 5,
    max_negatives_per_pair: int = 5,
    add_semantic_entities: bool = True,
    semantic_entities_count: int = 3
) -> List[Dict[str, Any]]:
    """Process multiple samples from a datasets.Dataset dictionary and return results with processing time."""
    results = []
    total_start_time = time.time()

    num_samples = len(samples['question'])  # Use length of 'question' to determine sample count
    if not num_samples:
        logger.warning("No samples found in input dictionary")
        return results

    for idx in range(num_samples):
        # Construct single sample dictionary from the dataset dictionary
        single_sample = {key: samples[key][idx] for key in samples.keys()}
        
        start_time = time.time()
        result = process_single_sample(
            single_sample, path_generator, max_pairs, max_negatives_per_pair,
            add_semantic_entities, semantic_entities_count
        )
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        results.append(result)
        logger.info(f"Processed sample {idx + 1}/{num_samples} (ID: {result['id']}), time: {processing_time:.2f} seconds")

    total_processing_time = time.time() - total_start_time
    logger.info(f"Total processing time for {num_samples} samples: {total_processing_time:.2f} seconds")
    return results

def format_path_for_json(path: List[Tuple[str, str, str]]) -> str:
    """Format a path as a string for JSON output."""
    return "; ".join([f"{s} {r} {t}" for s, r, t in path]) if path else ""

def verify_path_generation():
    parser = argparse.ArgumentParser(description="验证路径生成功能")
    parser.add_argument('--data_path', type=str, default='rmanluo/RoG-webqsp', help='数据集路径')
    parser.add_argument('--dataset', '-d', type=str, default='RoG-webqsp')
    parser.add_argument('--split', type=str, default='train', help='数据集分割')
    parser.add_argument('--num_samples', type=int, default=20, help='处理的样本数量')
    parser.add_argument('--model_name', type=str, default='msmarco-distilbert-base-tas-b', help='Pretrained model name')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Neo4j用户名')
    parser.add_argument('--neo4j_password', type=str, default='Martin1007Wang', help='Neo4j密码')
    parser.add_argument('--output_file', type=str, default='path_verification.json', help='输出文件路径')
    parser.add_argument('--sample_index', type=int, default=None, help='要处理的样本索引（可选）')
    
    args = parser.parse_args()
    
    logger.info(f"加载数据集: {args.data_path}")
    dataset = load_dataset(args.data_path, split=args.split)
    
    logger.info("初始化知识图谱")
    kg = KnowledgeGraph(
        uri=args.neo4j_uri, user=args.neo4j_user, password=args.neo4j_password,
        model_name=args.model_name
    )
    kg.initialize_embeddings(dataset=args.dataset, split=args.split)
    
    path_generator = PathGenerator(
        kg=kg,
        max_path_length=3,
        top_k_relations=5,
    )
    
    try:
        # Process samples based on arguments
        if args.sample_index is not None:
            if args.sample_index >= len(dataset):
                raise ValueError(f"Sample index {args.sample_index} exceeds dataset size {len(dataset)}")
            samples = dataset[args.sample_index]  # Single sample as a dict
            output_filename = f'path_verification_sample_{args.sample_index}.json'
            results = [process_single_sample(
                samples, path_generator,
                max_pairs=5, max_negatives_per_pair=5,
                add_semantic_entities=True, semantic_entities_count=3
            )]
            results[0]['processing_time'] = time.time() - time.time()  # Dummy time for consistency
        else:
            if args.num_samples > len(dataset):
                logger.warning(f"Requested {args.num_samples} samples, but dataset has only {len(dataset)}")
                args.num_samples = len(dataset)
            samples = dataset[:args.num_samples]  # Multiple samples as a dict
            output_filename = args.output_file
            results = process_samples(
                samples, path_generator,
                max_pairs=5, max_negatives_per_pair=5,
                add_semantic_entities=True, semantic_entities_count=3
            )

        # Save results
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"处理完成，结果已保存到: {output_filename}")

    finally:
        kg.close()
        gc.collect()

if __name__ == "__main__":
    verify_path_generation()