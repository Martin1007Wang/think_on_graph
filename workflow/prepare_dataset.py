import torch
import json
import argparse
from typing import List, Tuple, Set
from collections import deque
import random
import gc
import logging
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from workflow.build_knowledge_graph import KnowledgeGraph
from tqdm import tqdm
import os
from datasets import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(
        self,
        kg: KnowledgeGraph,
        similarity_model: SentenceTransformer,
        max_path_length: int = 3,
        top_k_relations: int = 5,
        max_backtrack: int = 5,
    ):
        self.kg = kg
        self.similarity_model = similarity_model
        self.max_path_length = max_path_length
        self.top_k_relations = top_k_relations
        self.max_backtrack = max_backtrack
        self.relation_cache = {}
        self.similarity_cache = {}
        self.embedding_cache = {}

    def _encode_text(self, text: str) -> 'torch.Tensor':
        """Encode text with caching to avoid recomputation."""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.similarity_model.encode(
                text, convert_to_tensor=True, show_progress_bar=False
            )
        return self.embedding_cache[text]

    def _preprocess_relation(self, relation: str) -> str:
        """Preprocess relation text."""
        return relation.replace('.', ' ').lower()

    def get_relation_similarity_scores(self, question: str, relations: List[str]) -> List[Tuple[str, float]]:
        """Compute similarity scores with caching."""
        if not relations:
            return []
        cache_key = (question, tuple(relations))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]

        question_embedding = self._encode_text(question)
        relation_texts = [self._preprocess_relation(rel) for rel in relations]
        relation_embeddings = self.similarity_model.encode(
            relation_texts, convert_to_tensor=True, show_progress_bar=False
        )
        cosine_scores = util.pytorch_cos_sim(question_embedding, relation_embeddings)[0]
        result = [(rel, score.item()) for rel, score in zip(relations, cosine_scores)]
        self.similarity_cache[cache_key] = result
        return result

    def get_golden_path(self, start_entity: str, answer_entity: str) -> List[Tuple[str, str, str]]:
        """Retrieve the shortest path between entities."""
        shortest_paths = self.kg.get_shortest_paths(start_entity, answer_entity)
        if shortest_paths and len(shortest_paths) > 0:
            return [(d['source'], d['relation'], d['target']) for d in shortest_paths[0]]
        return []

    def _get_cached_relations(self, entity: str) -> List[str]:
        """Fetch relations for an entity with caching."""
        if entity not in self.relation_cache:
            self.relation_cache[entity] = self.kg.get_connected_relations(entity)
        return self.relation_cache[entity]

    def get_positive_path(
        self,
        start_entity: str,
        answer_entity: str,
        question: str,
        initial_top_k: int = 5,
        max_top_k: int = 50,
        top_k_step: int = 10
    ) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """Generate a semantic path aligned with the question."""
        current_top_k = initial_top_k
        while current_top_k <= max_top_k:
            visited = set()
            queue = deque([(start_entity, [], set(), 0)])
            while queue:
                current_entity, current_path, current_visited, backtrack_count = queue.popleft()
                if backtrack_count > self.max_backtrack or len(current_path) >= self.max_path_length:
                    continue

                relations = self._get_cached_relations(current_entity)
                if not relations:
                    if backtrack_count < self.max_backtrack:
                        queue.append((current_entity, current_path, current_visited, backtrack_count + 1))
                    continue

                relation_scores = self.get_relation_similarity_scores(question, relations)
                relation_scores.sort(key=lambda x: x[1], reverse=True)
                top_relations = relation_scores[:current_top_k]

                for relation, _ in top_relations:
                    tail_entities = self.kg.get_target_entities(current_entity, relation)
                    for tail_entity in tail_entities:
                        if tail_entity in current_visited:
                            continue
                        new_path = current_path + [(current_entity, relation, tail_entity)]
                        new_visited = current_visited | {tail_entity}
                        if tail_entity == answer_entity:
                            return new_path, new_visited
                        queue.append((tail_entity, new_path, new_visited, backtrack_count))
            current_top_k += top_k_step
        return [], set()

    def get_negative_paths(
        self,
        positive_path: List[Tuple[str, str, str]],
        question: str,
        answer_entity: str,
        max_negatives_per_pair: int = 5
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Generate negative paths by perturbing the positive path efficiently.
        Computes relation similarities once per source entity and reuses them.
        """
        if not positive_path:
            return []

        negative_paths = []
        # Precompute positive path targets to avoid repeated checks
        positive_targets = {tgt for _, _, tgt in positive_path}
        # Cache for relation scores per source entity
        relation_scores_cache = {}

        # Iterate over hops in reverse order
        for hop_idx in range(len(positive_path) - 1, -1, -1):
            if len(negative_paths) >= max_negatives_per_pair:
                break

            src, rel, _ = positive_path[hop_idx]
            # Get or compute relation scores for this source entity
            if src not in relation_scores_cache:
                relations = self._get_cached_relations(src)
                if not relations:
                    continue
                other_relations = [r for r in relations if r != rel]
                if not other_relations:
                    continue
                scores = self.get_relation_similarity_scores(question, other_relations)
                # Sort ascending to get least similar relations
                relation_scores_cache[src] = sorted(scores, key=lambda x: x[1])
            else:
                other_relations = [r for r, _ in relation_scores_cache[src]]

            if not relation_scores_cache[src]:
                continue

            # Select least similar relations (up to a quarter of available relations)
            candidate_relations = [r for r, _ in relation_scores_cache[src][:max(1, len(relation_scores_cache[src]) // 4)]]
            
            # Collect all valid tail entities across candidate relations
            all_valid_tails = {}
            for new_rel in candidate_relations:
                tail_entities = self.kg.get_target_entities(src, new_rel)
                valid_tails = [t for t in tail_entities if t != answer_entity and t not in positive_targets]
                if valid_tails:
                    all_valid_tails[new_rel] = valid_tails

            # If no valid tails, skip to next hop
            if not all_valid_tails:
                continue

            # Calculate how many negative paths we still need
            remaining_needed = max_negatives_per_pair - len(negative_paths)
            if remaining_needed <= 0:
                break

            # Flatten and sample from all valid tails
            flat_tails = [(new_rel, tail) for new_rel, tails in all_valid_tails.items() for tail in tails]
            num_to_sample = min(remaining_needed, len(flat_tails))
            if num_to_sample > 0:
                sampled_tails = random.sample(flat_tails, num_to_sample)
                prefix = positive_path[:hop_idx]
                for new_rel, new_tgt in sampled_tails:
                    negative_paths.append(prefix + [(src, new_rel, new_tgt)])

        return negative_paths
    
def format_path_for_json(path: List[Tuple[str, str, str]]) -> str:
    """Format a path for JSON serialization."""
    return " ; ".join(f"{src} {rel} {tgt}" for src, rel, tgt in path)

def process_sample(sample, path_generator, max_pairs=5, max_negatives_per_pair=5):
    """Process a single dataset sample, handling multiple entities efficiently."""
    q_entities = sample['q_entity'] if isinstance(sample['q_entity'], list) else [sample['q_entity']]
    a_entities = sample.get('a_entity', []) if isinstance(sample.get('a_entity', []), list) else [sample.get('a_entity', '')]
    
    sample_result = {
        "question": sample['question'],
        "q_entities": q_entities,
        "a_entities": a_entities,
        "paths": []
    }
    pairs = [(q, a) for q in q_entities for a in a_entities if q and a]  # Ensure entities are non-empty
    for i, (q_entity, a_entity) in enumerate(pairs):
        if i >= max_pairs:
            break
        golden_path = path_generator.get_golden_path(q_entity, a_entity)
        positive_path, _ = path_generator.get_positive_path(q_entity, a_entity, sample['question'])
        negative_paths = path_generator.get_negative_paths(positive_path, sample['question'], a_entity, max_negatives_per_pair)
        negative_paths = negative_paths[:max_negatives_per_pair]
        pair = {
            "q_entity": q_entity,
            "a_entity": a_entity,
            "golden_path": format_path_for_json(golden_path),
            "positive_path": format_path_for_json(positive_path),
            "negative_paths": [format_path_for_json(np) for np in negative_paths]
        }
        sample_result["paths"].append(pair)
    
    return sample_result

def prepare_dataset(args: argparse.Namespace):
    """Prepare the dataset with optimized processing."""
    logger.info(f"Loading dataset: {args.data_path}")
    dataset = load_dataset(args.data_path, split=args.split)

    logger.info("Initializing knowledge graph")
    kg = KnowledgeGraph(        
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
        model_name=args.model_name,
        dataset=args.dataset,
        split=args.split
    )

    logger.info(f"Loading similarity model: {args.similarity_model}")
    similarity_model = SentenceTransformer(args.similarity_model)
    path_generator = PathGenerator(
        kg=kg,
        similarity_model=similarity_model,
        max_path_length=args.max_path_length,
        top_k_relations=args.top_k_relations,
        max_backtrack=args.max_backtrack,
    )

    try:
        results = []
        for sample in tqdm(dataset, desc="Processing dataset samples"):
            result = process_sample(sample, path_generator)
            results.append(result)
            if len(results) % 1000 == 0:
                path_generator.embedding_cache.clear()
                path_generator.relation_cache.clear()
                path_generator.similarity_cache.clear()
                gc.collect()

        output_dir = os.path.join(args.output_path, args.output_name)
        os.makedirs(output_dir, exist_ok=True)

        json_output_path = os.path.join(output_dir, 'data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Dataset saved as JSON to {json_output_path}")

        processed_dataset = Dataset.from_list(results)
        processed_dataset.save_to_disk(output_dir)
        logger.info(f"Dataset saved in Dataset format to {output_dir}")

    finally:
        kg.close()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare training dataset with various path types")
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input dataset')
    parser.add_argument('--split', type=str, default='train', help='Dataset split to process')
    parser.add_argument('--output_path', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--output_name', type=str, default='path_enhanced_dataset', help='Name of the output dataset')
    parser.add_argument('--neo4j_uri', type=str, default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j_user', type=str, default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j_password', type=str, default='password', help='Neo4j password')
    parser.add_argument('--max_path_length', type=int, default=3, help='Maximum path length')
    parser.add_argument('--top_k_relations', type=int, default=5, help='Top-K relations to consider')
    parser.add_argument('--max_backtrack', type=int, default=3, help='Maximum backtracking steps')
    parser.add_argument('--similarity_model', type=str, default='sentence-transformers/msmarco-distilbert-base-tas-b', help='Sentence transformer model')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes (ignored)')
    parser.add_argument('--max_pairs', type=int, default=5, help='Maximum pairs per sample')
    parser.add_argument('--max_negatives_per_pair', type=int, default=5, help='Maximum negatives per pair')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='Pretrained model name')
    args = parser.parse_args()
    prepare_dataset(args)