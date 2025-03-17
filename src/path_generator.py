from sentence_transformers import SentenceTransformer, util
from src.knowledge_graph import KnowledgeGraph
from typing import List, Tuple, Set, Dict, Optional, Any, Union
from collections import deque, defaultdict
import random
import logging

logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(
        self,
        kg: KnowledgeGraph,
        max_path_length: int = 5,
        top_k_relations: int = 5,
    ):
        self.kg = kg
        self.max_path_length = max_path_length
        self.top_k_relations = top_k_relations
        
    def get_golden_path(self, start_entity: str, answer_entity: str) -> List[Tuple[str, str, str]]:        
        shortest_paths = self.kg.get_shortest_paths(start_entity, answer_entity)
        if shortest_paths and len(shortest_paths) > 0:
            path = [(d['source'], d['relation'], d['target']) for d in shortest_paths[0]]
            return path
        return []

    def _expand_queue(
        self,
        queue: deque,
        visited: dict,
        target_entity: str,
        question: str,
        top_k: int,
        max_depth: int
    ) -> bool:
        while queue:
            current_entity, current_path = queue.popleft()
            if len(current_path) >= max_depth:
                continue
            related_relation_scores = self.kg.get_related_relations_by_question(current_entity, question)
            candidate_relations = related_relation_scores[:min(top_k, len(related_relation_scores))]
            for relation, _ in candidate_relations:
                tail_entities = self.kg.get_target_entities(current_entity, relation, direction="out")
                for tail_entity in tail_entities:
                    if tail_entity in visited:
                        continue
                    new_path = current_path + [(current_entity, relation, tail_entity)]
                    visited[tail_entity] = new_path
                    if tail_entity == target_entity:
                        return True
                    queue.append((tail_entity, new_path))
        return False

    def get_positive_path(
        self,
        start_entity: str,
        answer_entity: str,
        question: str,
        initial_top_k: int = 5,
        max_top_k: int = 20,
        top_k_step: int = 10
    ) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        queue = deque([(start_entity, [])])
        visited = {start_entity: []}
        found_path = []
        current_top_k = min(initial_top_k, max_top_k)
        while current_top_k <= max_top_k:
            temp_queue = deque(queue)
            temp_visited = visited.copy()
            found = self._expand_queue(
                temp_queue, 
                temp_visited, 
                answer_entity, 
                question, 
                current_top_k, 
                self.max_path_length
            )
            if found:
                queue = temp_queue
                visited = temp_visited
                found_path = visited[answer_entity]
                break
            else:
                current_top_k += top_k_step
        return found_path, set(visited.keys())

    def get_negative_paths(
        self,
        positive_path: List[Tuple[str, str, str]],
        question: str,
        answer_entity: str,
        max_negatives_per_pair: int = 5
    ) -> List[List[Tuple[str, str, str]]]:
        if not positive_path:
            return []  
        negative_paths = []
        for hop_idx in range(len(positive_path) - 1, -1, -1):
            if len(negative_paths) >= max_negatives_per_pair:
                break
            src, positive_rel, tgt = positive_path[hop_idx]
            related_relation_scores = self.kg.get_related_relations_by_question(src, question)
            filtered_relations = [(rel, score) for rel, score in related_relation_scores if rel != positive_rel]
            filtered_relations.sort(key=lambda x: x[1])
            low_similarity_relations = filtered_relations[:max_negatives_per_pair]
            for new_rel, _ in low_similarity_relations:
                tail_entities = self.kg.get_target_entities(src, new_rel)
                valid_tails = [t for t in tail_entities if t != answer_entity and t != tgt]
                if not valid_tails:
                    continue
                new_tail = random.choice(valid_tails)
                negative_path = positive_path[:hop_idx] + [(src, new_rel, new_tail)]
                if hop_idx < len(positive_path) - 1:
                    current_entity = new_tail
                    for i in range(hop_idx + 1, len(positive_path)):
                        _, orig_rel, _ = positive_path[i]
                        next_entities = self.kg.get_target_entities(current_entity, orig_rel)
                        if not next_entities:
                            other_relations = self.kg.get_related_relations_by_question(current_entity, question)
                            other_relations = [r for r, _ in other_relations if r != orig_rel]
                            if not other_relations:
                                break
                            alt_rel = random.choice(other_relations)
                            next_entities = self.kg.get_target_entities(current_entity, alt_rel)
                            if not next_entities:
                                break
                            next_entity = random.choice(next_entities)
                            negative_path.append((current_entity, alt_rel, next_entity))
                        else:
                            next_entity = random.choice(next_entities)
                            negative_path.append((current_entity, orig_rel, next_entity))
                        current_entity = next_entity
                negative_paths.append(negative_path)
                if len(negative_paths) >= max_negatives_per_pair:
                    break
        return negative_paths