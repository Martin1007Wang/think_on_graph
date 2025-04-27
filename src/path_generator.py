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
        
    def get_shortest_path(self, start_entity: str, answer_entity: str) -> List[Tuple[str, str, str]]:
        shortest_paths = self.kg.get_shortest_paths(start_entity, answer_entity)
        if shortest_paths:
            path = [(d['source'], d['relation'], d['target']) for d in shortest_paths[0]]
            return path
        return []

    def get_semantic_path(self, start_entity: str, answer_entity: str, question: str, max_top_k: int = 20):
        queue = deque([(start_entity, [])])
        visited = {start_entity: []}
        
        for top_k in range(5, max_top_k + 1, 5):
            found = self._expand_queue(queue, visited, answer_entity, question, top_k=top_k, max_depth=self.max_path_length)
            if found:
                return visited[answer_entity], set(visited.keys())
        return [], set(visited.keys())

    def _expand_queue(self,queue: deque,visited: dict,target_entity: str,question: str,top_k: int,max_depth: int) -> bool:
        while queue:
            current_entity, current_path = queue.popleft()
            if len(current_path) >= max_depth:
                continue
            related_relation_scores = self.kg.get_related_relations_by_question(current_entity, question)
            candidate_relations = related_relation_scores[:min(top_k, len(related_relation_scores))]
            for relation, score in candidate_relations:
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

    def get_negative_paths(self, positive_path: List[Tuple[str, str, str]], question: str, answer_entity: str, max_negatives_per_pair: int = 5) -> List[List[Tuple[str, str, str]]]:
        if not positive_path:
            return []
        negative_paths = []
        relation_cache = {}
        for hop_idx in range(len(positive_path) - 1, -1, -1):
            if len(negative_paths) >= max_negatives_per_pair:
                break
            src, positive_rel, tgt = positive_path[hop_idx]
            if src not in relation_cache:
                relation_cache[src] = self.kg.get_related_relations_by_question(src, question)
            filtered_relations = [(rel, score) for rel, score in relation_cache[src] if rel != positive_rel]
            mid_idx = len(filtered_relations) // 2
            candidate_relations = filtered_relations[mid_idx-max_negatives_per_pair//2:mid_idx+max_negatives_per_pair//2+1]
            if not candidate_relations:
                candidate_relations = filtered_relations[:max_negatives_per_pair]
            for new_rel, _ in candidate_relations:
                tail_entities = self.kg.get_target_entities(src, new_rel)
                valid_tails = [t for t in tail_entities if t != answer_entity and t != tgt]
                if not valid_tails:
                    continue
                new_tail = valid_tails[0] if valid_tails else None
                if not new_tail:
                    continue
                negative_path = positive_path[:hop_idx] + [(src, new_rel, new_tail)]
                if hop_idx < len(positive_path) - 1:
                    negative_path = self._simplified_complete_path(
                        negative_path[-1][2],
                        remaining_steps=len(positive_path) - hop_idx - 1,
                        question=question)
                    
                negative_paths.append(negative_path)
                if len(negative_paths) >= max_negatives_per_pair:
                    break
                    
        return negative_paths

    def _simplified_complete_path(self, start_entity: str, remaining_steps: int, question: str) -> List[Tuple[str, str, str]]:
        path = []
        current_entity = start_entity
        for _ in range(remaining_steps):
            if not current_entity:
                break
            relations = self.kg.get_related_relations_by_question(current_entity, question)
            if not relations:
                break
            relation = relations[0][0]
            targets = self.kg.get_target_entities(current_entity, relation)
            if not targets:
                break
            target = targets[0]
            path.append((current_entity, relation, target))
            current_entity = target
        return path
