from sentence_transformers import SentenceTransformer, util
from src.knowledge_graph import KnowledgeGraph
from typing import List, Tuple, Set, Dict, Optional, Any, Union
from collections import deque, defaultdict
import heapq
import random
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class PathGenerator:
    def __init__(
        self,
        kg: KnowledgeGraph,
        max_path_length: int = 3,
        top_k_relations: int = 5,
        cache_size: int = 1000
    ):
        """
        Initialize PathGenerator with KnowledgeGraph and parameters.
        
        Args:
            kg: Knowledge graph instance to query
            max_path_length: Maximum length of generated paths
            top_k_relations: Default number of top relations to consider
            cache_size: Size of the LRU cache for relation and path queries
        """
        if not isinstance(kg, KnowledgeGraph):
            raise TypeError("kg must be an instance of KnowledgeGraph")
        if max_path_length <= 0:
            raise ValueError("max_path_length must be positive")
        if top_k_relations <= 0:
            raise ValueError("top_k_relations must be positive")
            
        self.kg = kg
        self.max_path_length = max_path_length
        self.top_k_relations = top_k_relations
        self.relation_cache = {}
        self.path_cache = {}
        
        # Configure LRU cache for methods
        self._get_related_relations = lru_cache(maxsize=cache_size)(self._get_related_relations_impl)
        
    def get_golden_path(self, start_entity: str, answer_entity: str) -> List[Tuple[str, str, str]]:
        """
        Retrieve the shortest path between entities.
        
        Args:
            start_entity: The source entity ID
            answer_entity: The target entity ID
            
        Returns:
            A list of (source, relation, target) tuples representing the path
        """
        if not start_entity or not answer_entity:
            logger.warning("Invalid entities for golden path: empty entity ID")
            return []
            
        cache_key = (start_entity, answer_entity, "golden")
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        try:
            shortest_paths = self.kg.get_shortest_paths(start_entity, answer_entity)
            if shortest_paths and len(shortest_paths) > 0:
                path = [(d['source'], d['relation'], d['target']) for d in shortest_paths[0]]
                self.path_cache[cache_key] = path
                return path
        except ConnectionError as e:
            logger.error(f"Connection error getting golden path: {e}")
        except ValueError as e:
            logger.error(f"Value error getting golden path: {e}")
        except Exception as e:
            logger.error(f"Unexpected error getting golden path: {e}", exc_info=True)
        
        self.path_cache[cache_key] = []
        return []

    def _get_related_relations_impl(self, entity: str, question: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Implementation of get_related_relations (wrapped with LRU cache).
        
        Args:
            entity: The entity ID
            question: The question text
            top_k: Number of top relations to retrieve
            
        Returns:
            List of (relation, score) tuples
        """
        if not entity or not question:
            return []
            
        try:
            return self.kg.get_related_relations_by_question(entity, question, top_k=top_k)
        except Exception as e:
            logger.error(f"Error getting related relations for {entity}: {e}")
            return []

    def _get_target_entities(self, entity: str, relation: str) -> List[str]:
        """
        Get target entities for a given entity and relation.
        
        Args:
            entity: The source entity ID
            relation: The relation type
            
        Returns:
            List of target entity IDs
        """
        if not entity or not relation:
            return []
            
        cache_key = (entity, relation, "targets")
        if cache_key in self.relation_cache:
            return self.relation_cache[cache_key]
            
        try:
            targets = self.kg.get_target_entities(entity, relation)
            self.relation_cache[cache_key] = targets
            return targets
        except Exception as e:
            logger.error(f"Error getting target entities for {entity} and {relation}: {e}")
            return []

    def get_positive_path(
        self,
        start_entity: str,
        answer_entity: str,
        question: str,
        initial_top_k: int = 5,
        max_top_k: int = 20,
        top_k_step: int = 5
    ) -> Tuple[List[Tuple[str, str, str]], Set[str]]:
        """
        Find a path from start_entity to answer_entity that is relevant to the question.
        
        Args:
            start_entity: The source entity ID
            answer_entity: The target entity ID
            question: The question text
            initial_top_k: Initial number of top relations to consider
            max_top_k: Maximum number of top relations to consider
            top_k_step: Step size for increasing top_k
            
        Returns:
            Tuple of (path, visited_entities) where:
            - path is a list of (source, relation, target) tuples
            - visited_entities is a set of entity IDs that were visited during search
        """
        # Validate inputs
        if not start_entity or not answer_entity or not question:
            logger.warning("Invalid input for positive path generation: empty entity or question")
            return [], set()
            
        if initial_top_k <= 0 or max_top_k <= 0 or top_k_step <= 0:
            logger.warning(f"Invalid numerical parameters for positive path generation: {initial_top_k}, {max_top_k}, {top_k_step}")
            return [], set()
            
        if initial_top_k > max_top_k:
            logger.warning("initial_top_k should not be greater than max_top_k")
            initial_top_k = max_top_k
            
        # Check cache
        cache_key = (start_entity, answer_entity, question, "positive")
        if cache_key in self.path_cache:
            path, visited = self.path_cache[cache_key]
            return path, visited
            
        # Initialize search
        current_top_k = min(initial_top_k, max_top_k)
        visited = {start_entity}
        queue = deque([(start_entity, [], 0)])  # (entity, path, depth)
        
        # Iteratively search with increasing top_k
        while current_top_k <= max_top_k:
            temp_queue = deque(queue)  # Copy the queue for this iteration
            temp_visited = visited.copy()
            
            while temp_queue:
                current_entity, current_path, depth = temp_queue.popleft()
                
                # Skip if we've reached max depth
                if depth >= self.max_path_length:
                    continue
                    
                # Get related relations
                relation_scores = self._get_related_relations(current_entity, question, top_k=current_top_k)
                if not relation_scores:
                    continue
                    
                # Explore each relation
                for relation, _ in relation_scores:
                    tail_entities = self._get_target_entities(current_entity, relation)
                    
                    for tail_entity in tail_entities:
                        if tail_entity in temp_visited:
                            continue
                            
                        new_path = current_path + [(current_entity, relation, tail_entity)]
                        temp_visited.add(tail_entity)
                        
                        # Check if we found the answer
                        if tail_entity == answer_entity:
                            self.path_cache[cache_key] = (new_path, temp_visited)
                            return new_path, temp_visited
                            
                        temp_queue.append((tail_entity, new_path, depth + 1))
            
            # If queue is empty and no path found, increase top_k
            if not temp_queue:
                current_top_k += top_k_step
                # Keep the original queue and visited set for the next iteration
            else:
                # Update the queue and visited set for the next iteration
                queue = temp_queue
                visited = temp_visited

        logger.warning(f"No positive path found from {start_entity} to {answer_entity}")
        self.path_cache[cache_key] = ([], visited)
        return [], visited

    def get_negative_paths(
        self,
        positive_path: List[Tuple[str, str, str]],
        question: str,
        answer_entity: str,
        max_negatives_per_pair: int = 5
    ) -> List[List[Tuple[str, str, str]]]:
        """
        Generate negative paths by modifying the positive path.
        
        Args:
            positive_path: The positive path as a list of (source, relation, target) tuples
            question: The question text
            answer_entity: The correct answer entity ID
            max_negatives_per_pair: Maximum number of negative paths to generate
            
        Returns:
            List of negative paths, where each path is a list of (source, relation, target) tuples
        """
        if not positive_path:
            logger.warning("Positive path is empty, returning empty negative paths.")
            return []
            
        # Check cache
        cache_key = (tuple(map(tuple, positive_path)), question, answer_entity, max_negatives_per_pair, "negative")
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
            
        negative_paths = []
        positive_targets = {tgt for _, _, tgt in positive_path}
        
        # Start from the end of the path for more relevant negatives
        for hop_idx in range(len(positive_path) - 1, -1, -1):
            if len(negative_paths) >= max_negatives_per_pair:
                break
                
            src, positive_rel, _ = positive_path[hop_idx]
            
            # Get relations sorted by relevance to the question
            all_relations_scores = self._get_related_relations(src, question, top_k=self.top_k_relations * 2)
            
            # Filter out the positive relation and sort by least similar first
            least_similar_relations = sorted(
                [(rel, score) for rel, score in all_relations_scores if rel != positive_rel],
                key=lambda x: x[1]  # Least similar first
            )
            
            # Select a subset of candidate relations
            num_candidates = min(max(1, len(least_similar_relations) // 4), max_negatives_per_pair)
            candidate_relations = [rel for rel, _ in least_similar_relations[:num_candidates]]
            
            # Get valid tail entities for each candidate relation
            valid_tails = {}
            for new_rel in candidate_relations:
                tail_entities = self._get_target_entities(src, new_rel)
                valid_tails[new_rel] = [
                    t for t in tail_entities if t != answer_entity and t not in positive_targets
                ]
            
            # Flatten the tails list
            flat_tails = [(rel, tail) for rel, tails in valid_tails.items() for tail in tails if tails]
            if not flat_tails:
                continue
            
            # Sample negative paths
            remaining_needed = max_negatives_per_pair - len(negative_paths)
            num_to_sample = min(remaining_needed, len(flat_tails))
            if num_to_sample > 0:
                sampled_tails = random.sample(flat_tails, num_to_sample)
                prefix = positive_path[:hop_idx]
                for new_rel, new_tail in sampled_tails:
                    negative_path = prefix + [(src, new_rel, new_tail)]
                    negative_paths.append(negative_path)
        
        if not negative_paths:
            logger.warning(f"No negative paths generated for positive path: {positive_path}")
        
        self.path_cache[cache_key] = negative_paths
        return negative_paths
        
    def clear_caches(self):
        """Clear all internal caches."""
        self.relation_cache.clear()
        self.path_cache.clear()
        self._get_related_relations.cache_clear()